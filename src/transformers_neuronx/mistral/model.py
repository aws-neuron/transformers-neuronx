# Copyright Amazon Web Services and its Affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import torch
import os
from transformers_neuronx import decoder
from transformers_neuronx import module
from transformers_neuronx import ops
from transformers_neuronx import sampling
from transformers_neuronx import utils
from transformers_neuronx import bucket
from transformers_neuronx import base
from transformers_neuronx.constants import LAYOUT_BSH
from transformers_neuronx.config import NeuronConfig
from transformers_neuronx.mistral.config import MistralConfig
from transformers_neuronx.mistral.modules import MistralForCausalLM
from transformers_neuronx.mistral.hlo import MistralForSamplingNoEmbeddingHlo


class MistralForSampling(base.NeuronModelBase):

    def __init__(self, config, *, n_positions=2048, batch_size=1, amp='f32', tp_degree=2,
                 context_length_estimate=None, context_unroll=None, unroll=None,
                 neuron_config=None, **kwargs):
        config = MistralConfig(config, n_positions, batch_size, amp, tp_degree)
        super().__init__(MistralForCausalLM, config)
        self.context_pre_hook = None
        self.context_hook = None
        self.config = config
        self.neuron_config = neuron_config
        if context_unroll is None:
            context_unroll = config.num_hidden_layers
        self.context_unroll = context_unroll

        if unroll is None:
            unroll = config.num_hidden_layers

        self.unroll=unroll
        self.token_buckets = bucket.token_sizes(n_positions)
        self.context_buckets = bucket.context_sizes(context_length_estimate, self.token_buckets)
        self.max_positions = self.token_buckets[-1]

        if isinstance(batch_size,int):
            self.batch_sizes = [batch_size]
        elif isinstance(batch_size,list):
            self.batch_sizes = sorted(batch_size)
        else:
            raise TypeError("batch_size must be list of ints or int type")
        self.context_batch_sizes = [1] if self.neuron_config and self.neuron_config.continuous_batching else self.batch_sizes
        hlo_builder = MistralForSamplingNoEmbeddingHlo(config, neuron_config=neuron_config)
        self.decoder_param_set = decoder.DecoderLmHeadForSamplingNoEmbedding(
            tp_degree=tp_degree, n_positions_list=self.token_buckets, n_active_tokens=1, batch_size=self.batch_sizes,
            attention_head_size=config.attention_head_size, amp=amp,
            num_layers=config.num_hidden_layers, n_head=config.num_attention_heads, n_kv_head=config.num_key_value_heads,
            unroll=unroll, neuron_config=neuron_config, allow_pad=True,
            builder=hlo_builder
        )
        self.decoder_lm_head_for_context= self.decoder_param_set.init_context_decoder(unroll=self.context_unroll, buckets=self.context_buckets, model_obj=self)
        self.decoder_lm_head= self.decoder_param_set.init_token_decoder(unroll=self.unroll, buckets=self.token_buckets, model_obj=self)

        # Track number of processed tokens for sliding window attention
        self.num_processed_tokens = 0

    def load_weights(self):

        # Materialize the embedding to CPU
        self.chkpt_model.model.embed_tokens.materialize()

        ops.init()

        for layer in self.chkpt_model.model.layers:
            layer.materialize()
            attn = layer.self_attn
            mlp = layer.mlp
            new_layer = self.decoder_lm_head.new_layer()
            new_layer.add_pre_attention_layer_norm(layer.input_layernorm.weight.detach(), None)
            new_layer.add_attention_query(attn.q_proj.weight.detach().T, None)
            new_layer.add_attention_key(attn.k_proj.weight.detach().T, None)
            new_layer.add_attention_value(attn.v_proj.weight.detach().T, None)
            if self.neuron_config and self.neuron_config.attention_layout == LAYOUT_BSH:
                new_layer.add_attention_output(attn.o_proj.weight.detach().T, None, sharding=0, transposed=True)
            else:
                new_layer.add_attention_output(attn.o_proj.weight.detach(), None, sharding=1, transposed=False)
            new_layer.add_pre_mlp_layer_norm(layer.post_attention_layernorm.weight.detach(), None)

            # Note: Automatic MLP padding is safe since zeros are *only* introduced to intermediary state
            new_layer.add_parameter(mlp.gate_proj.weight.T, sharding=1, allow_pad=True,
                                    allow_quantize=True, allow_transform=True)
            new_layer.add_parameter(mlp.up_proj.weight.T, sharding=1, allow_pad=True,
                                    allow_quantize=True, allow_transform=True)
            if os.environ.get("NEURON_INTERNAL_TRANSFORM_WEIGHT_LAYOUT", None):
                new_layer.add_parameter(mlp.down_proj.weight.T, sharding=0, allow_pad=True,
                                        allow_quantize=True, allow_transform=True)
            else:
                new_layer.add_parameter(mlp.down_proj.weight, sharding=1, allow_pad=True,
                                        allow_quantize=True, out_feature_dim=0)

            new_layer.to_neuron()
            layer.nullify()

        ln_f = self.chkpt_model.model.norm
        ln_f.materialize()
        self.decoder_lm_head.add_final_layer_norm(ln_f.weight.detach(), None)

        lm_head = self.chkpt_model.lm_head
        lm_head.materialize()
        self.decoder_lm_head.add_lm_head(lm_head.weight.detach().T)
        lm_head.nullify()
        self.decoder_lm_head.to_neuron()
        self.decoder_lm_head.use_executor = True

        if self.context_buckets:
            for context_length_estimate in self.context_buckets:
                for batch_size in self.context_batch_sizes:
                    model = self.decoder_lm_head.build_weight_shared(share_caches=True,
                                                                     new=self.decoder_lm_head_for_context[context_length_estimate, batch_size])
                    # PERF: No latency improvement seen in multi-layer models from executor
                    if self.context_unroll == self.config.num_hidden_layers:
                        model.use_executor = True
                    self.decoder_lm_head_for_context[context_length_estimate,batch_size] = model

    def forward(self, input_ids, cache_ids=None, start_ids=None):
        # Compute the window starting index for specific mask patterns
        # For other patterns we pass in a default value of 0, it won't be used
        curr_window_start = max(0, self.num_processed_tokens - self.config.window_size)
        curr_window_start = torch.as_tensor(curr_window_start, dtype=torch.int32)

        input_ids, cache_ids, start_ids, last_token_id = self._preprocess(input_ids, start_ids=start_ids, cache_ids=cache_ids)
        hidden = self.chkpt_model.model.embed_tokens(input_ids)
        is_bsh = self.neuron_config and self.neuron_config.attention_layout == LAYOUT_BSH
        if is_bsh:
            hidden = hidden.permute(2, 1, 0)
        logits = self._forward(hidden, cache_ids, start_ids, last_token_id, curr_window_start, neuron_config=self.neuron_config)

        # Increment the token counter, last_token_id = 0 when in decoder mode
        self.num_processed_tokens += (last_token_id+1)
        return logits

    def sample(self, input_ids, sequence_length, start_ids=None,
               top_k=50, top_p=1.0, eos_token_override=None, temperature=1.0, streamer=None, stopping_criteria_list=None):

        if self.context_pre_hook is not None:
            self.context_pre_hook()
        batch_size, context_length = input_ids.shape
        if batch_size not in self.batch_sizes:
            raise ValueError(f"Model not compiled for batch_size : {batch_size}. Acceptable batch_size is one of the following {self.batch_sizes}")

        result = sampling.sample_llama(
            self, input_ids, start_ids, sequence_length,
            eos_token_id=self.config.eos_token_id if eos_token_override is None else eos_token_override,
            top_k=top_k, top_p=top_p, temperature=temperature, streamer=streamer,
            stopping_criteria_list=stopping_criteria_list
        )

        return result
