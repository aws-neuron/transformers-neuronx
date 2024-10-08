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
from transformers_neuronx.constants import LAYOUT_HSB, LAYOUT_BSH
from transformers_neuronx.config import NeuronConfig
from transformers_neuronx.mixtral.config import MixtralConfig
from transformers_neuronx.mixtral.modules import MixtralForCausalLM
from transformers_neuronx.mixtral.hlo import MixtralForSamplingNoEmbeddingHlo


class MixtralForSampling(base.NeuronModelBase):

    def __init__(self, config, *, n_positions=2048, batch_size=1, amp='f32', tp_degree=2,
                 context_length_estimate=None, context_unroll=None, unroll=None,
                 neuron_config=NeuronConfig(), **kwargs):
        config = MixtralConfig(config, n_positions, batch_size, amp, tp_degree)
        super().__init__(MixtralForCausalLM, config)

        self.context_pre_hook = None
        self.context_hook = None
        self.config = config
        self.neuron_config = neuron_config if neuron_config else NeuronConfig()
        if self.neuron_config.on_device_generation:
            self.neuron_config.on_device_generation.vocab_size = self.config.vocab_size
        if context_unroll is None:
            context_unroll = config.num_hidden_layers
        self.context_unroll = context_unroll

        if unroll is None:
            unroll = config.num_hidden_layers
        self.unroll=unroll

        self.token_buckets = bucket.token_sizes(n_positions)
        self.context_buckets = bucket.context_sizes(context_length_estimate, self.token_buckets)
        self.batch_sizes = bucket.batch_sizes(batch_size)

        self.context_batch_sizes = [1] if self.neuron_config and self.neuron_config.continuous_batching else self.batch_sizes
        hlo_builder = MixtralForSamplingNoEmbeddingHlo(config, neuron_config=self.neuron_config)
        self.decoder_param_set = decoder.DecoderLmHeadForSamplingNoEmbedding(
            tp_degree=tp_degree, n_positions_list=self.token_buckets, n_active_tokens=1, batch_size=self.batch_sizes,
            attention_head_size=config.attention_head_size, amp=amp,
            num_layers=config.num_hidden_layers, n_head=config.num_attention_heads, n_kv_head=config.num_key_value_heads,
            unroll=unroll, neuron_config=self.neuron_config, allow_pad=True,
            builder=hlo_builder
        )
        self.decoder_lm_head = self.decoder_param_set.init_token_decoder(unroll=self.unroll, buckets=self.token_buckets, model_obj=self)
        self.decoder_lm_head_for_context = self.decoder_param_set.init_context_decoder(unroll=self.context_unroll, buckets=self.context_buckets, model_obj=self)

    def load_weights(self):
        self.materialize_embeddings()

        for layer in self.chkpt_model.model.layers:
            layer.materialize()
            attn = layer.self_attn
            block_sparse_moe = layer.block_sparse_moe
            new_layer = self.decoder_lm_head.new_layer()
            new_layer.add_pre_attention_layer_norm(layer.input_layernorm.weight.detach(), None)
            new_layer.add_attention_query(attn.q_proj.weight.detach().T, None)
            new_layer.add_attention_key(attn.k_proj.weight.detach().T, None)
            new_layer.add_attention_value(attn.v_proj.weight.detach().T, None)
            new_layer.add_attention_output(attn.o_proj.weight.detach(), None, sharding=1, transposed=False)
            new_layer.add_pre_mlp_layer_norm(layer.post_attention_layernorm.weight.detach(), None)

            # Note:
            # 1. Automatic MLP padding is safe since zeros are *only* introduced to intermediary state
            # 2. Gating network is replicated and distributed across all TP degrees
            # 3. Concatenate weights of all experts
            # 4. Expert parallel is used. Each expert is distributed to a small group of Neuron devices, to reduce collective communication.
            expert_indices = torch.tensor(range(self.config.num_local_experts))
            expert_indices = expert_indices.repeat_interleave(max(1, self.config.tp_degree // self.config.num_local_experts)) # repeat when tp > num of experts
            new_layer.add_parameter(expert_indices, sharding=0, allow_pad=False, allow_quantize=False)
            new_layer.add_parameter(block_sparse_moe.gate.weight.detach().T, None)
            w1, w2, w3 = [], [], []
            for mlp in block_sparse_moe.experts:
                w1.append(mlp.w1.weight.T)  # gate_proj
                w2.append(mlp.w2.weight)  # down_proj (not transposed for HSB layout)
                w3.append(mlp.w3.weight.T)  # up_proj
            w1_concat = torch.concat(w1, dim=1)
            w2_concat = torch.concat(w2, dim=1)
            w3_concat = torch.concat(w3, dim=1)
            new_layer.add_parameter(w1_concat, sharding=1, allow_pad=True,  # gate_proj
                                    allow_quantize=True, allow_transform=True)
            new_layer.add_parameter(w2_concat, sharding=1, allow_pad=True,  # down_proj
                                    allow_quantize=True, allow_transform=True)
            new_layer.add_parameter(w3_concat, sharding=1, allow_pad=True,  # up_proj
                                    allow_quantize=True, allow_transform=True)

            
            new_layer.to_neuron()
            
            layer.nullify()

        ln_f = self.chkpt_model.model.norm
        ln_f.materialize()
        self.decoder_lm_head.add_final_layer_norm(ln_f.weight.detach(), None)

        lm_head = self.chkpt_model.lm_head
        lm_head.materialize()
        self.decoder_lm_head.add_lm_head(lm_head.weight.detach().T)
        if self.neuron_config.on_device_embedding:
            self.decoder_lm_head.add_pre_layer_parameter(self.chkpt_model.model.embed_tokens.weight, sharding=1, allow_pad=True)
        lm_head.nullify()

        self.decoder_lm_head.to_neuron()
        self.init_rest_of_model()

    def materialize_embeddings(self):
        # Materialize the embedding to CPU
        self.chkpt_model.model.embed_tokens.materialize()
    
    def init_rest_of_model(self):
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
        inputs, *rst = self._preprocess(input_ids, start_ids=start_ids, cache_ids=cache_ids)
        if not self.neuron_config.on_device_embedding:
            inputs = self.chkpt_model.model.embed_tokens(inputs)
            if self.neuron_config.attention_layout == LAYOUT_HSB:
                inputs = inputs.transpose(0, -1).contiguous()
        logits = self._forward(inputs, *rst)
        logits = self._postprocess(input_ids, logits, start_ids=start_ids)

        return logits

    def sample(self, input_ids, sequence_length, start_ids=None,
               top_k=50, top_p=1.0, eos_token_override=None, temperature=1.0, streamer=None, stopping_criteria_list=None):

        if self.neuron_config.on_device_generation:
            return sampling.sample_tokens(self, input_ids, start_ids, sequence_length=sequence_length,
                                            config=self.neuron_config.on_device_generation, streamer=streamer)

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
