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
import math
import warnings

from transformers_neuronx import decoder
from transformers_neuronx import module
from transformers_neuronx import ops
from transformers_neuronx import sampling
from transformers_neuronx import utils
from transformers_neuronx import bucket
from transformers_neuronx import base
from transformers_neuronx.constants import LAYOUT_BSH, LAYOUT_HSB
from transformers_neuronx.config import NeuronConfig
from transformers_neuronx.layers import alibi
from transformers_neuronx.bloom.config import BloomConfig
from transformers_neuronx.bloom.modules import BloomForCausalLM
from transformers_neuronx.bloom.hlo import BloomForSamplingNoEmbeddingHlo


class BloomForSampling(base.NeuronModelBase):

    def __init__(self, config, *, n_positions=2048, batch_size=1, amp='f32', tp_degree=2,
                 context_length_estimate=None, context_unroll=None,
                 unroll=None, neuron_config=None, **kwargs):
        config = BloomConfig(config, n_positions, batch_size, amp, tp_degree, **kwargs)
        super().__init__(BloomForCausalLM, config)
        self.config = config
        self.neuron_config = neuron_config if neuron_config else NeuronConfig()

        if self.neuron_config.on_device_generation:
            self.neuron_config.on_device_generation.vocab_size = self.config.vocab_size

        self.context_length_estimate = context_length_estimate
        if context_unroll is None:
            context_unroll = config.n_layer
        self.context_unroll = context_unroll

        if unroll is None:
            unroll = config.n_layer
        self.unroll = unroll

        self.token_buckets = bucket.token_sizes(n_positions)
        self.context_buckets = bucket.context_sizes(context_length_estimate, self.token_buckets)
        self.max_positions = self.token_buckets[-1]

        if isinstance(batch_size,int):
            self.batch_sizes = [batch_size]
        elif isinstance(batch_size,list):
            self.batch_sizes = sorted(batch_size)
        else:
            raise TypeError("batch_size must be list of ints or int type")

        hlo_builder = BloomForSamplingNoEmbeddingHlo(config, neuron_config=self.neuron_config)
        self.decoder_param_set = decoder.DecoderLmHeadForSamplingNoEmbedding(
            tp_degree=tp_degree, n_positions_list=self.token_buckets, n_active_tokens=1, batch_size=self.batch_sizes,
            attention_head_size=config.attention_head_size, amp=amp,num_layers=config.n_layer, n_head=config.n_head,
            unroll=unroll, neuron_config=self.neuron_config, allow_pad=True, builder=hlo_builder
        )
        self.decoder_lm_head_for_context = self.decoder_param_set.init_context_decoder(unroll=self.context_unroll, buckets=self.context_buckets, model_obj=self)
        self.decoder_lm_head = self.decoder_param_set.init_token_decoder(unroll=self.unroll, buckets=self.token_buckets, model_obj=self)

    def load_weights(self):
        # Materialize the embedding to CPU
        self.chkpt_model.transformer.word_embeddings.materialize()
        self.chkpt_model.transformer.word_embeddings_layernorm.materialize()

        ops.init()

        n_head = self.config.n_head
        hidden_size = self.config.hidden_size
        for layer in self.chkpt_model.transformer.h:
            layer.materialize()
            attn = layer.self_attention
            mlp = layer.mlp

            new_layer = self.decoder_lm_head.new_layer()
            new_layer.add_pre_attention_layer_norm(
                layer.input_layernorm.weight.detach(),
                layer.input_layernorm.bias.detach()
            )

            qkv = attn.query_key_value.weight
            qkv = qkv.reshape(n_head, 3, hidden_size // n_head, hidden_size).transpose(1, 0)
            qkv = qkv.reshape(3, hidden_size, hidden_size)

            qkv_bias = attn.query_key_value.bias
            qkv_bias = qkv_bias.reshape(n_head, 3, hidden_size // n_head).transpose(1, 0)
            qkv_bias = qkv_bias.reshape(3, hidden_size)

            q = qkv[0].T
            k = qkv[1].T
            v = qkv[2].T

            q_bias = qkv_bias[0]
            k_bias = qkv_bias[1]
            v_bias = qkv_bias[2]

            new_layer.add_attention_query(q, q_bias)
            new_layer.add_attention_key(k, k_bias)
            new_layer.add_attention_value(v, v_bias)

            is_bsh = self.neuron_config and self.neuron_config.attention_layout == LAYOUT_BSH
            if is_bsh:
                new_layer.add_attention_output(
                    attn.dense.weight.detach().T,
                    attn.dense.bias.detach(),
                    sharding=0,
                )
            else:
                new_layer.add_attention_output(
                    attn.dense.weight.detach(),
                    attn.dense.bias.detach(),
                    sharding=1,
                )
            new_layer.add_pre_mlp_layer_norm(
                layer.post_attention_layernorm.weight.detach(),
                layer.post_attention_layernorm.bias.detach()
            )
            new_layer.add_mlp_input(
                mlp.dense_h_to_4h.weight.detach().T,
                mlp.dense_h_to_4h.bias.detach()
            )
            if is_bsh:
                new_layer.add_mlp_output(
                    mlp.dense_4h_to_h.weight.detach().T,
                    mlp.dense_4h_to_h.bias.detach(),
                    sharding=0,
                    transposed=True,
                )
            else:
                new_layer.add_mlp_output(
                    mlp.dense_4h_to_h.weight.detach(),
                    mlp.dense_4h_to_h.bias.detach(),
                    sharding=1,
                    transposed=False,
                )
            new_layer.to_neuron()
            layer.nullify()

        ln_f = self.chkpt_model.transformer.ln_f
        ln_f.materialize()
        self.decoder_lm_head.add_final_layer_norm(ln_f.weight.detach(), ln_f.bias.detach())

        lm_head = self.chkpt_model.lm_head
        lm_head.materialize()
        self.decoder_lm_head.add_lm_head(lm_head.weight.detach().T)
        lm_head.nullify()
        slopes = alibi.build_slopes(self.config.n_head)
        self.decoder_lm_head.add_pre_layer_parameter(slopes, sharding=0, allow_pad=True)
        if self.neuron_config.on_device_embedding:
            self.decoder_lm_head.add_pre_layer_parameter(self.chkpt_model.transformer.word_embeddings.weight, sharding=1, allow_pad=True)
            self.decoder_lm_head.add_pre_layer_parameter(self.chkpt_model.transformer.word_embeddings_layernorm.weight)
            self.decoder_lm_head.add_pre_layer_parameter(self.chkpt_model.transformer.word_embeddings_layernorm.bias)
        self.decoder_lm_head.to_neuron()

        if self.context_buckets:
            for context_length_estimate in self.context_buckets:
                for batch_size in self.batch_sizes:
                    model = self.decoder_lm_head.build_weight_shared(new=self.decoder_lm_head_for_context[context_length_estimate, batch_size], share_caches=True)
                    self.decoder_lm_head_for_context[context_length_estimate, batch_size] = model

    def forward(self, input_ids, cache_ids=None, start_ids=None):
        inputs, *rst = self._preprocess(input_ids, start_ids=start_ids, cache_ids=cache_ids)
        if not self.neuron_config.on_device_embedding:
            inputs = self.chkpt_model.transformer.word_embeddings(inputs)
            inputs = self.chkpt_model.transformer.word_embeddings_layernorm(inputs)
            if self.neuron_config.attention_layout == LAYOUT_HSB:
                inputs = inputs.transpose(0, -1).contiguous()
        return self._forward(inputs, *rst)

    def sample(self, input_ids, sequence_length, start_ids=None, top_k=50):
        batch_size, *_  = input_ids.shape
        if batch_size not in self.batch_sizes:
            raise ValueError(f"Model not compiled for batch_size : {batch_size}. Acceptable batch_size is one of the following {self.batch_sizes}")

        if self.neuron_config.on_device_generation:
            result = sampling.sample_tokens(self, input_ids, start_ids, sequence_length=sequence_length, 
                                            config=self.neuron_config.on_device_generation)
        else:
            result = sampling.simple_sample(self, input_ids, start_ids, sequence_length,
                                            eos_token_id=self.config.eos_token_id, top_k=top_k)
        return result
