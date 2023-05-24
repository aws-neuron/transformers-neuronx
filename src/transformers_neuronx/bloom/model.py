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

from transformers_neuronx import decoder
from transformers_neuronx import module
from transformers_neuronx import ops
from transformers_neuronx import sampling
from transformers_neuronx import utils
from transformers_neuronx.bloom.config import BloomConfig
from transformers_neuronx.bloom.modules import BloomForCausalLM
from transformers_neuronx.bloom.hlo import BloomForSamplingNoEmbeddingHlo


class BloomForSampling(module.WrappingCheckpointCompatibleModel):

    def __init__(self, config, *, n_positions=2048, batch_size=1, amp='f32', tp_degree=2,
                 context_length_estimate=None, context_unroll=1,
                 activation='gelu', unroll=None, neuron_config=None, **kwargs):
        config = BloomConfig(config, n_positions, batch_size, amp, tp_degree, activation, **kwargs)
        super().__init__(BloomForCausalLM, config)
        self.config = config
        self.neuron_config =  neuron_config

        # TODO: Implement context encoding
        if context_length_estimate is not None:
            raise AssertionError("The 'context_length_estimate' argument is not supported")

        if unroll is None:
            unroll = config.n_layer
        self.n_positions_list = utils.power_of_two_bucket_sizes(32, n_positions)

        self.decoder_lm_head = decoder.DecoderLmHeadForSamplingNoEmbedding(
            tp_degree, self.n_positions_list, 1, batch_size, config.attention_head_size, amp,
            config.n_layer, unroll, neuron_config=neuron_config
        )
        hlo_builder = BloomForSamplingNoEmbeddingHlo(tp_degree, config.hidden_size, 'gelu_new', config.n_head, True, neuron_config=neuron_config)
        self.decoder_lm_head.add_inputs_builder(hlo_builder.inputs)
        self.decoder_lm_head.add_pre_layer_builder(hlo_builder.pre_layer)
        self.decoder_lm_head.add_layer_builder(hlo_builder.layer)
        self.decoder_lm_head.add_ln_lm_head_builder(hlo_builder.ln_lm_head)

    def build_alibi_slopes(self):
        # Reference: https://github.com/huggingface/transformers/blob/v4.29.2/src/transformers/models/bloom/modeling_bloom.py#L86
        num_heads = self.config.n_head

        closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
        base = 2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3)))
        powers = range(1, 1 + closest_power_of_2)
        slopes = list(map(lambda x: math.pow(base, x), powers))

        if closest_power_of_2 != num_heads:
            extra_base = 2 ** (-(2 ** -(math.log2(2 * closest_power_of_2) - 3)))
            num_remaining_heads = min(closest_power_of_2, num_heads - closest_power_of_2)
            extra_powers = range(1, 1 + 2 * num_remaining_heads, 2)
            extra_slopes = list(map(lambda x: math.pow(extra_base, x), extra_powers))
            slopes.extend(extra_slopes)

        assert len(slopes) == num_heads
        return torch.tensor(slopes).view(num_heads, 1)

    def to_neuron(self):

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

            new_layer.add_attention_output(
                attn.dense.weight.detach().T,
                attn.dense.bias.detach()
            )
            new_layer.add_pre_mlp_layer_norm(
                layer.post_attention_layernorm.weight.detach(),
                layer.post_attention_layernorm.bias.detach()
            )
            new_layer.add_mlp_input(
                mlp.dense_h_to_4h.weight.detach().T,
                mlp.dense_h_to_4h.bias.detach()
            )
            new_layer.add_mlp_output(
                mlp.dense_4h_to_h.weight.detach().T,
                mlp.dense_4h_to_h.bias.detach()
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
        slopes = self.build_alibi_slopes()
        self.decoder_lm_head.add_pre_layer_parameter(slopes, sharding=0)
        self.decoder_lm_head.to_neuron()

    def reset(self):
        self.decoder_lm_head.reset()

    def forward(self, input_ids, cache_ids, start_ids=None):
        hidden = self.chkpt_model.transformer.word_embeddings(input_ids)
        hidden = self.chkpt_model.transformer.word_embeddings_layernorm(hidden)
        start_ids = torch.zeros([self.config.batch_size], dtype=torch.int32)
        hidden = hidden.transpose(0, -1)
        logits = self.decoder_lm_head(hidden, cache_ids, start_ids)
        logits = logits.to(torch.float32)
        logits = logits[:self.config.vocab_size]
        logits = logits.transpose(0, -1)
        logits = logits[:, -1, :]
        return logits

    def sample(self, input_ids, sequence_length, start_ids=None, top_k=50):
        return sampling.simple_sample(self, input_ids, start_ids, sequence_length,
                                      eos_token_id=self.config.eos_token_id, top_k=top_k)
