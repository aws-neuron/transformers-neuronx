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
from transformers_neuronx.layers import alibi
from transformers_neuronx.bloom.config import BloomConfig
from transformers_neuronx.bloom.modules import BloomForCausalLM
from transformers_neuronx.bloom.hlo import BloomForSamplingNoEmbeddingHlo


class BloomForSampling(module.WrappingCheckpointCompatibleModel):

    def __init__(self, config, *, n_positions=2048, batch_size=1, amp='f32', tp_degree=2,
                 context_length_estimate=None, context_unroll=None,
                 unroll=None, neuron_config=None, **kwargs):
        config = BloomConfig(config, n_positions, batch_size, amp, tp_degree, **kwargs)
        super().__init__(BloomForCausalLM, config)
        self.config = config
        self.neuron_config =  neuron_config

        self.context_length_estimate = context_length_estimate
        if context_unroll is None:
            context_unroll = config.n_layer
        self.context_unroll = context_unroll

        if unroll is None:
            unroll = config.n_layer
        self.n_positions_list = utils.power_of_two_bucket_sizes(32, n_positions)

        self.decoder_lm_head = decoder.DecoderLmHeadForSamplingNoEmbedding(
            tp_degree, self.n_positions_list, 1, batch_size, config.attention_head_size, amp,
            config.n_layer, unroll, neuron_config=neuron_config, allow_pad=True
        )
        hlo_builder = BloomForSamplingNoEmbeddingHlo(config, neuron_config=neuron_config)
        self.decoder_lm_head.add_inputs_builder(hlo_builder.inputs)
        self.decoder_lm_head.add_pre_layer_builder(hlo_builder.pre_layer)
        self.decoder_lm_head.add_layer_builder(hlo_builder.layer)
        self.decoder_lm_head.add_ln_lm_head_builder(hlo_builder.ln_lm_head)
        self.decoder_lm_head_for_context = None

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
        slopes = alibi.build_slopes(self.config.n_head)
        self.decoder_lm_head.add_pre_layer_parameter(slopes, sharding=0, allow_pad=True)
        self.decoder_lm_head.to_neuron()

        if self.context_length_estimate is not None:
            self.decoder_lm_head_for_context = self.decoder_lm_head.build_weight_shared(
                n_positions_list=[self.context_length_estimate],
                n_active_tokens=self.context_length_estimate,
                unroll=self.context_unroll,
                share_caches=True,
            )

    def reset(self):
        self.decoder_lm_head.reset()

    def context(self, hidden, cache_ids, start_ids):
        context_length = hidden.shape[1]
        current = 0
        estimate = self.context_length_estimate
        if estimate is not None:
            hidden_context = hidden
            cache_context = cache_ids

            # Slice context that when it is too large
            if context_length > estimate:
                current = estimate
                hidden_context = hidden[:, :estimate]
                cache_context = cache_ids[:estimate]

            # Cannot use context encoding for a context that is too small. This
            # is because the caller must be aware of the cache-ids/start-ids
            # used.
            elif context_length < estimate:
                current = 0

            # Directly pass input to the context network when exactly sized
            else:
                current = estimate

            if current == estimate:
                logits = self.decoder_lm_head_for_context(hidden_context, cache_context, start_ids)

        for i in range(current, context_length):
            cache_ids = torch.as_tensor([i], dtype=torch.int32)
            logits = self.decoder_lm_head(hidden[:, i:i+1], cache_ids, start_ids)

        return logits

    def forward(self, input_ids, cache_ids=None, start_ids=None):

        batch_size, context_length = input_ids.shape
        if start_ids is None:
            start_ids = torch.zeros(batch_size, dtype=torch.int32)
        if cache_ids is None:
            cache_ids = torch.arange(context_length, dtype=torch.int32)

        hidden = self.chkpt_model.transformer.word_embeddings(input_ids)
        hidden = self.chkpt_model.transformer.word_embeddings_layernorm(hidden)
        hidden = hidden.transpose(0, -1)

        if context_length > 1:
            logits = self.context(hidden, cache_ids, start_ids)
        else:
            logits = self.decoder_lm_head(hidden, cache_ids, start_ids)

        logits = logits.to(torch.float32)
        logits = logits[:self.config.vocab_size]
        logits = logits.transpose(0, -1)
        logits = logits[:, -1, :]
        return logits

    def sample(self, input_ids, sequence_length, start_ids=None, top_k=50):

        # To enable optimized context encoding network, we must pad
        # up to the context length estimate or we will not correctly
        # select the final context logits (See: layers/transformer.py).
        # This also means we need to shift the start_ids over to correct
        # for padding.
        offset = 0
        if self.context_length_estimate:
            batch_size, context_length = input_ids.shape
            estimate = self.context_length_estimate
            if context_length < self.context_length_estimate:
                input_ids = utils.pad(input_ids, 1, estimate, left=True)
                offset = estimate - context_length
                if start_ids is None:
                    start_ids = torch.zeros(batch_size, dtype=torch.int32)
                start_ids += offset
                sequence_length += offset

        result = sampling.simple_sample(self, input_ids, start_ids, sequence_length,
                                          eos_token_id=self.config.eos_token_id, top_k=top_k)

        return result[:, offset:]
