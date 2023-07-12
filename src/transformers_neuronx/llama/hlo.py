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
from typing import Optional

from transformers_neuronx import hlo
from transformers_neuronx.layers import attention, transformer, rotary
from transformers_neuronx.llama.config import LlamaConfig
from transformers_neuronx.config import NeuronConfig


class LlamaForSamplingNoEmbeddingHlo:

    def __init__(self,
        config: LlamaConfig,
        neuron_config: Optional[NeuronConfig] = None
    ):
        self.config = config
        self.neuron_config = neuron_config

    def inputs(self, scribe, hidden_dtype, n_positions, n_active_tokens, batch_size):
        hidden_sizes = self.config.hidden_size, n_active_tokens, batch_size
        head_dim = self.config.attention_head_size

        hidden = hidden_dtype[hidden_sizes].Parameter(parameter_number=0)
        pos_embed = hidden_dtype[n_active_tokens, head_dim].Parameter(parameter_number=1)
        cache_ids = scribe.s32[n_active_tokens].Parameter(parameter_number=2)
        start_ids = scribe.s32[batch_size].Parameter(parameter_number=3)

        # NOTE: When using token generation network, we generate a mask for the
        #       past tokens and the current tokens separately. This allows us
        #       use the split "prefetch" attention layer.
        token_generation = n_active_tokens == 1
        triu_comparison = 'LT' if token_generation else 'LE'
        mask, active_mask = hlo.decoder_attention_mask(
            start_ids,
            cache_ids,
            n_positions,
            triu_comparison=triu_comparison,
            allow_kv_dot_prefetch=token_generation,
            start_mask=True,
        )
        return (hidden, pos_embed, cache_ids, mask, active_mask), (1, 0, 0, None)

    def layer(
            self, hidden, pos_embed, cache_ids, mask, active_mask,
            attn_k_cache, attn_v_cache,
            pre_attn_ln_weight, pre_attn_ln_bias,
            attn_q_weight, attn_q_scales, attn_q_bias,
            attn_k_weight, attn_k_scales, attn_k_bias,
            attn_v_weight, attn_v_scales, attn_v_bias,
            attn_out_weight, attn_out_scales, attn_out_bias,
            post_attn_ln_weight, post_attn_ln_bias,
            pre_mlp_ln_weight, pre_mlp_ln_bias,
            mlp_in_weight, mlp_in_scales, mlp_in_bias,
            mlp_out_weight, mlp_out_scales, mlp_out_bias,
            post_mlp_ln_weight, post_mlp_ln_bias,
            in0_weight, in1_weight, out_weight,
        ):
        dtype = hidden.dtype
        eps = self.config.rms_norm_eps
        ln_hidden = hlo.rms_norm(hidden, pre_attn_ln_weight, eps)
        attn_output, out_attn_k_cache, out_attn_v_cache = self.attention(
            ln_hidden, cache_ids, pos_embed, mask, active_mask,
            attn_k_cache, attn_v_cache,
            attn_q_weight, attn_q_scales, attn_q_bias,
            attn_k_weight, attn_k_scales, attn_k_bias,
            attn_v_weight, attn_v_scales, attn_v_bias,
            attn_out_weight, attn_out_scales, attn_out_bias,
            neuron_config=self.neuron_config
        )
        hidden = dtype[hidden.sizes].Add(attn_output, hidden)
        norm_hidden = hlo.rms_norm(hidden, pre_mlp_ln_weight, eps)
        mlp_hidden = hlo.gated_mlp(
            norm_hidden,
            in0_weight, in1_weight, out_weight,
            activation_function='silu',
            tp_degree=self.config.tp_degree,
        )
        res_hidden = dtype[hidden.sizes].Add(mlp_hidden, hidden)
        return res_hidden, out_attn_k_cache, out_attn_v_cache

    def ln_lm_head(self, hidden, rms_weight, unused_bias, lm_head_weight, lm_head_bias):
        return transformer.rms_lm_head(hidden, rms_weight, lm_head_weight, lm_head_bias, eps=self.config.rms_norm_eps)

    def attention(
        self,
        hidden, cache_ids, pos_embed, mask, active_mask,
        cached_keys, cached_values,
        q_weight, q_scales, q_bias,
        k_weight, k_scales, k_bias,
        v_weight, v_scales, v_bias,
        out_weight, out_scales, out_bias,
        neuron_config=None
    ):
        d_head = self.config.attention_head_size
        tp_degree = self.config.tp_degree

        # Q = (hidden @ wQ) + bQ
        # K = (hidden @ wK) + bK
        # V = (hidden @ wV) + bV
        query, key, value = attention.query_key_value(
            hidden,
            q_weight, q_scales, q_bias,
            k_weight, k_scales, k_bias,
            v_weight, v_scales, v_bias,
            d_head,
            neuron_config=neuron_config,
        )

        # Q = Rotate(Q)
        # K = Rotate(K)
        query, key = rotary.rotate_half(query, key, pos_embed)

        # Q = Q / sqrt(d_head)
        query = attention.scale(query, d_head)

        # Single Token Generation ("Prefetch"-style)
        if active_mask is not None:

            # Sp = Q @ Kp
            prior_scores = attention.score(query, cached_keys)
            prior_scores = attention.mask(prior_scores, mask)

            # Sa = Q @ Ka
            active_score = attention.score(query, key)
            active_score = attention.mask(active_score, active_mask)

            # C = softmax(Sa, Sp) @ (Va, Vp)
            context = attention.context(prior_scores, active_score, cached_values, value)

        # Multi-Token Context Encoding
        else:

            # S = Q @ K
            score = attention.score(query, key)
            score = attention.mask(score, mask)
            context = attention.context_combined(score, value)


        # O = (C @ wO) + bO
        output = attention.output(context, out_weight, out_scales, out_bias, tp_degree, neuron_config)

        # KCache[I] = K
        # VCache[I] = V
        updated_keys = attention.update_cache(cached_keys, cache_ids, key)
        updated_values = attention.update_cache(cached_values, cache_ids, value)

        return output, updated_keys, updated_values
