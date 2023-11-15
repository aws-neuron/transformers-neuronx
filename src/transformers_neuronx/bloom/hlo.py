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
from transformers_neuronx import hlo
from transformers_neuronx.constants import LAYOUT_BSH
from transformers_neuronx.layers import transformer, alibi
from transformers_neuronx.bloom.config import BloomConfig

class BloomForSamplingNoEmbeddingHlo:

    def __init__(self, config: BloomConfig, neuron_config=None):
        self.config = config
        self.neuron_config = neuron_config

    def inputs(self, scribe, dtype, n_positions, n_active_tokens, batch_size):
        hidden, cache_ids, start_ids, last_token_id, dims = transformer.inputs(
            scribe, dtype, batch_size, n_active_tokens, self.config.hidden_size, self.neuron_config
        )
        mask, active_mask = hlo.attention_mask(cache_ids, start_ids, n_positions)
        return (hidden, last_token_id, cache_ids, mask, active_mask), dims

    def pre_layer(self, hidden, last_token_id, cache_ids, mask, active_mask, slopes):
        prior_alibi, active_alibi = alibi.alibi(slopes, mask, active_mask)
        return hidden, last_token_id, cache_ids, mask, active_mask, prior_alibi, active_alibi

    def layer(self, hidden, last_token_id, cache_ids, mask, active_mask, prior_alibi, active_alibi, attn_k_cache, attn_v_cache,
              pre_attn_ln_weight, pre_attn_ln_bias,
              attn_q_weight, attn_q_scales, attn_q_bias,
              attn_k_weight, attn_k_scales, attn_k_bias,
              attn_v_weight, attn_v_scales, attn_v_bias,
              attn_out_weight, attn_out_scales, attn_out_bias,
              post_attn_ln_weight, post_attn_ln_bias,
              pre_mlp_ln_weight, pre_mlp_ln_bias,
              mlp_in_weight, mlp_in_scales, mlp_in_bias,
              mlp_out_weight, mlp_out_scales, mlp_out_bias,
              post_mlp_ln_weight, post_mlp_ln_bias):

        dtype = hidden.dtype
        is_bsh = self.neuron_config and self.neuron_config.attention_layout == LAYOUT_BSH
        layer_norm = hlo.layer_norm_bsh if is_bsh else hlo.layer_norm
        ln_hidden = layer_norm(hidden, pre_attn_ln_weight, pre_attn_ln_bias)
        attn_output, out_attn_k_cache, out_attn_v_cache = self.attention(
            ln_hidden, cache_ids, mask, active_mask, prior_alibi, active_alibi,
            attn_k_cache, attn_v_cache,
            attn_q_weight, attn_q_scales, attn_q_bias,
            attn_k_weight, attn_k_scales, attn_k_bias,
            attn_v_weight, attn_v_scales, attn_v_bias,
            attn_out_weight, attn_out_scales, attn_out_bias,
            neuron_config=self.neuron_config
        )
        hidden = dtype[hidden.sizes].Add(attn_output, hidden)
        ln_hidden = layer_norm(hidden, pre_mlp_ln_weight, pre_mlp_ln_bias)
        mlp = hlo.mlp_bsh if is_bsh else hlo.mlp
        mlp_hidden = mlp(
            ln_hidden,
            mlp_in_weight, mlp_in_bias, mlp_out_weight, mlp_out_bias,
            activation_function='gelu_new',
            tp_degree=self.config.tp_degree,
            in_scales=mlp_in_scales, out_scales=mlp_out_scales,
            neuron_config=self.neuron_config,
            transposed=True,
        )

        hidden = dtype[hidden.sizes].Add(mlp_hidden, hidden)
        return hidden, out_attn_k_cache, out_attn_v_cache

    def ln_lm_head(self, hidden, last_token_id, ln_f_weight, ln_f_bias, lm_head_weight, lm_head_bias, return_all_outputs=True):
        return transformer.ln_lm_head(self.config.tp_degree, hidden, last_token_id, ln_f_weight, ln_f_bias, lm_head_weight, lm_head_bias, return_all_outputs, neuron_config=self.neuron_config)

    def attention(self,
        hidden, cache_ids, mask, active_mask, prior_alibi, active_alibi,
        cached_keys, cached_values,
        q_weight, q_scales, q_bias,
        k_weight, k_scales, k_bias,
        v_weight, v_scales, v_bias,
        out_weight, out_scales, out_bias,
        neuron_config=None
    ):
        scribe = hidden.scribe
        f32 = scribe.f32
        dtype = hidden.dtype
        d_head = self.config.hidden_size // self.config.n_head

        is_bsh = neuron_config and neuron_config.attention_layout == LAYOUT_BSH
        if is_bsh:
            import transformers_neuronx.layers.attention as attention
        else:
            import transformers_neuronx.layers.attention_hsb as attention

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

        # Q = Q / sqrt(d_head)
        query = attention.scale(query, d_head)

        # Single Token Generation ("Prefetch"-style)
        if active_mask is not None:

            # This is an optimized `context` computation for a single token.
            # It uses a split prior/active key & value tensors. This allows
            # the KV cache updates to occur asynchronously with the attention
            # computation which improves performance on Neuron hardware.

            # Sp = Q @ Kp + Ap
            prior_scores = attention.score(query, cached_keys)
            prior_scores = f32[prior_scores.sizes].Convert(prior_scores)
            prior_scores = f32[prior_scores.sizes].Add(prior_scores, prior_alibi)
            prior_scores = attention.mask(prior_scores, mask)
            prior_scores = hlo.cast(prior_scores, dtype)

            # Sa = Q @ Ka + Aa
            active_score = attention.score(query, key)
            active_score = f32[active_score.sizes].Convert(active_score)
            active_score = f32[active_score.sizes].Add(active_score, active_alibi)
            active_mask_sh = hlo.unsqueeze(active_mask, 1)
            active_score = attention.mask(active_score, active_mask_sh)
            active_score = hlo.cast(active_score, dtype)

            # C = softmax(Sa, Sp) @ (Va, Vp)
            context = attention.context(prior_scores, active_score, cached_values, value)

            # KCache[I] = K
            # VCache[I] = V
            updated_keys = attention.update_cache(cached_keys, cache_ids, key)
            updated_values = attention.update_cache(cached_values, cache_ids, value)

        # Multi-Token Context Encoding
        else:

            # This `context` computation block is intended for populating the
            # KV cache with multiple `n_active_tokens` tokens. This assumes
            # that there is no prior history so it skips any computation
            # performed on the cache.

            # S = Q @ K + A
            score = attention.score(query, key)
            score = f32[score.sizes].Convert(score)
            score = f32[score.sizes].Add(score, prior_alibi)
            score = attention.mask(score, mask)
            score = hlo.cast(score, dtype)

            # C = softmax(S) @ V
            context = attention.context_combined(score, value)

            # KCache = K
            # VCache = V
            updated_keys = key
            updated_values = value

        # O = (C @ wO) + bO
        output = attention.output(context, out_weight, out_scales, out_bias, self.config.tp_degree, neuron_config)
        return output, updated_keys, updated_values
