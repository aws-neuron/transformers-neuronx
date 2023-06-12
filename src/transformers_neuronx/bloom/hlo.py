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
from transformers_neuronx.layers import attention, transformer
from transformers_neuronx.bloom.config import BloomConfig

class BloomForSamplingNoEmbeddingHlo:

    def __init__(self, config: BloomConfig, neuron_config=None):
        self.config = config
        self.neuron_config = neuron_config

    def inputs(self, scribe, hidden_dtype, n_positions, n_active_tokens, batch_size):
        hidden_sizes = self.config.hidden_size, n_active_tokens, batch_size
        hidden = hidden_dtype[hidden_sizes].Parameter(parameter_number=0)
        cache_ids = scribe.s32[n_active_tokens].Parameter(parameter_number=1)
        start_ids = scribe.s32[batch_size].Parameter(parameter_number=2)

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
            start_mask=True
        )
        return (hidden, cache_ids, mask, active_mask), (1, 0, None)

    def pre_layer(self, hidden, cache_ids, mask, active_mask, slopes):
        prior_alibi, active_alibi = build_alibi_from_slopes(slopes, mask, active_mask, self.config.n_head, self.config.tp_degree)
        return hidden, cache_ids, mask, active_mask, prior_alibi, active_alibi

    def layer(self, hidden, cache_ids, mask, active_mask, prior_alibi, active_alibi, attn_k_cache, attn_v_cache,
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
        ln_hidden = hlo.layer_norm(hidden, pre_attn_ln_weight, pre_attn_ln_bias)
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
        ln_hidden = hlo.layer_norm(hidden, pre_mlp_ln_weight, pre_mlp_ln_bias)
        mlp_hidden = hlo.mlp(
            ln_hidden,
            mlp_in_weight, mlp_in_bias, mlp_out_weight, mlp_out_bias,
            activation_function='gelu_new',
            tp_degree=self.config.tp_degree,
            in_scales=mlp_in_scales, out_scales=mlp_out_scales,
            neuron_config=self.neuron_config
        )

        hidden = dtype[hidden.sizes].Add(mlp_hidden, hidden)
        return hidden, out_attn_k_cache, out_attn_v_cache

    def ln_lm_head(self, hidden, ln_f_weight, ln_f_bias, lm_head_weight, lm_head_bias):
        return transformer.ln_lm_head(hidden, ln_f_weight, ln_f_bias, lm_head_weight, lm_head_bias)

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

        # Multi-Token Context Encoding
        else:

            # This `context` computation block is intended for populating the
            # KV cache with multiple `n_active_tokens` tokens. This assumes
            # that there is no prior history so it skips any computation
            # performed on the cache.

            # S = Q @ K + A
            score = attention.score(query, key)
            score = f32[score.sizes].Convert(score)
            score = f32[score.sizes].Add(score, active_alibi)
            score = attention.mask(score, mask)
            score = hlo.cast(score, dtype)

            # C = softmax(S) @ V
            context = attention.context_combined(score, value)

        # O = (C @ wO) + bO
        output = attention.output(context, out_weight, out_scales, out_bias, self.config.tp_degree, neuron_config)

        # KCache[I] = K
        # VCache[I] = V
        updated_keys = attention.update_cache(cached_keys, cache_ids, key)
        updated_values = attention.update_cache(cached_values, cache_ids, value)

        return output, updated_keys, updated_values


def build_alibi_from_slopes(slopes, attention_mask, active_mask, num_heads, tp_degree=1):

    assert num_heads % tp_degree == 0, (
        f"Attention heads ({num_heads}) must be divisible by tensor parellism degree {tp_degree}"
    )

    scribe = attention_mask.scribe
    dtype = scribe.f32
    num_heads_tp = num_heads // tp_degree

    def _alibi(summation, mask):

        size = mask.sizes
        batch_size, n_active_tokens, seq_length = mask.sizes

        one = dtype.Constant(constant_value=1)
        one_br = dtype[size].Broadcast(one, dimensions=[])
        summation_sub = dtype[size].Subtract(summation, one_br)
        sum_mul = dtype[size].Multiply(summation_sub, mask)

        slopes_sh = dtype[batch_size, n_active_tokens, num_heads_tp, 1].Broadcast(slopes, dimensions=[2, 3])
        sum_sh = dtype[batch_size, n_active_tokens, 1, seq_length].Reshape(sum_mul)
        dot_dims = dict(
            lhs_contracting_dimensions=[3],
            lhs_batch_dimensions=[0, 1],
            rhs_contracting_dimensions=[2],
            rhs_batch_dimensions=[0, 1]
        )
        product = dtype[batch_size, n_active_tokens, num_heads_tp, seq_length].Dot(slopes_sh, sum_sh, dot_dimension_numbers=dot_dims)
        result = dtype[batch_size, num_heads_tp, n_active_tokens, seq_length].Transpose(product, dimensions=[0, 2, 1, 3])
        return result

    scribe = attention_mask.scribe
    fp32 = scribe.f32

    # Create alibi for the `attention_mask` tokens
    mask_cast = hlo.cast(attention_mask, fp32)
    summation = hlo.cumsum(mask_cast, -1)
    alibi = _alibi(summation, mask_cast)

    # Create alibi for the `active_mask` tokens:
    #    Since the prior token mask is the `attention_mask` and the
    #    active token mask is the `active_mask`, we need to combine both masks to
    #    find the true cumulative sum.
    if active_mask is not None:
        total = hlo.reduce_sum(mask_cast, 2)
        active_cast = hlo.cast(active_mask, fp32)
        total = fp32[total.sizes].Add(total, active_cast)
        total_sh = hlo.unsqueeze(total, 1)
        active_cast_sh = hlo.unsqueeze(active_cast, 1)
        active_alibi = _alibi(total_sh, active_cast_sh)
        return alibi, active_alibi

    # When no active mask, we do not have a "prior" alibi
    return None, alibi
