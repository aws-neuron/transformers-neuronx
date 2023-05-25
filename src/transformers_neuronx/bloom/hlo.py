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
import copy
from transformers_neuronx import hlo


class BloomForSamplingNoEmbeddingHlo:

    def __init__(self, tp_degree, hidden_size, activation_function, num_heads, start_mask=True, neuron_config=None):
        self.tp_degree = tp_degree
        self.hidden_size = hidden_size
        self.activation_function = activation_function
        self.num_heads = num_heads
        self.start_mask = start_mask
        self.neuron_config = neuron_config

    def inputs(self, scribe, hidden_dtype, n_positions, n_active_tokens, batch_size):
        hidden_sizes = self.hidden_size, n_active_tokens, batch_size
        hidden = hidden_dtype[hidden_sizes].Parameter(parameter_number=0)
        cache_ids = scribe.s32[n_active_tokens].Parameter(parameter_number=1)
        start_ids = scribe.s32[batch_size].Parameter(parameter_number=2)
        mask, active_mask = hlo.decoder_attention_mask(start_ids, cache_ids, n_positions,
                                                       'LE', False, self.start_mask)
        return (hidden, cache_ids, mask, active_mask), (1, 0, None)

    def pre_layer(self, hidden, cache_ids, mask, active_mask, slopes):
        alibi = build_alibi_from_slopes(slopes, mask, self.num_heads, self.tp_degree)
        return hidden, cache_ids, mask, active_mask, alibi

    def layer(self, hidden, cache_ids, mask, active_mask, alibi, attn_k_cache, attn_v_cache,
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
            ln_hidden, cache_ids, mask, active_mask, alibi, attn_k_cache, attn_v_cache,
            attn_q_weight, attn_q_scales, attn_q_bias,
            attn_k_weight, attn_k_scales, attn_k_bias,
            attn_v_weight, attn_v_scales, attn_v_bias,
            attn_out_weight, attn_out_scales, attn_out_bias,
            neuron_config=self.neuron_config
        )
        hidden = dtype[hidden.sizes].Add(attn_output, hidden)
        ln_hidden = hlo.layer_norm(hidden, pre_mlp_ln_weight, pre_mlp_ln_bias)
        mlp_hidden = hlo.mlp(ln_hidden, mlp_in_weight, mlp_in_bias, mlp_out_weight, mlp_out_bias,
                             activation_function=self.activation_function, tp_degree=self.tp_degree,
                             in_scales=mlp_in_scales, out_scales=mlp_out_scales,
                             neuron_config=self.neuron_config)

        hidden = dtype[hidden.sizes].Add(mlp_hidden, hidden)
        return hidden, out_attn_k_cache, out_attn_v_cache

    def ln_lm_head(self, hidden, ln_f_weight, ln_f_bias, lm_head_weight, lm_head_bias):
        hidden_size, n_active_tokens, batch_size = hidden.sizes
        dtype = hidden.dtype
        slice_threshold = 2  # 1 doesn't work for now; see P86509517
        if n_active_tokens > slice_threshold:
            slice_dimensions = [
                dict(start=0, limit=hidden_size, stride=1),
                dict(start=n_active_tokens-slice_threshold, limit=n_active_tokens, stride=1),
                dict(start=0, limit=batch_size, stride=1),
            ]
            n_active_tokens = slice_threshold
            sizes = hidden_size, n_active_tokens, batch_size
            hidden = dtype[sizes].Slice(hidden, slice_dimensions=slice_dimensions)
        ln_hidden = hlo.layer_norm(hidden, ln_f_weight, ln_f_bias)
        ln_hidden = dtype[hidden_size,n_active_tokens*batch_size].Reshape(ln_hidden)
        logits = hlo.dot00(lm_head_weight, ln_hidden)
        if lm_head_bias is not None:
            lm_head_bias = dtype[logits.sizes].Broadcast(lm_head_bias, dimensions=[0])
            logits = dtype[logits.sizes].Add(logits, lm_head_bias)
        vocab_size, _ = logits.sizes
        result = dtype[vocab_size,n_active_tokens,batch_size].Reshape(logits)
        return result

    def attention(self, hidden, cache_ids, mask, active_mask, alibi,
                  cached_keys, cached_values,
                  q_weight, q_scales, q_bias,
                  k_weight, k_scales, k_bias,
                  v_weight, v_scales, v_bias,
                  out_weight, out_scales, out_bias,
                  neuron_config=None):

        enable_quantize = neuron_config and neuron_config.quant
        if enable_quantize and not neuron_config.quant.quantize_attn:
            neuron_config = copy.deepcopy(neuron_config)
            neuron_config.quant = None
            enable_quantize = False

        dtype = hidden.dtype
        scribe = hidden.scribe
        f32 = scribe.f32
        pred = scribe.pred
        hidden_size, n_active_tokens, n_seqs = hidden_sizes = hidden.sizes
        max_ctx_plus_n_active_tokens, _, n_heads_tp, d_head = cached_keys.sizes
        attn_size = hidden_size // self.tp_degree
        hidden_r_sizes = hidden_size, n_active_tokens * n_seqs
        active_sizes = n_active_tokens, n_seqs, n_heads_tp, d_head

        hidden_r = dtype[hidden_r_sizes].Reshape(hidden)
        active_q = hlo.dot00_add1(hidden_r, q_weight, q_bias, q_scales, neuron_config)
        active_q = dtype[active_sizes].Reshape(active_q)

        active_k = hlo.dot00_add1(hidden_r, k_weight, k_bias, k_scales, neuron_config)
        active_k = dtype[active_sizes].Reshape(active_k)

        active_v = hlo.dot00_add1(hidden_r, v_weight, v_bias, v_scales, neuron_config)
        active_v = dtype[active_sizes].Reshape(active_v)

        # compute self-attention: V x Softmax(QK^T)
        # Keep the attention weights computation in fp32 to avoid overflow issues
        scale_attn = d_head ** 0.5
        dot_dims = dict(lhs_contracting_dimensions=[3],
                        lhs_batch_dimensions=[1, 2],
                        rhs_contracting_dimensions=[3],
                        rhs_batch_dimensions=[1, 2])

        scatter_dims = dict(update_window_dims=[1,2,3],
                            inserted_window_dims=[0],
                            scatter_dims_to_operand_dims=[0],
                            index_vector_dim=1)
        assign_func = hlo.gen_assign_func(dtype)

        cached_keys = dtype[cached_keys.sizes].Scatter(
            cached_keys, cache_ids, active_k, scatter_dimension_numbers=scatter_dims, to_apply=assign_func)
        score_sizes = n_seqs, n_heads_tp, n_active_tokens, max_ctx_plus_n_active_tokens
        score = dtype[score_sizes].Dot(active_q, cached_keys, dot_dimension_numbers=dot_dims)

        # Bloom-specific logic - Add alibi as QK bias
        # Reference: https://github.com/huggingface/transformers/blob/v4.29.2/src/transformers/models/bloom/modeling_bloom.py#L317
        score = f32[score_sizes].Convert(score)
        scale = f32.Constant(constant_value=scale_attn)
        scale_attn_br = f32[score_sizes].Broadcast(scale, dimensions=[])
        scale_score = f32[score_sizes].Divide(score, scale_attn_br)
        score = f32[score_sizes].Add(scale_score, alibi)

        # Masking
        large_neg = f32.Constant(constant_value=-65535.0)
        large_neg_br = f32[score_sizes].Broadcast(large_neg, dimensions=[])
        if len(mask.sizes) == 2:
            mask_br = pred[score_sizes].Broadcast(mask, dimensions=[2, 3])
        else:
            mask_br = pred[score_sizes].Broadcast(mask, dimensions=[0, 2, 3])
        score = f32[score_sizes].Select(mask_br, score, large_neg_br)

        # Compute probabilities
        probs = hlo.softmax(score)
        probs = dtype[score_sizes].Convert(probs)

        # Update value cache
        dot_dims = dict(lhs_contracting_dimensions=[3],
                        lhs_batch_dimensions=[0, 1],
                        rhs_contracting_dimensions=[0],
                        rhs_batch_dimensions=[1, 2])
        sizes = n_seqs, n_heads_tp, n_active_tokens, d_head
        cached_values = dtype[cached_values.sizes].Scatter(
                cached_values, cache_ids, active_v, scatter_dimension_numbers=scatter_dims, to_apply=assign_func)

        output = dtype[sizes].Dot(probs, cached_values, dot_dimension_numbers=dot_dims)
        sizes = n_active_tokens, n_seqs, n_heads_tp, d_head
        output = dtype[sizes].Transpose(output, dimensions=[2, 0, 1, 3])

        if enable_quantize:
            out_weight = dtype[out_weight.sizes].Convert(out_weight)
        output_sizes_2d = n_active_tokens*n_seqs, attn_size
        output = dtype[output_sizes_2d].Reshape(output)
        dot_dims = dict(lhs_contracting_dimensions=[0], rhs_contracting_dimensions=[1])
        output = dtype[hidden_r_sizes].Dot(out_weight, output, dot_dimension_numbers=dot_dims)
        if enable_quantize:
            output = hlo.dequantize(output, out_scales, neuron_config, 0)
        out_bias = dtype[hidden_r_sizes].Broadcast(out_bias, dimensions=[0])
        output = dtype[hidden_r_sizes].Add(output, out_bias)
        output = dtype[hidden_sizes].Reshape(output)

        if self.tp_degree == 1:
            return output, cached_keys, cached_values
        replica_groups = [list(range(self.tp_degree))]
        add_func = hlo.gen_add_func(dtype)
        output = dtype[hidden_sizes].AllReduce(output, replica_groups=replica_groups, to_apply=add_func)

        return output, cached_keys, cached_values


def build_alibi_from_slopes(slopes, attention_mask, num_heads, tp_degree=1):

    assert num_heads % tp_degree == 0, (
        f"Attention heads ({num_heads}) must be divisible by tensor parellism degree {tp_degree}"
    )
    num_heads_tp = num_heads // tp_degree

    scribe = attention_mask.scribe
    dtype = scribe.f32
    size = attention_mask.sizes

    batch_size, n_active_tokens, seq_length = attention_mask.sizes

    mask_cast = dtype[size].Convert(attention_mask)
    summation = hlo.cumsum(mask_cast, -1)
    one = dtype.Constant(constant_value=1)
    one_br = dtype[size].Broadcast(one, dimensions=[])
    summation_sub = dtype[size].Subtract(summation, one_br)
    sum_mul = dtype[size].Multiply(summation_sub, mask_cast)

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
