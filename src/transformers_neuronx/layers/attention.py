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


def query_key_value(
    hidden,
    q_weight, q_scales, q_bias,
    k_weight, k_scales, k_bias,
    v_weight, v_scales, v_bias,
    d_head,
    neuron_config=None
):
    """
    Self-attention input projections.

    Q = (hidden @ wQ) + bQ
    K = (hidden @ wK) + bK
    V = (hidden @ wV) + bV
    """
    dtype = hidden.dtype
    hidden_size, n_active_tokens, n_seqs = hidden.sizes
    hidden_size, d_head_tp = q_weight.sizes
    n_heads_tp = d_head_tp // d_head

    hidden_r_sizes = hidden_size, n_active_tokens * n_seqs
    active_sizes = n_active_tokens, n_seqs, n_heads_tp, d_head

    hidden_r = dtype[hidden_r_sizes].Reshape(hidden)

    # Q = (hidden @ wQ) + bQ
    active_q = hlo.dot00_add1(hidden_r, q_weight, q_bias, q_scales, neuron_config)
    active_q = dtype[active_sizes].Reshape(active_q)

    # K = (hidden @ wK) + bK
    active_k = hlo.dot00_add1(hidden_r, k_weight, k_bias, k_scales, neuron_config)
    active_k = dtype[active_sizes].Reshape(active_k)

    # V = (hidden @ wV) + bV
    active_v = hlo.dot00_add1(hidden_r, v_weight, v_bias, v_scales, neuron_config)
    active_v = dtype[active_sizes].Reshape(active_v)

    return active_q, active_k, active_v

def get_up_down(q):
    """
    Given a tensor, returns its upper and lower halves (divided in the last dimension)
    """
    head_dim = q.sizes[-1]
    q_up = hlo.slice_along(q, -1, head_dim//2)
    q_down = hlo.slice_along(q, -1, head_dim, head_dim//2)
    return q_up, q_down

def rotate_vec(q, sin_r, cos_r):
    """
    Given vectors q, sin, and cos tables, apply rotation to vectors
    """
    q_up, q_down = get_up_down(q)
    q_rot_up = hlo.ax_minus_by(cos_r, q_up, sin_r, q_down)
    q_rot_down = hlo.ax_plus_by(cos_r, q_down, sin_r, q_up)
    q_rot = q.dtype[q.sizes].Concatenate(q_rot_up, q_rot_down, dimensions=[3])
    return q_rot

# TODO: This should be removed and rotate_half should be used instead after GPTNeoX changes.
def query_key_projection(query, key, qk_weight):
    """
    A secondary projection to apply to input query/key projections (used in
    specific models: GPT-J/GPT-NeoX).

    Q = Q @ W
    K = K @ W
    """
    dtype = key.dtype
    n_active_tokens, n_seqs, n_heads_tp, d_head = active_sizes = key.sizes
    active_r_sizes = n_active_tokens, n_seqs * n_heads_tp, d_head

    dot_dims = dict(
        lhs_batch_dimensions=[0],
        lhs_contracting_dimensions=[2],
        rhs_batch_dimensions=[0],
        rhs_contracting_dimensions=[1]
    )

    # Q = Q @ W
    query = dtype[active_r_sizes].Reshape(query)
    query = dtype[active_r_sizes].Dot(query, qk_weight, dot_dimension_numbers=dot_dims)
    query = dtype[active_sizes].Reshape(query)

    # K = K @ W
    key = dtype[active_r_sizes].Reshape(key)
    key = dtype[active_r_sizes].Dot(key, qk_weight, dot_dimension_numbers=dot_dims)
    key = dtype[active_sizes].Reshape(key)

    return query, key

def rotate_half(query, key, sin_cos):
    """
    A secondary projection to apply to input query/key projections (used in
    specific models: GPT-J/GPT-NeoX/Llama).

    """
    dtype = key.dtype
    n_active_tokens, n_seqs, n_heads_tp, d_head = active_sizes = key.sizes
    active_r_sizes = n_active_tokens, n_seqs * n_heads_tp, d_head

    """
        Vector approach:
        | q_up cos - q_down sin |
        | q_up sin + q_down cos |
    """
    # Rotate query and key
    n_active_tokens, head_dim = sin_cos.sizes
    sin_sizes = n_active_tokens, head_dim // 2        
    broadcast_sizes = n_active_tokens, n_seqs, n_heads_tp, d_head // 2
    
    # Get sin and cos as upper and lower half of input embedding
    sin, cos = get_up_down(sin_cos)        
    sin_r = dtype[broadcast_sizes].Broadcast(sin, dimensions=[0,3])
    cos_r = dtype[broadcast_sizes].Broadcast(cos, dimensions=[0,3])

    # Rotate query
    query = rotate_vec(query, sin_r, cos_r)        
    
    # Rotate key
    key = rotate_vec(key, sin_r, cos_r)
    return query, key

def update_cache(cache, cache_ids, values):
    """
    Cache[I] = X
    """
    dtype = values.dtype
    cache_size = cache.sizes
    scatter_dims = dict(update_window_dims=[1,2,3],
                        inserted_window_dims=[0],
                        scatter_dims_to_operand_dims=[0],
                        index_vector_dim=1)
    assign_func = hlo.gen_assign_func(dtype)
    updated = dtype[cache_size].Scatter(
        cache, cache_ids, values, scatter_dimension_numbers=scatter_dims, to_apply=assign_func)
    return updated


def scale(query, d_head):
    """
    Scales the query by the number of attention heads

    Q = Q / sqrt(d_head)
    """
    dtype = query.dtype
    scale = dtype.Constant(constant_value=d_head ** 0.5)
    scale_br = dtype[query.sizes].Broadcast(scale, dimensions=[])
    return dtype[query.sizes].Divide(query, scale_br)


def score(query, keys):
    """
    Compute the attention score by combining scaled-query & keys.

    S = Q @ K
    """

    dtype = query.dtype
    scribe = query.scribe
    pred = scribe.pred

    n_active_tokens, n_seqs, n_heads_tp, d_head = query.sizes
    n_positions, n_seqs, n_heads_tp, d_head = keys.sizes
    size = n_seqs, n_heads_tp, n_active_tokens, n_positions

    # Q @ K
    dot_dims = dict(lhs_contracting_dimensions=[3],
                    lhs_batch_dimensions=[1, 2],
                    rhs_contracting_dimensions=[3],
                    rhs_batch_dimensions=[1, 2])
    result = dtype[size].Dot(query, keys, dot_dimension_numbers=dot_dims)

    return result


def mask(score, mask):
    """
    Masks the computed attention scores with the attention mask.

    score = masked_fill(score, mask, -65535)
    """
    scribe = score.scribe
    dtype = score.dtype
    score_sizes = score.sizes
    pred = scribe.pred

    # Note: This value can cause NaN issues if it is too large
    large_neg = dtype.Constant(constant_value=-30000) # Valid for fp32/fp16/bf16
    large_neg_br = dtype[score_sizes].Broadcast(large_neg, dimensions=[])
    if len(mask.sizes) == 2:
        mask_br = pred[score_sizes].Broadcast(mask, dimensions=[2, 3])
    else:
        mask_br = pred[score_sizes].Broadcast(mask, dimensions=[0, 2, 3])
    score = dtype[score_sizes].Select(mask_br, score, large_neg_br)
    return score


def context(past_scores, active_score, past_values, active_values):
    """
    Compute "context" output from the QK score and value projection.

    This computes the output using split past and current values. This can be
    efficient when computing a *single* next token score since it removes the
    data dependency on an updated KV cache.

    O = softmax(S) @ V
    """

    dtype = active_score.dtype
    scribe = active_score.scribe
    f32 = scribe.f32

    n_seqs, n_heads_tp, n_active_tokens, n_active_tokens = active_score_sizes = active_score.sizes
    n_seqs, n_heads_tp, n_active_tokens, n_positions = past_scores.sizes
    n_positions, n_seqs, n_heads_tp, d_head = past_values.sizes

    # Upcast to f32 before computation
    past_scores = hlo.cast(past_scores, f32)
    active_score = hlo.cast(active_score, f32)

    # Compute maximum of both past_scores and active_scores
    reduce_sizes = n_seqs, n_heads_tp, n_active_tokens
    minus_inf = f32.Constant(constant_value=float('-inf'))
    max_func = hlo.gen_max_func(f32)
    reduce_max = f32[reduce_sizes].Reduce(past_scores, minus_inf, dimensions=[3], to_apply=max_func)
    active_reduce_max = f32[reduce_sizes].Reduce(active_score, minus_inf, dimensions=[3], to_apply=max_func)
    reduce_max = f32[reduce_sizes].Maximum(reduce_max, active_reduce_max)
    reduce_max_br = f32[past_scores.sizes].Broadcast(reduce_max, dimensions=[0, 1, 2])

    # Pa = softmax(Sa)
    # Pp = softmax(Sp)
    score_shifted = f32[past_scores.sizes].Subtract(past_scores, reduce_max_br)
    exp = f32[past_scores.sizes].Exp(score_shifted)
    zero = f32.Constant(constant_value=0)
    add_func = hlo.gen_add_func(f32)
    denom = f32[reduce_sizes].Reduce(exp, zero, dimensions=[3], to_apply=add_func)
    past_prob = dtype[exp.sizes].Convert(exp)
    reduce_max_bra = f32[active_score_sizes].Broadcast(reduce_max, dimensions=[0, 1, 2])
    active_score_shifted = f32[active_score_sizes].Subtract(active_score, reduce_max_bra)
    active_prob = f32[active_score_sizes].Exp(active_score_shifted)
    active_denom = f32[reduce_sizes].Reduce(active_prob, zero, dimensions=[3], to_apply=add_func)
    denom = f32[reduce_sizes].Add(denom, active_denom)
    active_prob = dtype[active_prob.sizes].Convert(active_prob)

    # Ca = Pa @ Va
    # Cp = Pp @ Vp
    # C = Ca + Cp
    sizes = n_seqs, n_heads_tp, n_active_tokens, d_head
    dot_dims = dict(lhs_contracting_dimensions=[3],
                    lhs_batch_dimensions=[0, 1],
                    rhs_contracting_dimensions=[0],
                    rhs_batch_dimensions=[1, 2])
    denom = dtype[denom.sizes].Convert(denom)
    output = dtype[sizes].Dot(past_prob, past_values, dot_dimension_numbers=dot_dims)
    active_output = dtype[sizes].Dot(active_prob, active_values, dot_dimension_numbers=dot_dims)
    output = dtype[sizes].Add(output, active_output)
    denom_br = dtype[sizes].Broadcast(denom, dimensions=[0, 1, 2])
    output = dtype[sizes].Divide(output, denom_br)
    sizes = n_active_tokens, n_seqs, n_heads_tp, d_head
    output = dtype[sizes].Transpose(output, dimensions=[2, 0, 1, 3])
    return output


def context_combined(score, values):
    """
    Compute "context" output from the QK score and value projection.

    This function assumes that scores and values contains the both current *and*
    past values. This is unlike the split `context` layer which assumes that
    the key/value tensors are split between current/past values. The combined
    context may be useful during input token KV cache population while the
    split context function will provide better performance during single token
    generation.

    O = softmax(S) @ V
    """
    probs = hlo.softmax(score)

    n_seqs, n_heads_tp, n_active_tokens, n_positions = probs.sizes
    n_positions, n_seqs, n_heads_tp, d_head = values.sizes

    dtype = values.dtype
    if probs.dtype != values.dtype:
        probs = dtype[score.sizes].Convert(score)

    dot_dims = dict(
        lhs_contracting_dimensions=[3],
        lhs_batch_dimensions=[0, 1],
        rhs_contracting_dimensions=[0],
        rhs_batch_dimensions=[1, 2]
    )
    sizes = n_seqs, n_heads_tp, n_active_tokens, d_head
    result = dtype[sizes].Dot(probs, values, dot_dimension_numbers=dot_dims)
    sizes = n_active_tokens, n_seqs, n_heads_tp, d_head
    result = dtype[sizes].Transpose(result, dimensions=[2, 0, 1, 3])
    return result


def output(
    context,
    out_weight, out_scales, out_bias,
    tp_degree,
    neuron_config=None
):
    """
    The output projection of a transformer applied to the attention context.

    O = (C @ wO) + bO
    """
    dtype = context.dtype
    n_active_tokens, n_seqs, n_heads_tp, d_head = context.sizes
    attn_size, hidden_size = out_weight.sizes
    hidden_sizes = hidden_size, n_active_tokens, n_seqs
    hidden_r_sizes = hidden_size, n_active_tokens * n_seqs
    attn_size = hidden_size // tp_degree

    enable_quantize = neuron_config and neuron_config.quant
    if enable_quantize:
        out_weight = dtype[out_weight.sizes].Convert(out_weight)

    result_sizes_2d = n_active_tokens * n_seqs, n_heads_tp * d_head
    result = dtype[result_sizes_2d].Reshape(context)
    dot_dims = dict(lhs_contracting_dimensions=[0], rhs_contracting_dimensions=[1])
    result = dtype[hidden_r_sizes].Dot(out_weight, result, dot_dimension_numbers=dot_dims)
    if enable_quantize:
        result = hlo.dequantize(result, out_scales, neuron_config, 0)

    if out_bias is not None:
        out_bias = dtype[hidden_r_sizes].Broadcast(out_bias, dimensions=[0])
        result = dtype[hidden_r_sizes].Add(result, out_bias)

    result = dtype[hidden_sizes].Reshape(result)

    if tp_degree == 1:
        return result

    replica_groups = [list(range(tp_degree))]
    add_func = hlo.gen_add_func(dtype)
    result = dtype[hidden_sizes].AllReduce(result, replica_groups=replica_groups, to_apply=add_func)
    return result
