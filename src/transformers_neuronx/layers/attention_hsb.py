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
    tp_degree=None,
    neuron_config=None,
    shard_over_batch=False,
):
    """
    Self-attention input projections.

    Q = (hidden @ wQ) + bQ
    K = (hidden @ wK) + bK
    V = (hidden @ wV) + bV

    If n_kv_heads != 0, uses multi-query, multi-group attention.
    n_kv_heads == 0 -> outputs shapes [n_active_tokens, n_seqs, n_heads_tp, d_head]
    n_kv_heads != 0 -> outputs shapes [n_active_tokens, n_seqs, n_kv_heads, n_repeats, d_head] (query)
    and [n_active_tokens, n_seqs, n_kv_heads, d_head] (key/value)
    """
    dtype = hidden.dtype
    hidden_size, n_active_tokens, n_seqs = hidden.sizes
    _, hidden_size_tp = q_weight.sizes
    _, kv_hidden_size_tp = k_weight.sizes
    n_heads_tp = hidden_size_tp // d_head
    n_kv_heads_tp = kv_hidden_size_tp // d_head
    hidden_r_sizes = hidden_size, n_active_tokens * n_seqs

    hidden_r = dtype[hidden_r_sizes].Reshape(hidden)

    # Q = (hidden @ wQ) + bQ
    active_q = hlo.dot00_add1(hidden_r, q_weight, q_bias, q_scales, neuron_config)

    # K = (hidden @ wK) + bK
    active_k = hlo.dot00_add1(hidden_r, k_weight, k_bias, k_scales, neuron_config)

    # V = (hidden @ wV) + bV
    active_v = hlo.dot00_add1(hidden_r, v_weight, v_bias, v_scales, neuron_config)

    if shard_over_batch:
        # shard over batch
        scribe = active_q.scribe
        s32 = scribe.s32
        zero = s32.Constant(constant_value=0)

        # split along batch dimension, and concat along head dimension
        # TODO: Emit all-to-all CC op, instead of allgather+slice
        full_q = hlo.all_gather(active_q, dim=1, tp_degree=tp_degree)
        full_k = hlo.all_gather(active_k, dim=1, tp_degree=tp_degree)
        full_v = hlo.all_gather(active_v, dim=1, tp_degree=tp_degree)

        n_seqs_per_nc = n_seqs // tp_degree
        slice_limit = n_active_tokens * n_seqs_per_nc
        active_q = hlo.dynamic_slice_along(full_q, dim=0, size=slice_limit, start=zero)
        active_k = hlo.dynamic_slice_along(full_k, dim=0, size=slice_limit, start=zero)
        active_v = hlo.dynamic_slice_along(full_v, dim=0, size=slice_limit, start=zero)

        n_heads = n_heads_tp * tp_degree
        n_kv_heads = n_heads_tp * tp_degree // (hidden_size_tp // kv_hidden_size_tp)
        n_repeats = n_heads // n_kv_heads
        active_q_sizes = n_active_tokens, n_seqs_per_nc, n_kv_heads * n_repeats, d_head
        active_kv_sizes = n_active_tokens, n_seqs_per_nc, n_kv_heads, d_head
        active_q = hlo.reshape(active_q, active_q_sizes)
        active_k = hlo.reshape(active_k, active_kv_sizes)
        active_v = hlo.reshape(active_v, active_kv_sizes)
    else:
        # shard over head
        n_repeats = hidden_size_tp // kv_hidden_size_tp
        active_q_sizes = n_active_tokens, n_seqs, n_kv_heads_tp * n_repeats, d_head
        active_kv_sizes = n_active_tokens, n_seqs, n_kv_heads_tp, d_head
        active_q = hlo.reshape(active_q, active_q_sizes)
        active_k = hlo.reshape(active_k, active_kv_sizes)
        active_v = hlo.reshape(active_v, active_kv_sizes)

    return active_q, active_k, active_v


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


def score(query, keys, tp_degree=None, n_kv_heads=0, shard_over_batch=False):
    """
    Compute the attention score by combining scaled-query & keys.

    S = Q @ K

    If n_kv_heads != 0, uses multi-query, multi-group attention.
    NOTE: Since we may pad along head dimension,
          tp_degree argument is required to be an integer for grouped-query attention models.
    """

    dtype = query.dtype
    scribe = query.scribe
    pred = scribe.pred

    # Check for multi-query attention
    if n_kv_heads != 0:
        if shard_over_batch:
            _, _, n_heads, _ = query.sizes
            n_repeats = n_heads // n_kv_heads
        else:
            assert isinstance(tp_degree, int), \
                f"tp_degree ({tp_degree}) is required to be an integer for grouped-query attention models."
            _, _, n_heads_tp, _ = query.sizes
            n_heads = n_heads_tp * tp_degree
            n_repeats = n_heads // n_kv_heads
        keys = hlo.repeat_kv(keys, n_repeats=n_repeats, repeat_dim=2)

    # Q @ K
    dot_dims = dict(lhs_contracting_dimensions=[3],
                    lhs_batch_dimensions=[1, 2],
                    rhs_contracting_dimensions=[3],
                    rhs_batch_dimensions=[1, 2])

    result_dot = hlo.dot_general(query, keys, dimension_numbers=dot_dims)

    return result_dot


def mask(score, mask, tp_degree=None, shard_over_batch=False, constant_value=-30000):
    """
    Masks the computed attention scores with the attention mask.

    score = masked_fill(score, mask, -65535)
    """
    scribe = score.scribe
    dtype = score.dtype
    score_sizes = score.sizes
    pred = scribe.pred

    # Note: This value can cause NaN issues if it is too large
    large_neg = dtype.Constant(constant_value=constant_value) # Valid for fp32/fp16/bf16
    large_neg_br = dtype[score_sizes].Broadcast(large_neg, dimensions=[])
    if len(mask.sizes) == 2:
        if shard_over_batch:
            assert isinstance(tp_degree, int), \
                f"tp_degree ({tp_degree}) is required to be an integer for shard-over-batch."
            zero = scribe.s32.Constant(constant_value=0)
            n_seqs_per_nc = score_sizes[0]
            assert n_seqs_per_nc == mask.sizes[0] // tp_degree, f"invalid n_seqs_per_nc ({n_seqs_per_nc}) vs mask_sizes ({mask.sizes})"
            mask = hlo.dynamic_slice_along(mask, dim=0, start=zero, size=n_seqs_per_nc)
        # broadcast from [n_seqs, n_active_tokens] to [n_seqs, n_heads, n_active_tokens, n_positions]
        mask_br = pred[score_sizes].Broadcast(mask, dimensions=[0, 2])
    else:
        if shard_over_batch:
            assert isinstance(tp_degree, int), \
                f"tp_degree ({tp_degree}) is required to be an integer for shard-over-batch."
            zero = scribe.s32.Constant(constant_value=0)
            n_seqs_per_nc = score_sizes[0]
            assert n_seqs_per_nc == mask.sizes[0] // tp_degree, f"invalid n_seqs_per_nc ({n_seqs_per_nc}) vs mask_sizes ({mask.sizes})"
            mask = hlo.dynamic_slice_along(mask, dim=0, start=zero, size=n_seqs_per_nc)
        mask_br = pred[score_sizes].Broadcast(mask, dimensions=[0, 2, 3])
    score = dtype[score_sizes].Select(mask_br, score, large_neg_br)
    return score


def context(past_scores, active_score, past_values, active_values, sparse_mask=None,
            n_kv_heads=0, dtype=None, shard_over_batch=False, tp_degree=None):
    """
    Compute "context" output from the QK score and value projection.

    This computes the output using split past and current values. This can be
    efficient when computing a *single* next token score since it removes the
    data dependency on an updated KV cache.

    C = softmax(S) @ V

    If n_kv_heads != 0, uses multi-query, multi-group attention.
    If dtype is None, uses values datatype.
    If sparse_mask or active_sparse_mask is not None, use sparse attention on the corresponding values.
    """

    if dtype == None:
        dtype = active_score.dtype
    scribe = active_score.scribe
    f32 = scribe.f32

    n_seqs, n_heads, n_active_tokens, n_active_tokens = active_score_sizes = active_score.sizes
    n_seqs, n_heads, n_active_tokens, n_positions = past_scores.sizes
    if shard_over_batch:
        n_positions, n_seqs_per_nc, n_kv_heads, d_head = past_values.sizes
        n_seqs = n_seqs_per_nc * tp_degree
        n_heads_tp = n_heads // tp_degree
        reduce_sizes = n_seqs_per_nc, n_heads, n_active_tokens
    else:
        n_positions, n_seqs, n_kv_heads_tp, d_head = past_values.sizes
        _, n_heads_tp, _, _ = active_score.sizes
        reduce_sizes = n_seqs, n_heads_tp, n_active_tokens

    # Upcast to f32 before computation
    past_scores = hlo.cast(past_scores, f32)
    active_score = hlo.cast(active_score, f32)

    # Compute maximum of both past_scores and active_scores
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

    # Apply sparse masks after softmax to help compiler optimization
    if sparse_mask is not None:
        past_prob = mask(past_prob, sparse_mask, tp_degree=None, shard_over_batch=False, constant_value=0)

    # Ca = Pa @ Va
    # Cp = Pp @ Vp
    # C = Ca + Cp
    sizes = n_seqs, n_heads_tp, n_active_tokens, d_head
    dot_dims = dict(lhs_contracting_dimensions=[3],
                    lhs_batch_dimensions=[0, 1],
                    rhs_contracting_dimensions=[0],
                    rhs_batch_dimensions=[1, 2])
    denom = dtype[denom.sizes].Convert(denom)

    if n_kv_heads != 0:
        assert isinstance(tp_degree, int), \
            f"tp_degree ({tp_degree}) is required to be an integer for grouped-query attention."
        n_heads = n_heads_tp * tp_degree
        n_repeats = n_heads // n_kv_heads
        # values layout: (n_positions, n_seqs_per_nc, n_kv_heads, d_head) -> repeat_dim=2
        past_values = hlo.repeat_kv(past_values, n_repeats=n_repeats, repeat_dim=2)
        active_values = hlo.repeat_kv(active_values, n_repeats=n_repeats, repeat_dim=2)

    # lhs (past_prob): (n_seqs, n_heads, n_active_tokens, n_positions)
    # rhs (value): (n_positions, n_seqs, n_heads, d_head)
    dot_dims = dict(lhs_contracting_dimensions=[3],
                lhs_batch_dimensions=[0, 1],
                rhs_contracting_dimensions=[0],
                rhs_batch_dimensions=[1, 2])

    output_dot = hlo.dot_general(past_prob, past_values, dimension_numbers=dot_dims)
    active_output_dot = hlo.dot_general(active_prob, active_values, dimension_numbers=dot_dims)
    output = hlo.add(output_dot, active_output_dot)

    if shard_over_batch:
        scribe = output_dot.scribe
        s32 = scribe.s32
        zero = s32.Constant(constant_value=0)

        # concat along batch dimension and split along head dimension
        full_output = hlo.all_gather(output, dim=0, tp_degree=tp_degree)
        output = hlo.dynamic_slice_along(full_output, dim=1, start=zero, size=n_heads_tp)

    denom_br = dtype[sizes].Broadcast(denom, dimensions=[0, 1, 2])
    output = dtype[sizes].Divide(output, denom_br)
    sizes = n_active_tokens, n_seqs, n_heads_tp, d_head
    output = dtype[sizes].Transpose(output, dimensions=[2, 0, 1, 3])
    return output


def context_combined(score, values, sparse_mask=None, n_kv_heads=0, dtype=None, tp_degree=None, shard_over_batch=False):
    """
    Compute "context" output from the QK score and value projection.

    This function assumes that scores and values contains the both current *and*
    past values. This is unlike the split `context` layer which assumes that
    the key/value tensors are split between current/past values. The combined
    context may be useful during input token KV cache population while the
    split context function will provide better performance during single token
    generation.

    C = softmax(S) @ V

    If n_kv_heads != 0, uses multi-query, multi-group attention.
    If dtype is None, uses values datatype.
    If sparse_mask is not None, use sparse attention on the corresponding values.
    """
    probs = hlo.softmax(score)
    # Apply sparse masks after softmax to help compiler optimization
    if sparse_mask is not None:
        probs = mask(probs, sparse_mask, tp_degree=None, shard_over_batch=False, constant_value=0)

    n_seqs, n_heads_tp, n_active_tokens, n_positions = probs.sizes
    _, _, n_kv_heads_tp, d_head = values.sizes

    if dtype is None:
        dtype = values.dtype
    probs = hlo.cast(probs, dtype)

    if n_kv_heads != 0:
        if shard_over_batch:
            assert isinstance(tp_degree, int), f"expect tp_degree as int, but it is {tp_degree}"
            _, n_seqs_per_nc, n_kv_heads, d_head = values.sizes
            n_repeats = n_heads_tp // n_kv_heads
            n_seqs = n_seqs_per_nc * tp_degree
            _, n_heads, _, _ = probs.sizes
            n_heads_tp = n_heads // tp_degree
        else:
            _, n_seqs, n_kv_heads_tp, d_head = values.sizes
            n_repeats = n_heads_tp // n_kv_heads_tp
        values = hlo.repeat_kv(values, n_repeats=n_repeats, repeat_dim=2)

    dot_dims = dict(
        lhs_contracting_dimensions=[3],
        lhs_batch_dimensions=[0, 1],
        rhs_contracting_dimensions=[0],
        rhs_batch_dimensions=[1, 2]
    )
    result = hlo.dot_general(probs, values, dimension_numbers=dot_dims)

    if n_kv_heads != 0:
        if shard_over_batch:
            scribe = result.scribe
            s32 = scribe.s32
            zero = s32.Constant(constant_value=0)

            result_sizes = n_seqs_per_nc, n_heads, n_active_tokens, d_head
            result = hlo.reshape(result, result_sizes)

            # concat along batch dimension and split along head dimension
            slice_size = n_heads // tp_degree
            full_result = hlo.all_gather(result, dim=0, tp_degree=tp_degree)
            result = hlo.dynamic_slice_along(full_result, dim=1, start=zero, size=slice_size)
        else:
            result_sizes = n_seqs, n_heads_tp, n_active_tokens, d_head
            result = hlo.reshape(result, result_sizes)

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
    hidden_size, _ = out_weight.sizes
    hidden_sizes = hidden_size, n_active_tokens, n_seqs

    enable_quantize = neuron_config and neuron_config.quant
    if enable_quantize:
        out_weight = dtype[out_weight.sizes].Convert(out_weight)

    result_sizes_2d = n_active_tokens * n_seqs, n_heads_tp * d_head
    result = dtype[result_sizes_2d].Reshape(context)

    # (b * s, padded_h) @ (h, padded_h) contract=(1, 1) => (b * s, h)
    result = hlo.dot11_add1(result, out_weight, out_bias, out_scales, neuron_config=neuron_config)

    # (b * s, h) => (h, s, b)
    result = hlo.transpose(result, 0, 1)
    result = dtype[hidden_sizes].Reshape(result)

    if tp_degree == 1:
        return result

    all_reduce_dtype = None
    if neuron_config:
        all_reduce_dtype = neuron_config.all_reduce_dtype
    result = hlo.all_reduce_sum(result, tp_degree, dtype=all_reduce_dtype)
    return result
