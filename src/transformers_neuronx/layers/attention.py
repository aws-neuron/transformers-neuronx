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
from transformers_neuronx import utils
from transformers_neuronx.constants import FUSED_QKV_TP_FACTOR
from transformers_neuronx.constants import LAYOUT_BSH
from transformers_neuronx import constants
from transformers_neuronx.config import NeuronConfig
from transformers_neuronx.layers import attention_utils
from transformers_neuronx.nki.compile import nki_call
import logging


def query_key_value(
    hidden,
    q_weight, q_scales, q_bias,
    k_weight, k_scales, k_bias,
    v_weight, v_scales, v_bias,
    d_head,
    tp_degree=None,
    neuron_config=None,
    shard_over_batch=False,
    n_kv_heads_tp=None,
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
    if neuron_config and neuron_config.attention_layout == LAYOUT_BSH:
        hidden = hlo.transpose210(hidden)

    dtype = hidden.dtype
    hidden_size, n_active_tokens, n_seqs = hidden.sizes
    weight_tiling = False
    if len(q_weight.sizes) == 4:
        weight_tiling = True
        tile_size, hidden_size_tile, _, _ = q_weight.sizes
        hidden_size_tp = tile_size * hidden_size_tile
    else:
        _, hidden_size_tp = q_weight.sizes
    fuse_qkv = neuron_config and neuron_config.fuse_qkv
    if fuse_qkv:
        # If KV head count explicit, find Q head count
        if n_kv_heads_tp != None:
            n_total_heads_tp = hidden_size_tp // d_head
            n_heads_tp = n_total_heads_tp - 2 * n_kv_heads_tp
            # Q hidden size
            hidden_size_tp = d_head * n_heads_tp
            # KV hidden size
            kv_hidden_size_tp = d_head * n_kv_heads_tp
        # KV head count not specified, assume same as Q
        else:
            hidden_size_tp //= FUSED_QKV_TP_FACTOR
            kv_hidden_size_tp = hidden_size_tp
            n_heads_tp = hidden_size_tp // d_head
            n_kv_heads_tp = kv_hidden_size_tp // d_head
    else:
        _, kv_hidden_size_tp = k_weight.sizes
        n_heads_tp = hidden_size_tp // d_head
        n_kv_heads_tp = kv_hidden_size_tp // d_head

    sharded_gqa_kv = (
        kv_hidden_size_tp < d_head
        and neuron_config is not None
        and neuron_config.group_query_attention == constants.GQA.ALL_GATHER_HEADS
    )

    # (h, s, b) => (h, s * b)
    hidden_r = hlo.reshape(hidden, (hidden_size, n_active_tokens * n_seqs))

    # Sharded KV GQA
    if sharded_gqa_kv:
        # Q = (hidden @ wQ) + bQ
        active_q = hlo.dot00_add1(hidden_r, q_weight, q_bias, q_scales, neuron_config)

        # K = (hidden @ wK) + bK
        active_k = _sharded_kv_projection(hidden, k_weight, k_bias, k_scales, neuron_config, d_head, tp_degree)

        # V = (hidden @ wV) + bV
        active_v = _sharded_kv_projection(hidden, v_weight, v_bias, v_scales, neuron_config, d_head, tp_degree)

    # Fused MHA
    elif fuse_qkv:
        # QKV = (hidden @ wQKV) + bQKV
        if weight_tiling:
            assert hidden_size % constants.TILE_SIZE == 0, \
                f"Hidden size needs to be divisible by {constants.TILE_SIZE}" \
                f"in order to use weight tiling. Received hidden with size {hidden_size}."
            # (h, s * b) -> (h // TILE_SIZE, TILE_SIZE, b x s)
            hidden_tiled_sizes = hidden_size // constants.TILE_SIZE, constants.TILE_SIZE, n_seqs * n_active_tokens
            hidden_r = hlo.reshape(hidden_r, hidden_tiled_sizes)
            active_qkv = hlo.dot_0120_add1(hidden_r, q_weight, q_bias, q_scales, neuron_config=neuron_config)
        else:
            active_qkv = hlo.dot00_add1(hidden_r, q_weight, q_bias, q_scales, neuron_config=neuron_config)

        # Split
        slice_lim = active_qkv.sizes[-1] // (n_heads_tp + 2 * n_kv_heads_tp)
        active_q = hlo.slice_along(active_qkv, -1, n_heads_tp*slice_lim, start=0)
        active_k = hlo.slice_along(active_qkv, -1, (n_heads_tp+n_kv_heads_tp)*slice_lim, start=n_heads_tp*slice_lim)
        active_v = hlo.slice_along(active_qkv, -1, (n_heads_tp+2*n_kv_heads_tp)*slice_lim, start=(n_heads_tp+n_kv_heads_tp)*slice_lim)

    # MHA & Non-sharded KV GQA
    else:
        # Q = (hidden @ wQ) + bQ
        active_q = hlo.dot00_add1(hidden_r, q_weight, q_bias, q_scales, neuron_config)

        # K = (hidden @ wK) + bK
        active_k = hlo.dot00_add1(hidden_r, k_weight, k_bias, k_scales, neuron_config)

        # V = (hidden @ wV) + bV
        active_v = hlo.dot00_add1(hidden_r, v_weight, v_bias, v_scales, neuron_config)

    if shard_over_batch:
        # shard over batch
        n_seqs_per_nc = n_seqs // tp_degree
        n_heads = n_heads_tp * tp_degree
        n_kv_heads = n_heads_tp * tp_degree // (hidden_size_tp // kv_hidden_size_tp)
        n_repeats = n_heads // n_kv_heads
        active_q_sizes = n_active_tokens, n_seqs_per_nc, n_kv_heads * n_repeats, d_head
        active_kv_sizes = n_active_tokens, n_seqs_per_nc, n_kv_heads, d_head

        # split along batch dimension, and concat along head dimension
        active_q = hlo.all_to_all(active_q, split_dim=0, concat_dim=1, tp_degree=tp_degree)
        active_k = hlo.all_to_all(active_k, split_dim=0, concat_dim=1, tp_degree=tp_degree)
        active_v = hlo.all_to_all(active_v, split_dim=0, concat_dim=1, tp_degree=tp_degree)

        active_q = hlo.reshape(active_q, active_q_sizes)
        active_k = hlo.reshape(active_k, active_kv_sizes)
        active_v = hlo.reshape(active_v, active_kv_sizes)
    else:
        # shard over head
        active_q_sizes = n_active_tokens, n_seqs, n_heads_tp, d_head
        active_kv_sizes = n_active_tokens, n_seqs, n_kv_heads_tp, d_head
        active_q = hlo.reshape(active_q, active_q_sizes)
        if not sharded_gqa_kv:
            active_k = hlo.reshape(active_k, active_kv_sizes)
            active_v = hlo.reshape(active_v, active_kv_sizes)

    return active_q, active_k, active_v


def _sharded_kv_projection(hidden, weight, bias, scales, neuron_config, d_head, tp_degree):

    _, hidden_size_tp = weight.sizes
    group_size = d_head // hidden_size_tp
    num_groups = tp_degree // group_size
    n_head = (tp_degree * hidden_size_tp) // d_head
    num_heads_per_group = n_head // num_groups

    # (h, s, b) => (h, s * b)
    hidden_size, n_active_tokens, n_seqs = hidden.sizes
    hidden = hlo.reshape(hidden, (hidden_size, n_active_tokens * n_seqs))

    # O = (hidden @ W) + B
    # (h, s * b) @ (h, n_head * d_head) contract(0, 0) => (s * b, n_head * d_head)
    active = hlo.dot00_add1(hidden, weight, bias, scales, neuron_config)

    # Gather portions of the groups together
    replica_groups = utils.build_replica_groups(num_groups, group_size)
    active = hlo.all_gather(active, dim=1, tp_degree=tp_degree, replica_groups=replica_groups)

    # (s * b, n_head * d_head) => (s * b, n_head, d_head)
    active = hlo.reshape(active, (n_active_tokens, n_seqs, num_heads_per_group, d_head))

    return active


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


def fused_kv_update_cache(cached_keys, cached_vals, cache_ids, keys, vals, start_ids=None, neuron_config=None):
    """
    The fused K/V cache update is intended for reducing replicated index value calculation for both keys and values,
    since we are updating K/V with the same index offset.

    KeyCache[I], ValueCache[I] = Keys, Values
    """
    # Check K/V cache layout
    bsh_cache_layout = False
    if neuron_config is not None:
        bsh_cache_layout = neuron_config.cache_layout == constants.LAYOUT_BSH

    dtype = cached_keys.dtype
    cache_ids_dtype = cache_ids.dtype
    use_2d_cache_ids = len(cache_ids.sizes) > 1
    if not use_2d_cache_ids:
        updated_keys = update_cache(cached_keys, cache_ids, keys)
        updated_vals = update_cache(cached_vals, cache_ids, vals)
        return updated_keys, updated_vals

    # 2D cache_ids
    cache_ids = hlo.transpose(cache_ids, 0, 1)
    assign_func = hlo.gen_assign_func(dtype)
    if bsh_cache_layout:
        n_seqs, n_positions, n_kv_heads, d_head = cached_keys.sizes
        n_active_seqs, n_active_tokens, _, _  = keys.sizes
    else:
        n_positions, n_seqs, n_kv_heads, d_head = cached_keys.sizes
        n_active_tokens, n_active_seqs, _, _  = keys.sizes
    assert cache_ids.sizes[0] == n_active_tokens, \
        f"inconsistent sizes between cache_ids ({cache_ids.sizes}) and values ({keys.sizes})"

    # reshape cache, and scatter values in a for loop.
    #
    # NOTE: Due to limitation in hlo.scatter, we make cache flatten: (p0 as positions, s0 as sequences)
    #       (p0, s0), (p0, s1), (p0, s2), (p1, s0), (p1, s1), (p1, s2)
    #       This means we cannot update the sequence in the cache with one scatter op, without reordering the cache.
    kv_hidden_size = n_kv_heads * d_head
    cached_keys_r = hlo.reshape(cached_keys, [n_positions * n_seqs, kv_hidden_size])
    cached_vals_r = hlo.reshape(cached_vals, [n_positions * n_seqs, kv_hidden_size])

    if n_active_tokens == 1 and n_seqs == n_active_seqs:
        # cache (2D): [n_positions * n_seqs, n_kv_heads * d_head]
        #        +---------3-4-----6-7------9-10-----------------
        # seq 0  |                [A,B]
        # seq 1  |        [C,D]
        # seq 2  |                         [E,F]
        #        +-----------------------------------------------
        # seq_ids:      cache_ids: (n_active_tokens, n_seqs)     values: (n_active_tokens, n_seqs, n_heads, d_head)
        # seq 0         [[6,7],                                  [[A,B],
        # seq 1          [3,4],                                   [C,D],
        # seq 2          [9,10]]                                  [E,F]]
        #
        keys_r = hlo.reshape(keys, [n_seqs, kv_hidden_size])
        vals_r = hlo.reshape(vals, [n_seqs, kv_hidden_size])

        indices = attention_utils.update_indices_decode(cached_keys, cache_ids, neuron_config)
        indices = hlo.transpose(indices, 0, 1)

        scatter_dims = dict(update_window_dims=[1],
                            inserted_window_dims=[0],
                            scatter_dims_to_operand_dims=[0],
                            index_vector_dim=1)
        updated_keys = hlo.scatter(cached_keys_r, indices, keys_r, scatter_dims=scatter_dims, to_apply=assign_func)
        updated_vals = hlo.scatter(cached_vals_r, indices, vals_r, scatter_dims=scatter_dims, to_apply=assign_func)

    elif n_active_tokens == n_positions and n_seqs > n_active_seqs:
        # cache (2D): [n_positions * n_seqs, n_kv_heads * d_head]
        #        +-0-1-2-3-4-5-----------------------------------
        # seq 0  |[x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x]
        # seq 1  |[A,B,C,D,E,F] <- insert new sequence here
        # seq 2  |[y,y,y,y,y,y,y,y,y,y,y,y,y,y,y,y,y,y,y,y]
        #        +-----------------------------------------------
        # seq_ids:      cache_ids: (n_active_tokens, n_seqs)     values: (n_active_tokens, n_seqs, n_heads, d_head)
        # seq 1         [[0,1,2,3,4,5]]                          [[A,B,C,D,E,F]]
        keys_r = hlo.reshape(keys, [n_active_tokens, kv_hidden_size])
        vals_r = hlo.reshape(vals, [n_active_tokens, kv_hidden_size])

        indices = attention_utils.update_indices_context(cached_keys, cache_ids, start_ids, neuron_config)

        # For prefill, assuming n_active_seqs == 1, due to KV cache layout issue.
        assert n_active_seqs == 1, "n_active_seqs is expected to be 1 for 2D cache_ids"

        scatter_dims = dict(update_window_dims=[1],
                            inserted_window_dims=[0],
                            scatter_dims_to_operand_dims=[0],
                            index_vector_dim=1)
        updated_keys = hlo.scatter(cached_keys_r, indices, keys_r, scatter_dims=scatter_dims, to_apply=assign_func)
        updated_vals = hlo.scatter(cached_vals_r, indices, vals_r, scatter_dims=scatter_dims, to_apply=assign_func)

    elif n_active_tokens > 1 and n_active_tokens < n_positions:
        # Speculative forward: n_active_tokens > 1 and < n_positions
        # similar to case above, but modifies a K-token chunk (K > 1) to one of the sequences in the batch
        # cache (2D): [n_positions * n_seqs, n_kv_heads * d_head]
        #        +-0-1-2-3-4-5-----------------------------------
        # seq 0  |[x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x]
        # seq 1  |[y,y,y,y,y,y,y,y,y,A,B,C,D,E,F] <- Modify 5 tokens to this sequence
        # seq 2  |[z,z,z,z,z,z,z,z,z,z,z,z,z,z,z,z,z,z,z]
        #        +-----------------------------------------------
        # seq_ids:      cache_ids: (n_active_tokens, n_seqs)     values: (n_active_tokens, n_seqs, n_heads, d_head)
        # seq 1         [[45,46,47,48,49,50]]                    [[A,B,C,D,E,F]]
        n_active_tokens, batch_size, n_head, d_head = keys.sizes
        keys_r = hlo.reshape(keys, [n_active_tokens * batch_size, n_head, d_head])
        vals_r = hlo.reshape(vals, [n_active_tokens * batch_size, n_head, d_head])

        indices = attention_utils.update_indices_speculative(cached_keys, cache_ids, start_ids, neuron_config)

        updated_keys, updated_vals = hlo.reshape_and_cache(keys_r, vals_r, cached_keys, cached_vals, indices)

    else:
        raise NotImplementedError(f"Updating 2D cache_ids is not implemented for "
                                  f"n_active_tokens={n_active_tokens}, n_positions={n_positions}, "
                                  f"n_seqs={n_seqs}, n_active_seqs={n_active_seqs}.")

    return updated_keys, updated_vals


def update_cache(cache, cache_ids, values):
    """
    Cache[I] = X
    """
    dtype = cache.dtype
    cache_ids_dtype = cache_ids.dtype
    # 1D cache_ids
    scatter_dims = dict(update_window_dims=[1,2,3],
                        inserted_window_dims=[0],
                        scatter_dims_to_operand_dims=[0],
                        index_vector_dim=1)
    assign_func = hlo.gen_assign_func(dtype)
    updated = hlo.scatter(cache, cache_ids, values, scatter_dims=scatter_dims, to_apply=assign_func)
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


def score(query, keys, tp_degree=None, n_kv_heads=0, neuron_config=None):
    """
    Compute the attention score by combining scaled-query & keys.

    S = Q @ K

    If n_kv_heads != 0, uses multi-query, multi-group attention.
    NOTE: Since we may pad along head dimension,
          tp_degree argument is required to be an integer for grouped-query attention models.
    """
    # Check K/V cache layout
    bsh_cache_layout = False
    if neuron_config is not None:
        bsh_cache_layout = neuron_config.cache_layout == constants.LAYOUT_BSH

    # Check for MQA/GQA attention
    if n_kv_heads != 0:
        _, _, n_kv_heads_tp, _ = keys.sizes
        _, _, n_heads_tp, _ = query.sizes
        n_repeats = n_heads_tp // n_kv_heads_tp
        keys = hlo.repeat_kv(keys, n_repeats=n_repeats, repeat_dim=2)

    # Q @ K
    batch_dimensions = [0, 2] if bsh_cache_layout else [1, 2]
    dot_dims = dict(lhs_contracting_dimensions=[3],
                    lhs_batch_dimensions=batch_dimensions,
                    rhs_contracting_dimensions=[3],
                    rhs_batch_dimensions=batch_dimensions)

    result_dot = hlo.dot_general(query, keys, dimension_numbers=dot_dims)

    return result_dot

def sparse_attn_mask(score, mask, constant_value=-30000):
    """
    Masks the computed attention scores with a sparse attention mask. This method
    has different assumptions to the mask shape from the mask() method below.

    score = masked_fill(score, mask, constant_value)
    """
    dtype = score.dtype
    score_sizes = score.sizes

    # Note: This value can cause NaN issues if it is too large
    masked_val = hlo.full(constant_value, dtype, score_sizes)

    # We accept two mask shapes: [q_seq_len, kv_seq_len], or [nheads, q_seq_len, kv_seq_len]
    bs, nheads, q_seq_len, kv_seq_len = score_sizes
    assert (tuple(mask.sizes) == (q_seq_len, kv_seq_len)) or (tuple(mask.sizes) == (nheads, q_seq_len, kv_seq_len)), \
        f'Expecting sparse mask shape of ({q_seq_len}, {kv_seq_len}) or ({nheads}, {q_seq_len}, {kv_seq_len}), but got {mask.sizes}!'

    if len(mask.sizes) == 2:
        mask_br = hlo.broadcast(mask, out_dim_size=score_sizes, broadcast_dimensions=[2, 3])
    else:
        mask_br = hlo.broadcast(mask, out_dim_size=score_sizes, broadcast_dimensions=[1, 2, 3])
    score = dtype[score_sizes].Select(mask_br, score, masked_val)
    return score


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
        mask_br = hlo.broadcast(mask, score_sizes, [0, 2])
    else:
        if shard_over_batch:
            assert isinstance(tp_degree, int), \
                f"tp_degree ({tp_degree}) is required to be an integer for shard-over-batch."
            zero = scribe.s32.Constant(constant_value=0)
            n_seqs_per_nc = score_sizes[0]
            assert n_seqs_per_nc == mask.sizes[0] // tp_degree, f"invalid n_seqs_per_nc ({n_seqs_per_nc}) vs mask_sizes ({mask.sizes})"
            mask = hlo.dynamic_slice_along(mask, dim=0, start=zero, size=n_seqs_per_nc)
        # broadcast from [n_seqs, n_active_tokens, n_positions] to [n_seqs, n_heads, n_active_tokens, n_positions]
        mask_br = hlo.broadcast(mask, score_sizes, [0, 2, 3])
    score = dtype[score_sizes].Select(mask_br, score, large_neg_br)
    return score


def context(past_scores, active_score, past_values, active_values, sparse_mask=None,
            n_kv_heads=0, dtype=None, neuron_config=None, tp_degree=None):
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

    shard_over_batch = False
    bsh_cache_layout = False
    if neuron_config is not None:
        shard_over_batch = neuron_config.group_query_attention == constants.GQA.SHARD_OVER_BATCH
        bsh_cache_layout = neuron_config.cache_layout == constants.LAYOUT_BSH

    n_seqs, n_heads, n_active_tokens, n_active_tokens = active_score_sizes = active_score.sizes
    n_seqs, n_heads, n_active_tokens, n_positions = past_scores.sizes
    if shard_over_batch:
        n_positions, n_seqs_per_nc, n_kv_heads, d_head = past_values.sizes
        n_seqs = n_seqs_per_nc * tp_degree
        n_heads_tp = n_heads // tp_degree
        reduce_sizes = n_seqs_per_nc, n_heads, n_active_tokens
    else:
        if bsh_cache_layout:
            n_seqs, n_positions, n_kv_heads_tp, d_head = past_values.sizes
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
        past_prob = sparse_attn_mask(past_prob, sparse_mask, constant_value=0)

    # Ca = Pa @ Va
    # Cp = Pp @ Vp
    # C = Ca + Cp
    dot_dims = dict(lhs_contracting_dimensions=[3],
                    lhs_batch_dimensions=[0, 1],
                    rhs_contracting_dimensions=[0],
                    rhs_batch_dimensions=[1, 2])
    denom = dtype[denom.sizes].Convert(denom)

    if n_kv_heads != 0:
        if shard_over_batch:
            n_heads = n_heads_tp * tp_degree
            n_repeats = n_heads // n_kv_heads
        else:
            _, n_heads_tp, *_ = past_prob.sizes
            _, _, n_kv_heads_tp, *_ = past_values.sizes
            n_repeats = n_heads_tp // n_kv_heads_tp

        # values layout: (n_positions, n_seqs_per_nc, n_kv_heads, d_head) -> repeat_dim=2
        past_values = hlo.repeat_kv(past_values, n_repeats=n_repeats, repeat_dim=2)
        active_values = hlo.repeat_kv(active_values, n_repeats=n_repeats, repeat_dim=2)

    # lhs (past_prob): (n_seqs, n_heads, n_active_tokens, n_positions)
    # rhs (value):
    # - SBH cache layout: (n_positions, n_seqs, n_heads, d_head)
    # - BSH cache layout: (n_seqs, n_positions, n_heads, d_head)
    rhs_contracting_dimensions = [1] if bsh_cache_layout else [0]
    rhs_batch_dimensions = [0, 2] if bsh_cache_layout else [1, 2]
    dot_dims = dict(lhs_contracting_dimensions=[3],
                lhs_batch_dimensions=[0, 1],
                rhs_contracting_dimensions=rhs_contracting_dimensions,
                rhs_batch_dimensions=rhs_batch_dimensions)

    output_dot = hlo.dot_general(past_prob, past_values, dimension_numbers=dot_dims)
    active_output_dot = hlo.dot_general(active_prob, active_values, dimension_numbers=dot_dims)
    output = hlo.add(output_dot, active_output_dot)

    if shard_over_batch:
        # concat along batch dimension and split along head dimension
        output = hlo.all_to_all(output, split_dim=1, concat_dim=0, tp_degree=tp_degree)
        denom = hlo.all_to_all(denom, split_dim=1, concat_dim=0, tp_degree=tp_degree)

    sizes = n_seqs, n_heads_tp, n_active_tokens, d_head
    denom_br = dtype[sizes].Broadcast(denom, dimensions=[0, 1, 2])
    output = dtype[sizes].Divide(output, denom_br)
    sizes = n_active_tokens, n_seqs, n_heads_tp, d_head
    output = dtype[sizes].Transpose(output, dimensions=[2, 0, 1, 3])
    return output


def context_combined(score, values, sparse_mask=None, n_kv_heads=0, dtype=None, tp_degree=None, neuron_config=None):
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
    shard_over_batch = False
    bsh_cache_layout = False
    if neuron_config is not None:
        shard_over_batch = neuron_config.group_query_attention == constants.GQA.SHARD_OVER_BATCH
        bsh_cache_layout = neuron_config.cache_layout == constants.LAYOUT_BSH

    probs = hlo.softmax(score)
    # Apply sparse masks after softmax to help compiler optimization
    if sparse_mask is not None:
        probs = sparse_attn_mask(probs, sparse_mask, constant_value=0)

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
            if bsh_cache_layout:
                n_seqs, _, n_kv_heads_tp, d_head = values.sizes
            else:
                _, n_seqs, n_kv_heads_tp, d_head = values.sizes
            n_repeats = n_heads_tp // n_kv_heads_tp
        values = hlo.repeat_kv(values, n_repeats=n_repeats, repeat_dim=2)

    rhs_contracting_dimensions = [1] if bsh_cache_layout else [0]
    rhs_batch_dimensions = [0, 2] if bsh_cache_layout else [1, 2]
    dot_dims = dict(
        lhs_contracting_dimensions=[3],
        lhs_batch_dimensions=[0, 1],
        rhs_contracting_dimensions=rhs_contracting_dimensions,
        rhs_batch_dimensions=rhs_batch_dimensions
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
            result = hlo.all_to_all(result, split_dim=1, concat_dim=0, tp_degree=tp_degree)
        else:
            result_sizes = n_seqs, n_heads_tp, n_active_tokens, d_head
            result = hlo.reshape(result, result_sizes)

    sizes = n_active_tokens, n_seqs, n_heads_tp, d_head
    result = dtype[sizes].Transpose(result, dimensions=[2, 0, 1, 3])
    return result


def output(
    context: 'HloShape',
    out_weight: 'HloShape',
    out_scales: 'HloShape',
    out_bias: 'HloShape',
    tp_degree: int,
    neuron_config: Optional[NeuronConfig] = None,
    transposed: Optional[bool] = False,
):
    """
    The output projection of a transformer applied to the attention context.

    O = (C @ wO) + bO

    Arguments:
        context: Attention context.
        out_weight: Model attention outout projection weight.
        out_scales: Scales to "rescale" the quantized weight after it's
            multiplied where W_f = W_q * scales.
        out_bias: Model attention outout projection bias.
        tp_degree: Tensor parallelism degree.
        neuron_config: NeuronConfig object that specifies the quantization and
            collectives configurations.
        transposed: Whether the weight is transposed.

    Implementation details:

        2D out_weight case:
            Weight shape if transposed:
                [d_head * n_heads_tp, hidden]
            else:
                [hidden, d_head * n_heads_tp]

            Dot if transposed:
                (s * b, padded_h) @ (padded_h, h) contract=(1, 0) => (s * b, h)
            else:
                (s * b, padded_h) @ (h, padded_h) contract=(1, 1) => (s * b, h)

        3D out_weight case
            Weight shape if transposed:
                [d_head, n_heads_tp, hidden]
            else:
                [hidden, d_head, n_heads_tp]

            Dot if transposed:
                (s * b, d_head, n_heads_tp) @ (d_head, n_heads_tp, h) contract=((1, 2), (0, 1)) => (s * b, h)
            else:
                (s * b, d_head, n_heads_tp) @ (h, d_head, n_heads_tp) contract=((1, 2), (1, 2)) => (s * b, h)
    """
    dtype = context.dtype
    n_active_tokens, n_seqs, n_heads_tp, d_head = context.sizes

    if transposed:
        *_, hidden_size = out_weight.sizes
    else:
        hidden_size, *_ = out_weight.sizes
    hidden_sizes = hidden_size, n_active_tokens, n_seqs

    enable_quantize = neuron_config and neuron_config.quant
    if enable_quantize:
        out_weight = hlo.cast(out_weight, dtype)

    three_dims = len(out_weight.sizes) == 3

    if three_dims:
        result_sizes = n_active_tokens * n_seqs, n_heads_tp, d_head
    else:
        result_sizes = n_active_tokens * n_seqs, n_heads_tp * d_head

    result = hlo.reshape(context, result_sizes)

    if three_dims:
        # (s * b, n_heads_tp, d_head) -> (s * b, d_head, n_heads_tp)
        result = hlo.permute(result, (0, 2, 1))

    if three_dims:
        if transposed:
            lhs_contract_dims = [1, 2]
            rhs_contract_dims = [0, 1]
        else:
            lhs_contract_dims = [1, 2]
            rhs_contract_dims = [1, 2]
    else:
        if transposed:
            lhs_contract_dims = [1]
            rhs_contract_dims = [0]
        else:
            lhs_contract_dims = [1]
            rhs_contract_dims = [1]

    result = hlo.dot_add(
        lhs=result,
        rhs=out_weight,
        bias=out_bias,
        lhs_contracting_dimension=lhs_contract_dims,
        rhs_contracting_dimension=rhs_contract_dims,
        bias_dimension=1,
        scales=out_scales,
        neuron_config=neuron_config,
    )

    bsh_collective = neuron_config and neuron_config.collectives_layout == LAYOUT_BSH
    bsh_output = neuron_config and neuron_config.attention_layout == LAYOUT_BSH

    if bsh_output or bsh_collective:
        # (s * b, h) => (b, s, h)
        result = hlo.reshape(result, (n_active_tokens, n_seqs, hidden_size))
        result = hlo.transpose(result, 0, 1)
    else:
        # (s * b, h) => (h, s, b)
        result = hlo.transpose(result, 0, 1)
        result = hlo.reshape(result, hidden_sizes)

    dtype, replica_groups = utils.parse_dtype_replica_groups(neuron_config, tp_degree)
    result = hlo.all_reduce_sum(result, tp_degree, dtype=dtype, replica_groups=replica_groups)

    # Transpose back to HSB if applicable
    if bsh_collective and not bsh_output:
        return hlo.permute(result, (2, 1, 0))
    return result


def flash_attention(query, key, value):

    n_active_tokens, batch_size, n_kv_heads_tp, d_head = query.sizes

    flash_attention_nki_compatiblilty = (n_active_tokens % 2048 == 0 and n_active_tokens >= 8192 and query.sizes[2] == key.sizes[2])
    context = None

    if flash_attention_nki_compatiblilty:

        # query shape is (n_active_tokens, n_seqs, n_kv_heads_tp, d_head)
        # expected by nki flash_fwd (batch_size, n_kv_heads_tp, d_head, n_active_tokens)
        n_kv_heads_tp = query.sizes[2]

        query_nki = hlo.permute(query, [1, 2, 3, 0])
        key_nki = hlo.permute(key, [1, 2, 3, 0])
        value_nki = hlo.permute(value, [1, 2, 0, 3]) # shape (batch_size, n_kv_heads_tp, n_active_tokens, d_head)

        context_nki_shape = nki_call(attention_utils.wrapper_flash_attention_nki, query_nki, key_nki, value_nki, grid=[batch_size, n_kv_heads_tp], output_HloShapes=[query.dtype[batch_size, n_kv_heads_tp, n_active_tokens, d_head]])

        # nki flash_fwd output shape (n_seqs, n_kv_heads_tp, n_active_tokens, d_head)
        context = hlo.permute(context_nki_shape, [0, 2, 1, 3])

    elif n_active_tokens >= 8192 and n_active_tokens % 2048 != 0:
        logging.warning("Flash Attention is not active. context length should be a multiple of 2k tokens and larger than 8k in order to use flash attention.")

    return context


def context_shard_over_seq(neuron_config, past_scores, active_score, past_values, active_values,
                        core_id, past_mask, active_mask, n_kv_heads=0, sparse_mask=None, dtype=None,
                        shard_over_batch=False, tp_degree=None, slice_output=False):
    """
    Context method with sharding over sequence under a GQA scenario.
    """
    # Check the conditions on sharding over seq
    # assert tp_degree % self.config.num_q_heads == 0
    assert not shard_over_batch, "Cannot shard over both batch and seq dimensions!"
    assert sparse_mask is None, "Not supposed to be used for context encoding for now!"

    if neuron_config and neuron_config.attention_layout == constants.LAYOUT_BSH:
        import transformers_neuronx.layers.attention as attention
    else:
        import transformers_neuronx.layers.attention_hsb as attention

    if dtype == None:
        dtype = active_score.dtype
    scribe = active_score.scribe
    f32 = scribe.f32

    n_seqs, n_heads_tp, n_active_tokens, n_active_tokens = active_score_sizes = active_score.sizes
    n_seqs, _, n_active_tokens, _ = past_scores.sizes
    _, n_seqs, n_kv_heads_tp, d_head = past_values.sizes
    reduce_sizes = n_seqs, n_heads_tp, n_active_tokens

    # How many cores should compute each head collectively
    # assert tp_degree % n_heads_tp == 0, "Each head must be distributed to the same number of cores!"
    # All cores that hold the KV cache for the same head should communicate here
    cores_per_kv_head = tp_degree // neuron_config.num_kv_heads
    replica_groups = utils.build_replica_groups(num_groups=neuron_config.num_kv_heads,
                                                group_size=cores_per_kv_head)
    # Upcast to f32 before computation
    past_scores = hlo.cast(past_scores, f32)
    active_score = hlo.cast(active_score, f32)

    # Compute maximum of both past_scores and active_scores
    minus_inf = f32.Constant(constant_value=float('-inf'))
    max_func = hlo.gen_max_func(f32)
    max_past_score = f32[reduce_sizes].Reduce(past_scores, minus_inf, dimensions=[3], to_apply=max_func)
    max_active_score = f32[reduce_sizes].Reduce(active_score, minus_inf, dimensions=[3], to_apply=max_func)
    max_score_per_core = f32[reduce_sizes].Maximum(max_past_score, max_active_score)
    # We use the flashattention trick, compute the sum of scores locally and rescale later
    # Shift the scores by the local max for now
    max_score_per_core_br = hlo.broadcast(max_score_per_core, past_scores.sizes, broadcast_dimensions=[0, 1, 2])
    max_score_per_core_br_active = hlo.broadcast(max_score_per_core, active_score_sizes, broadcast_dimensions=[0, 1, 2])
    past_score_shifted = hlo.subtract(past_scores, max_score_per_core_br)
    active_score_shifted = hlo.subtract(active_score, max_score_per_core_br_active)
    # Compute the local sum: L_i = sum(exp(s_i - m_i))
    exp = hlo.exp(past_score_shifted)
    active_exp = hlo.exp(active_score_shifted)
    zero = f32.Constant(constant_value=0)
    add_func = hlo.gen_add_func(f32)
    past_denom = f32[reduce_sizes].Reduce(exp, zero, dimensions=[3], to_apply=add_func)
    active_denom = f32[reduce_sizes].Reduce(active_exp, zero, dimensions=[3], to_apply=add_func)
    denom_per_core = hlo.add(past_denom, active_denom)

    # Communication 2: send the local max and local denom around
    # First concatenate them together so we can use one all-gather for them both
    # The two tensors are supposed to have the same shape
    payload = hlo.concatenate((max_score_per_core, denom_per_core), dimension=1)
    comm_res = hlo.all_gather(payload, dim=1, tp_degree=tp_degree, replica_groups=replica_groups)
    comm_res_reshaped = hlo.reshape(comm_res, (n_seqs, cores_per_kv_head, 2, n_heads_tp, n_active_tokens))
    all_max_scores = hlo.slice_along(comm_res_reshaped, dim=2, limit=1, start=0)
    all_denoms = hlo.slice_along(comm_res_reshaped, dim=2, limit=2, start=1)
    # Each core is now handling multiple Q heads, we constrain our reduce to be in the same Q head
    # After this step we get the global max M on each core, and all the local sums on each core
    all_max_scores = hlo.reshape(all_max_scores, (n_seqs, cores_per_kv_head, n_heads_tp, n_active_tokens))
    max_score = hlo.reduce_max(all_max_scores, dim=1, keepdim=False) # (n_seqs, n_heads_tp, n_active_tokens)
    max_score_br = hlo.broadcast(max_score, all_max_scores.sizes, broadcast_dimensions=[0, 2, 3])
    all_denoms = hlo.reshape(all_denoms, (n_seqs, cores_per_kv_head, n_heads_tp, n_active_tokens))
    # Compute the global denominator L = sum(exp(m_i - M) * L_i)
    # Notice that the softmax has an additional 1 in the denominator
    scaling_factor = hlo.exp(hlo.subtract(all_max_scores, max_score_br))
    mult_res = hlo.multiply(all_denoms, scaling_factor)
    denom = hlo.reduce_sum(mult_res, dim=1, keepdim=False) # (n_seqs, n_heads_tp, n_active_tokens)
    # denom = hlo.add(denom, 1.0)
    # Recompute the scores with the updated max
    # kgopalsw: TODO: can rescale the past_exp wth scaling factor
    max_score_br = hlo.broadcast(max_score, past_scores.sizes, broadcast_dimensions=[0, 1, 2])
    max_score_br_active = hlo.broadcast(max_score, active_score_sizes, broadcast_dimensions=[0, 1, 2])
    past_score_shifted = hlo.subtract(past_scores, max_score_br)
    active_score_shifted = hlo.subtract(active_score, max_score_br_active)
    # Cast the scores back to the original datatype
    exp = hlo.exp(past_score_shifted)
    active_exp = hlo.exp(active_score_shifted)
    past_prob = hlo.cast(exp, dtype)
    active_prob = hlo.cast(active_exp, dtype)
    # Add an additional step of masking after the exp
    past_prob = attention.mask(past_prob, past_mask, tp_degree=tp_degree, shard_over_batch=shard_over_batch, constant_value=0)
    active_prob = attention.mask(active_prob, active_mask, tp_degree=tp_degree, shard_over_batch=shard_over_batch, constant_value=0)

    # Ca = Pa @ Va
    # Cp = Pp @ Vp
    # C = Ca + Cp
    # lhs (past_prob): (n_seqs, n_heads, n_active_tokens, n_positions)
    # rhs (value): (n_positions, n_seqs, n_heads, d_head)
    dot_dims = dict(lhs_contracting_dimensions=[3],
                lhs_batch_dimensions=[0, 1],
                rhs_contracting_dimensions=[0],
                rhs_batch_dimensions=[1, 2])
    if n_kv_heads != 0:
        _, n_heads_tp, *_ = past_prob.sizes
        _, _, n_kv_heads_tp, *_ = past_values.sizes
        n_repeats = n_heads_tp // n_kv_heads_tp
        past_values = hlo.repeat_kv(past_values, n_repeats=n_repeats, repeat_dim=2)
        active_values = hlo.repeat_kv(active_values, n_repeats=n_repeats, repeat_dim=2)
    output_dot = hlo.dot_general(past_prob, past_values, dimension_numbers=dot_dims)
    active_output_dot = hlo.dot_general(active_prob, active_values, dimension_numbers=dot_dims)
    output = hlo.add(output_dot, active_output_dot)

    sizes = n_seqs, n_heads_tp, n_active_tokens, d_head
    denom_br = hlo.broadcast(denom, sizes, broadcast_dimensions=[0, 1, 2])
    denom_br = hlo.cast(denom_br, dtype)
    output = hlo.divide(output, denom_br) # output is f16, denom_br is f32

    # Communication 3: send the results of other Q heads back to their corresponding cores
    # Also gather the results of the current Q head from other cores
    # Ideally we should use a reduce-scatter here, but we don't have it yet
    # Using an all-gather to do the job for now
    # The replica group includes all cores handling the same KV head, same as above
    all_outputs = hlo.all_gather(output, 1, tp_degree, replica_groups)
    # Reshape and do the reduce
    all_outputs = hlo.reshape(all_outputs, (n_seqs, cores_per_kv_head, n_heads_tp, n_active_tokens, d_head))
    all_outputs_sum = hlo.reduce_sum(all_outputs, dim=1, keepdim=False)
    # Choose the ones that corresponds to this core
    # The order of tensors in the all-gather result is consistent with the order in the replica group
    cores_per_q_head = tp_degree // neuron_config.num_q_heads
    cores_per_kv_head = tp_degree // neuron_config.num_kv_heads
    if cores_per_q_head:
        # handle casese where we have single q per core or q is replicated 
        group_size = cores_per_kv_head // cores_per_q_head
        q_head_id = hlo.remainder(hlo.remainder(core_id, cores_per_kv_head), group_size)
        size = 1
    else:
        # cases for mulitple q heads per core, group_size is equal to cores_per_kv_head
        group_size = tp_degree // neuron_config.num_kv_heads  
        q_head_id = hlo.multiply(hlo.remainder(core_id, group_size), group_size)
        size = neuron_config.num_q_heads // tp_degree
    # must reshape to a scalar
    q_head_id = q_head_id.dtype[[]].Reshape(q_head_id)
    if slice_output:
        output = hlo.dynamic_slice_along(all_outputs_sum, dim=1, start=q_head_id, size=size)
    else:
        output = all_outputs_sum
    output = hlo.permute(output, dimensions=[2, 0, 1, 3])
    # Each core now has a partial result. In output projection, result for each head is
    # multiplied with its corresponding weights, and then an all-reduce is used to sum
    # results for all heads together.
    # We need a scaling here because multiple cores hold the same result
    if cores_per_q_head:
        output = hlo.divide(output, cores_per_q_head)
    return output


def convert_attn_mask_and_cache_id(cache_ids, core_id, config,
                                batch_size=1, is_context_encoding=False):
    """
    Convert normal cache IDs to the format suitable for sharded KV cache, and create proper attention
    masks. Since each Q/KV head can be distributed to multiple cores, each core will have a
    different mask and cache ID.

    In this version, the KV cache of all KV heads is evenly split across all cores. Tokens are
    written to the KV caches in a strided way: token 0 goes to core 0's cache, token 1 goes to core
    1's cache, etc. When computing active tokens, each core is in charge of tokens that are written
    to it's cache.

    For tokens that should not be written to the current core's KV cache, the cache ID for this token
    is set to cache_size. We need 1 or more garbage entries in the KV cache for this purpose.
    """
    assert len(cache_ids.sizes) == 1, "Assuming 1D cache IDs!"
    # assert n_heads_tp == 1, "Assuming each core only process 1 Q head!"
    
    n_active_tokens = cache_ids.sizes[0]
    cores_per_kv_head = config.tp_degree // config.num_kv_heads
    cache_size = config.n_positions // cores_per_kv_head
    pred = cache_ids.scribe.pred
    dtype = cache_ids.dtype
    # Real cache ID = raw cache ID // the number of cores that hold a single head's KV cache
    num_cache_splits = cores_per_kv_head
    real_cache_ids = hlo.divide(cache_ids, num_cache_splits)
    # Default cache ID = cache_size
    default_cache_ids = hlo.full(cache_size, dtype, real_cache_ids.sizes)
    # Now mask out the entries that should not go to this core's cache
    target_core_ids = hlo.remainder(cache_ids, num_cache_splits)
    core_id_cast = hlo.cast(core_id, dtype)
    curr_core_id_in_head = hlo.remainder(core_id_cast, num_cache_splits)
    curr_core_id_in_head = hlo.broadcast(curr_core_id_in_head, target_core_ids.sizes, [0])
    mask = hlo.compare(target_core_ids, curr_core_id_in_head, "EQ")
    converted_cache_ids = dtype[cache_ids.sizes].Select(mask, real_cache_ids, default_cache_ids)

    # Generate masks
    if is_context_encoding:
        # We don't need active mask for context encoding
        converted_active_mask = None
        # Prior mask is simpler for context encoding
        converted_mask = hlo.tril_mask(pred, (n_active_tokens, n_active_tokens))
        converted_mask = hlo.broadcast(converted_mask, (batch_size, n_active_tokens, n_active_tokens), [1, 2])
        return (converted_cache_ids, converted_mask)
    else:
        converted_mask_size = batch_size, n_active_tokens, cache_size

        # For prior mask, we compute how many tokens are there in this core's KV cache
        num_processed_tokens = hlo.reduce_min(cache_ids, dim=0, keepdim=True)
        core_id_in_head = hlo.remainder(core_id_cast, num_cache_splits)
        num_tokens_on_core = hlo.divide(hlo.subtract(hlo.add(num_processed_tokens, num_cache_splits-1), core_id_in_head), num_cache_splits)
        # Use Iota to generate the mask
        iota = dtype[converted_mask_size].Iota(dimensions=[2])
        num_tokens_on_core_br = hlo.broadcast(num_tokens_on_core, converted_mask_size, [2])
        converted_mask = hlo.less(iota, num_tokens_on_core_br)

        # Construct the active mask based on the rule above, each core is in charge of tokens
        # that are written to its own cache
        converted_active_mask = hlo.tril_mask(pred, (n_active_tokens, n_active_tokens))
        converted_active_mask = hlo.broadcast(converted_active_mask, (batch_size, n_active_tokens, n_active_tokens), broadcast_dimensions=[1, 2])
        mask_br = hlo.broadcast(mask, converted_active_mask.sizes, [2])
        converted_active_mask = hlo.logical_and(converted_active_mask, mask_br)

        return converted_cache_ids, converted_mask, converted_active_mask