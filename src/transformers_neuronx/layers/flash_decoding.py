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
from transformers_neuronx import utils
from transformers_neuronx.layers import attention

"""
    Helper functions for shard over sequence / Flash decoding implementaions
"""

def gather_query_group(query, cores_per_kv_head, n_heads, tp_degree):

    # Communication 1: all-gather query from cores
    # Notice that this is not necessary for context encoding because we don't read from the KV cache
    cores_per_q_head = tp_degree // n_heads
    group_size = cores_per_kv_head # note this cores per kv head is already divide by cores_per_q_head
    num_groups = tp_degree // group_size 
    interleave=False
    n_kv_heads = tp_degree // cores_per_kv_head
    interleave = utils.is_attn_node_interleaved(n_heads, n_kv_heads, tp_degree)
    replica_groups = utils.build_replica_groups(num_groups, group_size, interleave=interleave)
    # Query shape: n_active_tokens, n_seqs, n_heads_tp, d_head
    query = hlo.all_gather(query, 2, tp_degree, replica_groups)
    return query

def context(past_scores, active_score, past_values, active_values,
                        core_id, past_mask, active_mask, n_kv_heads=0, n_heads=None, 
                        sparse_mask=None, dtype=None, shard_over_batch=False, tp_degree=None, 
                        neuron_config=None):
    """
    Context method with sharding over sequence under a GQA scenario.
    """
    # Check the conditions on sharding over seq
    assert not shard_over_batch, "Cannot shard over both batch and seq dimensions!"
    assert sparse_mask is None, "Not supposed to be used for context encoding for now!"
    assert n_kv_heads > 0 and n_heads > 0 , "n_kv_heads and n_heads has to be non-zero"


    if dtype == None:
        dtype = active_score.dtype
    scribe = active_score.scribe
    f32 = scribe.f32

    n_seqs, n_heads_tp, n_active_tokens, n_active_tokens = active_score_sizes = active_score.sizes
    n_seqs, _, n_active_tokens, _ = past_scores.sizes
    _, n_seqs, n_kv_heads_tp, d_head = past_values.sizes
    # reduce_sizes: (n_seqs, n_heads_tp, n_active_tokens)

    # How many cores should compute each head collectively
    # All cores that hold the KV cache for the same head should communicate here
    cores_per_kv_head = tp_degree // n_kv_heads
    cores_per_q_head = tp_degree // n_heads
    cores_per_kv_head = cores_per_kv_head // cores_per_q_head if cores_per_q_head else cores_per_kv_head
    if cores_per_kv_head > 1:
        group_size = cores_per_kv_head
        num_groups = tp_degree // group_size
    else:
        # MHA case, assume all cores will have all heads in cache and kv sharded by seq
        num_groups = 1
        group_size = tp_degree
        cores_per_kv_head = tp_degree

    interleave = utils.is_attn_node_interleaved(n_heads=n_heads, n_kv_heads=n_kv_heads,tp_degree=tp_degree)
    replica_groups = utils.build_replica_groups(num_groups=num_groups,
                                                group_size=group_size, interleave=interleave)
    # Upcast to f32 before computation
    past_scores = hlo.cast(past_scores, f32)
    active_score = hlo.cast(active_score, f32)

    # Compute maximum of both past_scores and active_scores
    max_past_score = hlo.reduce_max(past_scores, dim=3)
    max_active_score = hlo.reduce_max(active_score, dim=3)
    max_score_per_core = hlo.maximum(max_past_score, max_active_score)
    # We use the flashattention trick, compute the sum of scores locally and rescale later
    # Shift the scores by the local max for now
    max_score_per_core_br = hlo.broadcast(max_score_per_core, past_scores.sizes, broadcast_dimensions=[0, 1, 2])
    max_score_per_core_br_active = hlo.broadcast(max_score_per_core, active_score_sizes, broadcast_dimensions=[0, 1, 2])
    past_score_shifted = hlo.subtract(past_scores, max_score_per_core_br)
    active_score_shifted = hlo.subtract(active_score, max_score_per_core_br_active)
    # Compute the local sum: L_i = sum(exp(s_i - m_i))
    exp = hlo.exp(past_score_shifted)
    active_exp = hlo.exp(active_score_shifted)
    past_denom = hlo.reduce_sum(exp, dim=3)
    active_denom = hlo.reduce_sum(active_exp, dim=3)
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
    # Recompute the scores with the updated max
    # TODO: kgopalsw: can rescale the past_exp wth scaling factor
    max_score_br = hlo.broadcast(max_score, past_scores.sizes, broadcast_dimensions=[0, 1, 2])
    max_score_br_active = hlo.broadcast(max_score, active_score_sizes, broadcast_dimensions=[0, 1, 2])
    past_score_shifted = hlo.subtract(past_scores, max_score_br)
    active_score_shifted = hlo.subtract(active_score, max_score_br_active)
    # Cast the scores back to the original datatype
    exp = hlo.exp(past_score_shifted)
    active_exp = hlo.exp(active_score_shifted)
    past_prob = hlo.cast(exp, dtype)
    active_prob = hlo.cast(active_exp, dtype)

    # Ca = Pa @ Va
    # Cp = Pp @ Vp
    # C = Ca + Cp
    # lhs (past_prob): (n_seqs, n_heads, n_active_tokens, n_positions)
    # rhs (value): (n_positions, n_seqs, n_heads, d_head)
    dot_dims = dict(lhs_contracting_dimensions=[3],
                lhs_batch_dimensions=[0, 1],
                rhs_contracting_dimensions=[0],
                rhs_batch_dimensions=[1, 2])
    n_repeats = n_heads_tp // n_kv_heads_tp
    if n_repeats > 1:
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

    cores_per_q_head = tp_degree // n_heads
    cores_per_kv_head = tp_degree // n_kv_heads
    if cores_per_q_head:
        # handle casese where we have single q per core or q is replicated 
        group_size = cores_per_kv_head // cores_per_q_head
        size = 1
    else:
        # cases for mulitple q heads per core, group_size is equal to cores_per_kv_head
        group_size = tp_degree // n_kv_heads
        group_size = group_size if group_size > 1 else tp_degree # case when kv_heads > tp_degree
        size = n_heads // tp_degree

    # Communication 3: send the results of other Q heads back to their corresponding cores
    # Also gather the results of the current Q head from other cores
    assert output.sizes[1] == group_size*size , f"n_heads {output.sizes[1]} after gather not matching kv_replication x n_heads_tp {group_size}x {size}"
    apply_fn = hlo.gen_add_func(output.dtype)
    output = hlo.reduce_scatter(output, dim=1, replica_groups=replica_groups, to_apply=apply_fn)
    assert output.sizes[1] == size , f"n_heads post scatter size mismatch, check replica_groups {replica_groups}"

    output = hlo.permute(output, dimensions=[2, 0, 1, 3])
    # Each core now has a partial result. In output projection, result for each head is
    # multiplied with its corresponding weights, and then an all-reduce is used to sum
    # results for all heads together.
    # We need a scaling here because multiple cores hold the same result
    #if cores_per_q_head: # we do zero padding now, so enable once replication is done
    #    output = hlo.divide(output, cores_per_q_head)
    return output


def convert_attn_mask_and_cache_id(cache_ids, start_ids,  core_id, n_positions, cores_per_kv_head=1):
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

    is_2d_cache = len(cache_ids.sizes) > 1
    batch_size = start_ids.sizes[0] if not is_2d_cache else cache_ids.sizes[0]
    n_batches, n_active_tokens = cache_ids.sizes if is_2d_cache else (batch_size, cache_ids.sizes[0])
    seq_dim = 1 if is_2d_cache else 0
    
    is_context_encoding = n_active_tokens == n_positions

    cache_size = n_positions // cores_per_kv_head
    pred = cache_ids.scribe.pred
    dtype = cache_ids.dtype
    # Real cache ID = raw cache ID // the number of cores that hold a single head's KV cache
    num_cache_splits = cores_per_kv_head
    real_cache_ids = hlo.divide(cache_ids, num_cache_splits)
    # Default cache ID = cache_size -1
    default_cache_ids = hlo.full(cache_size - 1, dtype, real_cache_ids.sizes)
    # Now mask out the entries that should not go to this core's cache
    target_core_ids = hlo.remainder(cache_ids, num_cache_splits)
    core_id_cast = hlo.cast(core_id, dtype)
    curr_core_id_in_head = hlo.remainder(core_id_cast, num_cache_splits)
    curr_core_id_in_head_br = hlo.broadcast(curr_core_id_in_head, target_core_ids.sizes, [0])
    mask = hlo.equal(target_core_ids, curr_core_id_in_head_br)
    converted_cache_ids = hlo.masked_select(mask, real_cache_ids, default_cache_ids)

    # Generate masks for context encoding / windowed /speculation
    if is_context_encoding:
        # We don't need active mask for context encoding
        converted_mask, converted_active_mask = hlo.attention_mask(cache_ids, start_ids, n_positions)

        return converted_cache_ids, converted_mask, converted_active_mask
    else:
        # token generation / windowed / speculative 
        converted_mask_size = batch_size, n_active_tokens, cache_size

        # For prior mask, we compute how many tokens are there in this core's KV cache
        if n_active_tokens > 1:
            # Multi-token speculative sampling & windowed attention
            num_processed_tokens = hlo.reduce_min(cache_ids, dim=seq_dim, keepdim=True)
        else:
            num_processed_tokens = cache_ids
        # core_id_in_head = hlo.remainder(core_id_cast, num_cache_splits)
        num_tokens_on_core = hlo.divide(hlo.subtract(hlo.add(num_processed_tokens, num_cache_splits-1),
                                                     hlo.reshape(curr_core_id_in_head, [])), num_cache_splits)

        # Use Iota to generate the mask
        iota = dtype[converted_mask_size].Iota(dimensions=[2])
        num_tokens_on_core_br = hlo.broadcast(num_tokens_on_core, converted_mask_size, [0, 1] if is_2d_cache else [0])
        converted_mask = hlo.less(iota, num_tokens_on_core_br)

        # Construct the active mask based on the rule above, each core is in charge of tokens
        # that are written to its own cache
        converted_active_mask = hlo.tril_mask(pred, (n_active_tokens, n_active_tokens))
        converted_active_mask = hlo.broadcast(converted_active_mask, (batch_size, n_active_tokens, n_active_tokens),
                                              broadcast_dimensions=[1, 2])
        mask_br = hlo.broadcast(mask, converted_active_mask.sizes, [0, 2] if is_2d_cache else [2])
        converted_active_mask = hlo.logical_and(converted_active_mask, mask_br)

        # rhs aligned so we need to account for start_ids as well
        if not is_2d_cache:
            real_start_ids = hlo.divide(start_ids, num_cache_splits)
            start_ids_remainder = hlo.remainder(start_ids, num_cache_splits)
            core_id_in_head_br = hlo.broadcast(curr_core_id_in_head, start_ids.sizes, [0])
            start_core_mask = hlo.less(core_id_in_head_br, start_ids_remainder)
            real_start_ids = hlo.add(real_start_ids, hlo.cast(start_core_mask, dtype=real_start_ids.dtype))
            real_start_ids_br = hlo.broadcast(real_start_ids, converted_mask_size, [0])
            start_mask = hlo.greater_equal(iota, real_start_ids_br)

            converted_mask = hlo.logical_and(converted_mask, start_mask)

        return converted_cache_ids, converted_mask, converted_active_mask


def select_values_within_bound(cache_ids, values, keys, cores_per_kv_head, core_id, dim, n_positions):
    dtype = cache_ids.dtype

    core_id = hlo.reshape(core_id,[])
    num_cache_splits = cores_per_kv_head
    sizes = list(values.sizes)
    cache_sizes = list(cache_ids.sizes)
    is_2d_cache_id = len(cache_ids.sizes) > 1
    n_active_tokens = cache_ids.sizes[-1]
    is_context_encoding = n_active_tokens == n_positions

    # don't slice for token gen
    if cache_ids.sizes[-1] > 1 and is_context_encoding:
        cache_dim = 1 if is_2d_cache_id else 0
        slice_size = sizes[dim] - (num_cache_splits - 1)
        cache_slice_size = cache_sizes[cache_dim] - (num_cache_splits - 1)
        curr_core_id_in_head = hlo.remainder(core_id, num_cache_splits)
        stride = num_cache_splits
        values = hlo.dynamic_slice_along(values,dim,curr_core_id_in_head, slice_size)
        keys = hlo.dynamic_slice_along(keys,dim,curr_core_id_in_head, slice_size)
        cache_ids = hlo.dynamic_slice_along(cache_ids,cache_dim,curr_core_id_in_head, cache_slice_size)
    
        values =  hlo.slice_along(values, dim=dim,limit=slice_size,stride=stride)
        keys =  hlo.slice_along(keys, dim=dim,limit=slice_size,stride=stride)
        cache_ids = hlo.slice_along(cache_ids, dim=cache_dim,limit=cache_slice_size, stride=stride)
        
    return cache_ids, values, keys
