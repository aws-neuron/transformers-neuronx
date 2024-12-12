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
from transformers_neuronx import constants
from neuronxcc.nki.kernels.attention import flash_fwd
try:
    from neuronxcc.nki.kernels.attention import attention_isa_kernel
except ImportError:
    from neuronxcc.nki._private_kernels.attention import attention_isa_kernel
from dataclasses import dataclass


def transpose_qkv(query, key, value):
    """
    Transform between BSH and SBH cache layout.

    inputs shapes [n_active_tokens, n_seqs, n_heads_tp, d_head]
    outputs shapes [n_seqs, n_active_tokens, n_heads_tp, d_head]
    """
    query = hlo.transpose(query, 0, 1)
    key = hlo.transpose(key, 0, 1)
    value = hlo.transpose(value, 0, 1)
    return query, key, value


def update_indices_decode(cached_keys, cache_ids, neuron_config=None):
    # Check K/V cache layout
    bsh_cache_layout = False
    if neuron_config is not None:
        bsh_cache_layout = neuron_config.cache_layout == constants.LAYOUT_BSH
    if bsh_cache_layout:
        n_seqs, n_positions, n_kv_heads, d_head = cached_keys.sizes
    else:
        n_positions, n_seqs, n_kv_heads, d_head = cached_keys.sizes
    cache_ids_dtype = cache_ids.dtype
    if bsh_cache_layout:
        # [6,3,9] -> [(0,6),(1,3),(2,9)] -> [6+0*128,3+1*128,9+2*128] -> [6,131,265]
        # cache_ids + iota * n_positions
        n_positions_br = hlo.full(n_positions, cache_ids_dtype, cache_ids.sizes)
        offset = cache_ids_dtype[cache_ids.sizes].Iota(dimensions=[1])
        offset = cache_ids_dtype[cache_ids.sizes].Multiply(offset, n_positions_br)
        indices = cache_ids_dtype[cache_ids.sizes].Add(cache_ids, offset)
    else:
        # [6,3,9] -> [(0,6),(1,3),(2,9)] -> [6*3,3*3+1,9*3+2] -> [18,10,29]
        # cache_ids * n_seqs + iota
        batch_size_br = hlo.full(n_seqs, cache_ids_dtype, cache_ids.sizes)
        indices = cache_ids_dtype[cache_ids.sizes].Multiply(cache_ids, batch_size_br)
        offset = cache_ids_dtype[cache_ids.sizes].Iota(dimensions=[1])
        indices = cache_ids_dtype[cache_ids.sizes].Add(indices, offset)
    return indices


def update_indices_context(cached_keys, cache_ids, start_ids, neuron_config=None):
    # Check K/V cache layout
    bsh_cache_layout = False
    if neuron_config is not None:
        bsh_cache_layout = neuron_config.cache_layout == constants.LAYOUT_BSH
    if bsh_cache_layout:
        n_seqs, n_positions, n_kv_heads, d_head = cached_keys.sizes
    else:
        n_positions, n_seqs, n_kv_heads, d_head = cached_keys.sizes
    cache_ids_dtype = cache_ids.dtype
    if bsh_cache_layout:
        # [0,1,2,3] -> [(1,0),(1,1),(1,2),(1,3)] -> [1*128+0,1*128+1,1*128+2,1*128+3] -> [128,129,130,140]
        # start_ids * n_positions + iota
        n_positions_br = hlo.full(n_positions, cache_ids_dtype, cache_ids.sizes)
        start_ids_br = hlo.broadcast(start_ids, cache_ids.sizes, [1])
        indices = cache_ids_dtype[cache_ids.sizes].Iota(dimensions=[0])
        offset = cache_ids_dtype[cache_ids.sizes].Multiply(start_ids_br, n_positions_br)
        indices = cache_ids_dtype[cache_ids.sizes].Add(indices, offset)
    else:
        # [0,1,2,3] -> [(1,0),(1,1),(1,2),(1,3)] -> [1+0*3,1+1*3,1+2*3,1+3*3] -> [1,4,7,10]
        # start_ids + iota * n_seqs
        batch_size_br = hlo.full(n_seqs, cache_ids_dtype, cache_ids.sizes)
        start_ids_br = hlo.broadcast(start_ids, cache_ids.sizes, [1])
        indices = cache_ids_dtype[cache_ids.sizes].Iota(dimensions=[0])
        indices = cache_ids_dtype[cache_ids.sizes].Multiply(indices, batch_size_br)
        indices = cache_ids_dtype[cache_ids.sizes].Add(indices, start_ids_br)
    return indices


def update_indices_speculative(cached_keys, cache_ids, start_ids, neuron_config=None):
    # Check K/V cache layout: similar to `update_indices_context` above,
    # but handles the case where n_active_tokens > 1 and n_active_tokens < n_positions
    bsh_cache_layout = False
    if neuron_config is not None:
        bsh_cache_layout = neuron_config.cache_layout == constants.LAYOUT_BSH
    if bsh_cache_layout:
        n_seqs, n_positions, n_kv_heads, d_head = cached_keys.sizes
    else:
        n_positions, n_seqs, n_kv_heads, d_head = cached_keys.sizes
    cache_ids_dtype = cache_ids.dtype
    if bsh_cache_layout:
        # start_ids * n_positions + iota
        n_active_tokens, batch_size = cache_ids.sizes
        n_positions_br = hlo.full(n_positions, cache_ids_dtype, cache_ids.sizes)
        start_ids_br = hlo.broadcast(start_ids, cache_ids.sizes, [1])
        offset = hlo.multiply(start_ids_br, n_positions_br)
        indices = hlo.add(indices, cache_ids)
        indices = hlo.reshape(indices, [n_active_tokens * batch_size])
    else:
        # start_ids + cache_ids * n_seqs
        n_active_tokens, batch_size = cache_ids.sizes
        batch_size_br = hlo.full(n_seqs, cache_ids_dtype, cache_ids.sizes)
        start_ids_br = hlo.broadcast(start_ids, cache_ids.sizes, [1])
        indices = hlo.multiply(cache_ids, batch_size_br)
        indices = hlo.add(indices, start_ids_br)
        indices = hlo.reshape(indices, [n_active_tokens * batch_size])
    return indices


def gather_blocks(key_cache, block_tables, neuron_config=None):
    if neuron_config and neuron_config.optimized_paged_attention:
        return gather_blocks_active(key_cache, block_tables)
    else:
        return gather_blocks_all(key_cache, block_tables)


def active_block_tables(block_tables, context_lens, num_active_blocks, neuron_config):
    """
    Inputs:
        block_tables:
        [[149,   0],
         [148,   0],
         [147, 146],
         [145,   0]]

        context_lens:
        [[  6], [ 16], [170], [  6]]

        num_active_blocks: 6

    Expected Outputs:
        active_table:
        [149, 148, 147, 146, 145, 0]

    Algorithm for calculation:
    >>> blocks_cumsum = lax.cumsum((context_lens+block_size-1)//block_size, axis=0)
    >>> seq_iota = jnp.arange(max_num_seqs)
    >>> for block_id in range(num_blocks):
    >>>     seq_mask = blocks_cumsum <= block_id
    >>>     seq_id = jnp.minimum(max_num_seqs-1,jnp.max(jnp.int32(seq_mask)*(seq_iota+1)))
    >>>     seq_start_block_id = 0 if seq_id == 0 else blocks_cumsum[seq_id-1]
    >>>     offset = block_id - seq_start_block_id
    """
    s32 = block_tables.scribe.s32
    f32 = block_tables.scribe.f32
    num_seqs, blocks_per_seq = block_tables.sizes
    block_size = neuron_config.continuous_batching.block_size
    assert num_seqs == context_lens.sizes[0], "invalid shape of context_lens"
    assert len(context_lens.sizes) == 2, "context_lens is expected to be a 2D vector."

    # HACK: need to take broadcasted active table to the output, in order to workaround a known issue.
    # (N need to be greater than 1)
    N = 2

    # input/output initialization
    active_table = hlo.full(0, block_tables.dtype, (num_active_blocks, N))
    block_tables_flat = hlo.reshape(block_tables, (num_seqs * blocks_per_seq, 1))
    block_tables_br = hlo.broadcast(block_tables_flat, out_dim_size=(num_seqs * blocks_per_seq, N), broadcast_dimensions=[0, 1])
    context_lens = hlo.reshape(context_lens, (num_seqs,))

    zero = hlo.reshape(s32.Constant(constant_value=0), (1,))
    assign_func = hlo.gen_assign_func(active_table.dtype)
    scatter_dims = dict(update_window_dims=[1],
                        inserted_window_dims=[0],
                        scatter_dims_to_operand_dims=[0],
                        index_vector_dim=1)

    # blocks_cumsum = cumsum((context_lens+block_size-1)//block_size, axis=0)
    blocks_add = hlo.add(context_lens, block_size-1)
    blocks_div = hlo.cast(hlo.floor(hlo.divide(hlo.cast(blocks_add, f32), block_size)), s32)
    blocks_cumsum = hlo.cumsum(blocks_div, dim=0)

    # seq_iota = jnp.arange(max_seqs)
    seq_iota = hlo.iota(s32, (num_seqs,), [0])
    seq_iota = hlo.add(seq_iota, 1)

    for block_id in range(num_active_blocks):
        seq_mask = hlo.cast(hlo.less_equal(blocks_cumsum, block_id), s32)
        # seq_id = jnp.minimum(max_num_seqs-1,jnp.max(jnp.int32(seq_mask)*(seq_iota+1)))
        seq_id = hlo.minimum(hlo.reduce_max(hlo.multiply(seq_mask, seq_iota), dim=0), num_seqs-1)
        # seq_start_block_id = 0 if seq_id == 0 else blocks_cumsum[seq_id-1]
        blocks_cumsum_slice = hlo.dynamic_slice_along(blocks_cumsum, dim=0, start=hlo.maximum(hlo.subtract(seq_id, 1), 0), size=1)
        seq_id = hlo.reshape(seq_id, (1,))
        seq_start_block_id = hlo.masked_select(hlo.equal(seq_id, 0), zero, blocks_cumsum_slice)
        offset = hlo.subtract(hlo.full(block_id, s32, (1,)), seq_start_block_id)
        src_index = hlo.minimum(hlo.add(hlo.multiply(seq_id, blocks_per_seq), offset), num_seqs * blocks_per_seq - 1)

        dst_index = hlo.full(block_id, s32, (1,))
        block_data = hlo.index_select(block_tables_br, dim=0, index=src_index)
        active_table = hlo.scatter(active_table, dst_index, block_data,
                                   scatter_dims=scatter_dims, to_apply=assign_func)

    # HACK: need to take broadcasted active table to the output, in order to workaround a known issue.
    active_table = hlo.reduce_min(active_table, dim=1)
    active_table = hlo.reshape(active_table, (num_active_blocks,))
    return active_table


def gather_blocks_active(key_cache, block_tables):
    """
    Select KV cache blocks and gather assigned blocks into output buffer.

    Reference design in PyTorch:
        result = torch.index_select(key_cache, dim=0, index=block_table.flatten())
        result = torch.reshape(result, (n_seqs, max_model_len, n_kv_heads, d_head))

    Args:
        key_cache: The KV cache blocks.
            The input shape is [num_blocks, block_size, n_kv_heads, d_head]
        block_tables: The block table that contains block indices for each of the sequences.
            The input shape is [n_active_blocks]

    Returns:
        cached_keys: The selected KV cache blocks.
            The output layout is [n_seqs, max_model_len, n_kv_heads, d_head],
            where max_model_len=max_num_blocks_per_seq*block_size
    """
    num_blocks, block_size, n_kv_heads, d_head = key_cache.sizes
    assert len(block_tables.sizes) == 1, f"invalid block_table input shape."
    n_active_blocks, = block_tables.sizes
    dtype = key_cache.dtype
    hidden_size = n_kv_heads * d_head
    chunk_size = block_size * hidden_size
    key_cache = hlo.reshape(key_cache, (num_blocks, chunk_size))
    cached_keys = hlo.index_select(key_cache, dim=0, index=block_tables)
    cached_keys = hlo.reshape(cached_keys, (n_active_blocks, block_size, n_kv_heads, d_head))
    return cached_keys


def gather_blocks_all(key_cache, block_tables):
    """
    Select KV cache blocks and gather assigned blocks into output buffer.

    Reference design in PyTorch:
        result = torch.index_select(key_cache, dim=0, index=block_table.flatten())
        result = torch.reshape(result, (n_seqs, max_model_len, n_kv_heads, d_head))

    Args:
        key_cache: The KV cache blocks.
            The input shape is [num_blocks, block_size, n_kv_heads, d_head]
        block_tables: The block table that contains block indices for each of the sequences.
            The input shape is [n_seqs, max_num_blocks_per_seq]

    Returns:
        cached_keys: The selected KV cache blocks.
            The output layout is [n_seqs, max_model_len, n_kv_heads, d_head],
            where max_model_len=max_num_blocks_per_seq*block_size
    """
    num_blocks, block_size, n_kv_heads, d_head = key_cache.sizes
    n_seqs, max_num_blocks_per_seq = block_tables.sizes
    dtype = key_cache.dtype
    hidden_size = n_kv_heads * d_head
    chunk_size = block_size * hidden_size
    key_cache = hlo.reshape(key_cache, (num_blocks, chunk_size))
    index = hlo.reshape(block_tables, (n_seqs * max_num_blocks_per_seq,))
    o_sizes = (n_seqs * max_num_blocks_per_seq, chunk_size)
    cached_keys = hlo.index_select(key_cache, dim=0, index=index)
    cached_keys = hlo.reshape(cached_keys, (n_seqs, max_num_blocks_per_seq * block_size, n_kv_heads, d_head))
    return cached_keys


def blockwise_qk_matmul(query, keys, block_to_seq):
    num_seqs, _, num_heads, d_head = query.sizes
    num_blocks, block_size, num_kv_heads, _ = keys.sizes
    o_dtype = query.dtype
    o_sizes = (num_blocks, num_heads, 1, block_size)

    block_to_seq_vec = hlo.reshape(block_to_seq, (num_blocks, 1))
    replicated_queries = gather_blocks_all(query, block_to_seq_vec)
    dot_dims = dict(lhs_contracting_dimensions=[3],
                    lhs_batch_dimensions=[0, 2],
                    rhs_contracting_dimensions=[3],
                    rhs_batch_dimensions=[0, 2])
    output_dot = hlo.dot_general(replicated_queries, keys, dimension_numbers=dot_dims)
    return output_dot


def block_to_seq_indexing(context_lens, num_seqs, num_blocks, block_size):
    """
    Generate block-wise output space with sequence index, such that

      block_to_seq[block_id] = seq_id

    INPUTS:
    context_lens: [3,5,2,0]
    num_seqs: 4
    num_blocks: 6
    block_size: 4

    OUTPUT:
    [0,1,1,2,3,3]

    Algorithm for calculation:
    >>> blocks_cumsum = lax.cumsum((context_lens+block_size-1)//block_size, axis=0)
    >>> seq_iota = jnp.arange(max_num_seqs)
    >>> block_to_seq = jnp.zeros(num_blocks)
    >>> for block_id in range(num_blocks):
    >>>     seq_mask = blocks_cumsum <= block_id
    >>>     seq_id = jnp.minimum(max_num_seqs-1,jnp.max(jnp.int32(seq_mask)*(seq_iota+1)))
    >>>     block_to_seq = block_to_seq.at[block_id].set(seq_id)
    """
    s32 = context_lens.scribe.s32
    f32 = context_lens.scribe.f32
    assert num_seqs == context_lens.sizes[0], "invalid shape of context_lens"
    assert len(context_lens.sizes) == 2, "context_lens is expected to be a 2D vector."

    # hack to work around runtime/compiler issue
    # (N need to be greater than 1)
    N = 2

    # input/output initialization
    block_to_seq_vec = hlo.full(0, context_lens.dtype, (num_blocks, N))
    context_lens = hlo.reshape(context_lens, (num_seqs,))

    zero = hlo.reshape(s32.Constant(constant_value=0), (1,))
    assign_func = hlo.gen_assign_func(block_to_seq_vec.dtype)
    scatter_dims = dict(update_window_dims=[1],
                        inserted_window_dims=[0],
                        scatter_dims_to_operand_dims=[0],
                        index_vector_dim=1)

    # blocks_cumsum = cumsum((context_lens+block_size-1)//block_size, axis=0)
    blocks_add = hlo.add(context_lens, block_size-1)
    blocks_div = hlo.floor(hlo.divide(hlo.cast(blocks_add, f32), block_size))
    blocks_cumsum = hlo.cumsum(blocks_div, dim=0)

    # seq_iota = jnp.arange(max_seqs)
    seq_iota = hlo.iota(s32, (num_seqs,), [0])
    seq_iota = hlo.add(seq_iota, 1)

    for block_id in range(num_blocks):
        seq_mask = hlo.cast(hlo.less_equal(blocks_cumsum, block_id), s32)
        # seq_id = jnp.minimum(max_num_seqs-1,jnp.max(jnp.int32(seq_mask)*(seq_iota+1)))
        seq_id = hlo.minimum(hlo.reduce_max(hlo.multiply(seq_mask, seq_iota), dim=0), num_seqs-1)
        seq_id_br = hlo.broadcast(hlo.reshape(seq_id, (1,1)), out_dim_size=(1, N), broadcast_dimensions=[0, 1])
        dst_index = hlo.full(block_id, s32, (1,))
        block_to_seq_vec = hlo.scatter(block_to_seq_vec, dst_index, seq_id_br,
                                       scatter_dims=scatter_dims, to_apply=assign_func)

    # HACK: need to take broadcasted active table to the output, in order to workaround a known issue.
    block_to_seq_red = hlo.reduce_min(block_to_seq_vec, dim=1)
    block_to_seq_res = hlo.reshape(block_to_seq_red, (num_blocks,))
    return block_to_seq_res


def blockwise_softmax(logits, context_lens, neuron_config, dim=None, tp_degree=1):
    """
    When tp_degree > 1 this function assumes that the softmax operation is
    sharded over the `dim` dimension.
    """
    num_seqs = neuron_config.continuous_batching.max_num_seqs
    block_size = neuron_config.continuous_batching.block_size
    num_active_blocks, *_ = logits.sizes
    rank = len(logits.sizes)
    if dim is None:
        dim = rank - 1
    dims = list(range(rank))
    dims.pop(dim)

    maximum = hlo.reduce_max(logits, dim)
    block_to_seq = block_to_seq_indexing(context_lens, num_seqs, num_active_blocks, block_size)
    maximum_bw = blockwise_reduce_max(maximum, block_to_seq, num_seqs)
    if tp_degree > 1:
        maximum_bw = hlo.all_reduce_max(maximum_bw, tp_degree=tp_degree)
    maximum_br = hlo.broadcast(maximum_bw, logits.sizes, dims)

    difference = hlo.subtract(logits, maximum_br)
    exponential = hlo.exp(difference)

    denominator = hlo.reduce_sum(exponential, dim)
    denominator_bw = blockwise_reduce_sum(denominator, block_to_seq, num_seqs)
    if tp_degree > 1:
        denominator_bw = hlo.all_reduce_sum(denominator_bw, tp_degree=tp_degree)
    denominator_br = hlo.broadcast(denominator_bw, logits.sizes, dims)

    output = hlo.divide(exponential, denominator_br)
    return output


def blockwise_reduce_max(logits, block_to_seq, num_seqs):
    s32 = logits.scribe.s32
    num_blocks, *_ = logits.sizes
    o_dtype = logits.dtype
    o_sizes = logits.sizes
    output = hlo.full(0, o_dtype, o_sizes)
    neg = hlo.full(-3000, o_dtype, o_sizes)
    seq_iota = hlo.broadcast(block_to_seq, o_sizes, [0])
    for seq_id in range(num_seqs):
        seq_full = hlo.full(seq_id, s32, o_sizes)
        seq_mask = hlo.equal(seq_iota, seq_full)
        masked_logits = hlo.masked_select(seq_mask, logits, neg)
        output_br = hlo.broadcast(hlo.reduce_max(masked_logits, dim=0, keepdim=True), o_sizes, [0, 1, 2])
        output = hlo.masked_select(seq_mask, output_br, output)
    return output


def blockwise_reduce_sum(logits, block_to_seq, num_seqs):
    s32 = logits.scribe.s32
    num_blocks, *_ = logits.sizes
    o_dtype = logits.dtype
    o_sizes = logits.sizes
    # NOTE: initialize output with a large number to avoid divide-by-zero.
    output = hlo.full(3000, o_dtype, o_sizes)
    zero = hlo.full(0, o_dtype, o_sizes)
    seq_iota = hlo.broadcast(block_to_seq, o_sizes, [0])
    for seq_id in range(num_seqs):
        seq_full = hlo.full(seq_id, s32, o_sizes)
        seq_mask = hlo.equal(seq_iota, seq_full)
        masked_logits = hlo.masked_select(seq_mask, logits, zero)
        output_br = hlo.broadcast(hlo.reduce_sum(masked_logits, dim=0, keepdim=True), o_sizes, [0, 1, 2])
        output = hlo.masked_select(seq_mask, output_br, output)
    return output


def prior_context(past_scores, past_values,
                  n_kv_heads=0, dtype=None, tp_degree=None,
                  context_lens=None,
                  num_active_blocks=None,
                  neuron_config=None):
    """
    Compute "context" output from the QK score and value projection.

    C = softmax(S) @ V

    If n_kv_heads != 0, uses multi-query, multi-group attention.
    If dtype is None, uses past_scores datatype.
    """

    if dtype == None:
        dtype = past_scores.dtype
    scribe = past_scores.scribe
    f32 = scribe.f32

    bsh_cache_layout = False
    if neuron_config is not None:
        bsh_cache_layout = neuron_config.cache_layout == constants.LAYOUT_BSH

    n_seqs, n_heads, n_active_tokens, n_positions = past_scores.sizes
    if bsh_cache_layout:
        n_seqs, n_positions, n_kv_heads_tp, d_head = past_values.sizes
    else:
        n_positions, n_seqs, n_kv_heads_tp, d_head = past_values.sizes
    reduce_sizes = n_seqs, n_heads, n_active_tokens

    # Upcast to f32 before computation
    past_scores = hlo.cast(past_scores, f32)
    if neuron_config and neuron_config.optimized_paged_attention:
        past_prob = blockwise_softmax(past_scores, context_lens, neuron_config=neuron_config)
    else:
        past_prob = hlo.softmax(past_scores)

    if n_kv_heads != 0:
        _, n_heads_tp, *_ = past_prob.sizes
        _, _, n_kv_heads_tp, *_ = past_values.sizes
        n_repeats = n_heads_tp // n_kv_heads_tp

        # values layout: (n_positions, n_seqs_per_nc, n_kv_heads, d_head) -> repeat_dim=2
        past_values = hlo.repeat_kv(past_values, n_repeats=n_repeats, repeat_dim=2)

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

    # past_prob: (n_seqs, n_heads, n_active_tokens, n_positions)
    # past_values: (n_seqs, n_positions, n_heads, d_head)
    output_dot = hlo.dot_general(past_prob, past_values, dimension_numbers=dot_dims)
    if neuron_config and neuron_config.optimized_paged_attention:
        block_size = neuron_config.continuous_batching.block_size
        selected_block_indices = sample_block_indices(context_lens, num_active_blocks, block_size)
        output_dot = blockwise_tensor_contraction(output_dot, selected_block_indices)

    return output_dot


def sample_block_indices(context_lens, num_active_blocks, block_size):
    f32 = context_lens.scribe.f32
    s32 = context_lens.scribe.s32
    num_seqs = context_lens.sizes[0]

    ## calculate index for index_select
    # INPUT:
    # - context_lens: [3, 9, 5, 0]
    # - block_size: 4
    # OUTPUT:
    # - num_blocks_cumsum: [0, 3, 5, 5]
    # ALGORITHM:
    # - num_blocks: (context_lens + block_size - 1) // block_size
    # [1, 3, 2, 0]
    # - num_blocks_cumsum: cumsum(num_blocks)
    # [1, 4, 6, 6]
    # - output: minimum(num_blocks_cumsum - 1, num_active_blocks - 1)
    # (The extra minimum operator is needed to avoid out-of-bound issues)
    # [0, 3, 5, 5]
    blocks_add = hlo.add(context_lens, block_size-1)
    blocks_div = hlo.cast(hlo.floor(hlo.divide(hlo.cast(blocks_add, f32), block_size)), s32)
    blocks_cumsum = hlo.subtract(hlo.cumsum(blocks_div, dim=0), 1)
    indices = hlo.minimum(blocks_cumsum, num_active_blocks-1)
    indices = hlo.reshape(indices, (num_seqs,))
    return indices

def blockwise_tensor_contraction(input_tensor, selected_block_indices):
    input_cumsum = hlo.cumsum(input_tensor, dim=0)
    selected = hlo.index_select(input_cumsum, 0, selected_block_indices)

    # Get diff out of cumsum output, and concat with first row
    first_row = hlo.slice_along(selected, dim=0, limit=1)
    other_row = hlo.diff(selected, dim=0)
    output = hlo.concatenate([first_row, other_row], dimension=0)
    return output


@dataclass(frozen=True)
class FlashConfig:
  """
    Config class for flash attention with default values
  """
  seq_tile_size:int = 2048
  training:bool=False
  should_transpose_v:bool=False

def wrapper_flash_attention_nki(q, k, v, o, lse=None):
    softmax_scale = 1.0
    config = FlashConfig()
    seed = None
    flash_fwd(q, k, v, seed, o, lse, softmax_scale=softmax_scale, use_causal_mask=True, mixed_precision=True, dropout_p=0.0, config=config)

def wrapper_flash_attention_bir(q, k, v, out, scale=1.0, kernel_name="CausalAttentionMMSoftmaxMMWithoutSwap"):
    attention_isa_kernel(q, k, v, scale, out, kernel_name)
