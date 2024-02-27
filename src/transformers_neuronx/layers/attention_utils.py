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


def gather_blocks(key_cache, block_tables):
    """
    Select KV cache blocks and gather assigned blocks into output buffer.

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
