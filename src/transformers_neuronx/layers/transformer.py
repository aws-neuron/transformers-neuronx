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
import logging
from transformers_neuronx import hlo
from transformers_neuronx.constants import LAYOUT_BSH


def inputs(scribe, dtype, batch_size, n_active_tokens, hidden_size, neuron_config=None, tp_degree=None):
    """
    Defines the set of required inputs for all decoder models.

    For any model that requires *more* inputs than the required inputs produced
    by this function, the next parameters must be linearly allocated starting
    at 4. Additonal parameters must also define their own sequence slice
    dimensions (See below).

    Args:
        scribe: The PyHLO scribe object to write operations with.
        dtype: The data type of the hidden state.
        batch_size: The active batch size (may differ from cache batch size)
        n_active_tokens: The number of active tokens to process. During context
            prefill, this will be larger than 1. During autogregressive
            token generation this will be exactly equal to 1.
        hidden_size: The size of the hidden state.
        neuron_config: Optional configurations.

    Returns:
        hidden: The hidden state (Assumed to be embedded on CPU)
        cache_ids: The positions to update in the KV cache. This is 1d when
            using RHS-alignment since all batch lines update the same
            places in the KV cache. This is 2d when LHS-alignment since
            each batch line can update a different offset in the KV cache.
        start_ids: The offset into each batch line. When using
            LHS-alignment, this indicates the start offset. When using
            RHS-alignment, this indicates the batch line to update.
        last_token_id: An integer index (along the sequence dimenson) which
            indicates which is the last token. This is used in the language
            model head to slice the hidden state.
        sequence_slice_dimensions: The dimension of each input tensor which can
            be sliced during token generation.
    """
    s32 = scribe.s32

    if neuron_config and neuron_config.sequence_parallel_norm and n_active_tokens > neuron_config.sequence_parallel_norm_threshold:
        neuron_config.is_sequence_parallel = True
    else:
        neuron_config.is_sequence_parallel = False

    # Multilayer on device embedding will use the already-embedded inputs for the layers NEFF
    # because there is a separate neff for embedding.
    if neuron_config and neuron_config.on_device_embedding:
        hidden_sizes = batch_size, n_active_tokens
    else:
        if neuron_config and neuron_config.attention_layout == LAYOUT_BSH:
            hidden_sizes = batch_size, n_active_tokens, hidden_size
        else: # HASB LAyout
            hidden_sizes = hidden_size, n_active_tokens, batch_size

    if neuron_config.is_sequence_parallel:
        hidden_sizes = list(hidden_sizes)
        hidden_sizes[1] = hidden_sizes[1] // tp_degree
        hidden_sizes = tuple(hidden_sizes)

    hidden = (
        s32[hidden_sizes].Parameter(parameter_number=0) if neuron_config and neuron_config.on_device_embedding
        else
        dtype[hidden_sizes].Parameter(parameter_number=0)
    )
    cache_2d = neuron_config and neuron_config.use_2d_cache_ids
    if cache_2d:
        position_sizes = batch_size, n_active_tokens
        cache_ids = s32[position_sizes].Parameter(parameter_number=1)   # 2d cache_ids
    else:
        cache_ids = s32[n_active_tokens].Parameter(parameter_number=1)  # 1d cache_ids

    if cache_2d and neuron_config.use_1d_query and n_active_tokens>1:
        start_ids = s32[n_active_tokens].Parameter(parameter_number=2)
    elif neuron_config.paged_attention:
        # start_ids will be used as slot_mappings
        if n_active_tokens > 1:
            start_ids = s32[batch_size, n_active_tokens].Parameter(parameter_number=2)
        else:
            start_ids = s32[batch_size].Parameter(parameter_number=2)
    else:
        start_ids = s32[batch_size].Parameter(parameter_number=2)

    # Build parameters for last_token_id and others
    if cache_2d:
        if neuron_config and neuron_config.use_1d_query and n_active_tokens > 1:
            # context encoding: last_token_id is used as prompt_lens
            max_num_seqs = neuron_config.continuous_batching.batch_size_for_shared_caches
            last_token_id = s32[max_num_seqs].Parameter(parameter_number=3)
        elif neuron_config and neuron_config.paged_attention and n_active_tokens == 1:
            # decode with multiple KV cache blocks: last_token_id is used as block_tables
            max_model_len = neuron_config.continuous_batching.max_model_len
            block_size = neuron_config.continuous_batching.block_size
            max_num_blocks_per_seq = (max_model_len + block_size - 1) // block_size
            last_token_id = s32[batch_size, max_num_blocks_per_seq].Parameter(parameter_number=3)
        else:
            # regular token gen
            last_token_id = s32[batch_size].Parameter(parameter_number=3)
    else:
        last_token_id = s32[1].Parameter(parameter_number=3)

    sequence_slice_dimensions = (
        1,                        # hidden        | In both HSB/BSH the sequence dim is 1
        1 if cache_2d else 0,     # cache_ids     | Sequence dim varies based on alignment
        None,                     # start_ids     | Offset is per batch, no slicing required
        0 if cache_2d else None,  # last_token_id | Scalar, no slicing required
    )

    return (hidden, cache_ids, start_ids, last_token_id), sequence_slice_dimensions


def ln_lm_head(tp_degree, hidden, last_token_id, ln_f_weight, ln_f_bias, lm_head_weight, lm_head_bias, return_all_outputs=True, neuron_config=None):
    """
    Language model head with layer normalization.

    Context encoding network:
    n_active_tokens will be equal to context_length_estimate and return_all_outputs will be False.
    In this case we slice the hidden input and compute the next token logits only for the last context token.

    Normal token gen network:
    n_active_tokens will be 1 and return_all_outputs will be True.
    No slicing required. Will return the next token logits for the current active token.

    Speculative network:
    n_active_tokens will be equal to "k" (k value is passed by user) and return_all_outputs will be True.
    No slicing required. Will return next token logits for "k" active tokens.

    Models: GPT2, OPT, GPT-J, GPTNeoX, BLOOM.

    logits = (layer_norm(H) @ W) + B
    """
    is_bsh = neuron_config and neuron_config.attention_layout == LAYOUT_BSH
    if is_bsh:
        batch_size, n_active_tokens, hidden_size = hidden.sizes
    else:
        hidden_size, n_active_tokens, batch_size = hidden.sizes

    if neuron_config and neuron_config.is_sequence_parallel and not return_all_outputs and not neuron_config.use_1d_query:
        # For sequence parallel norm, we need to gather the hidden but we don't need the full hidden (which creates a large 
        # tensor). Since we only need last_token_id, we can first pick out last_token_id % seq_shard_len from each tp rank,
        # then all-gather only this part, and then pick out the last_token_id // seq_shard_len to pick out the hidden coming
        # from the correct rank.

        # We do not enable for use_1d_query case since the logic is more involved.
        last_token_id_pos_in_shard = hlo.remainder(last_token_id, n_active_tokens)
        hidden = _dynamic_logits_slice(hidden, last_token_id_pos_in_shard, neuron_config)
        hidden = hlo.all_gather(hidden, 1, tp_degree)
        last_token_id_shard_idx = hlo.divide(last_token_id, n_active_tokens)
        hidden = _dynamic_logits_slice(hidden, last_token_id_shard_idx, neuron_config)
        n_active_tokens = 1
    else:
        if neuron_config and neuron_config.is_sequence_parallel:
            hidden = hlo.all_gather(hidden, 1, tp_degree)

        # Check and perform slicing if needed
        if not return_all_outputs:
            hidden = _dynamic_logits_slice(hidden, last_token_id, neuron_config)
            n_active_tokens = 1

    if is_bsh:
        ln_hidden = hlo.layer_norm_bsh(hidden, ln_f_weight, ln_f_bias, neuron_config=None, tp_degree=tp_degree)
        ln_hidden = hlo.transpose210(ln_hidden)
    else:
        ln_hidden = hlo.layer_norm(hidden, ln_f_weight, ln_f_bias, neuron_config=None, tp_degree=tp_degree)
    ln_hidden = hlo.reshape(ln_hidden, shape=(hidden_size, n_active_tokens * batch_size))

    logits = hlo.dot00(lm_head_weight, ln_hidden)
    if lm_head_bias is not None:
        lm_head_bias = hlo.broadcast(lm_head_bias, out_dim_size=logits.sizes, broadcast_dimensions=[0])
        logits = hlo.add(logits, lm_head_bias)
    vocab_size, _ = logits.sizes
    result = hlo.reshape(logits, shape=(vocab_size, n_active_tokens, batch_size))

    if neuron_config and tp_degree != neuron_config.get_local_tp(tp_degree) and not neuron_config.on_device_generation:
        result = hlo.all_gather(result, 0, tp_degree)

    return result


def rms_lm_head(tp_degree, hidden, last_token_id, rms_weight, lm_head_weight, lm_head_bias, return_all_outputs=True, eps=1e-6, neuron_config=None):
    """
    Language model head with rms normalization.

    Context encoding network:
    n_active_tokens will be equal to context_length_estimate and return_all_outputs will be False.
    In this case we slice the hidden input and compute the next token logits only for the last context token.

    Normal token gen network:
    n_active_tokens will be 1 and return_all_outputs will be True.
    No slicing required. Will return the next token logits for the current active token.

    Speculative network:
    n_active_tokens will be equal to "k" (k value is passed by user) and return_all_outputs will be True.
    No slicing required. Will return next token logits for "k" active tokens.

    Models: LLaMa.

    logits = (rms_norm(H) @ W) + B
    """
    is_bsh = neuron_config and neuron_config.attention_layout == LAYOUT_BSH
    if is_bsh:
        batch_size, n_active_tokens, hidden_size = hidden.sizes
    else:
        hidden_size, n_active_tokens, batch_size = hidden.sizes
    if neuron_config and neuron_config.use_1d_query:
        batch_size = last_token_id.sizes[0]
    dtype = hidden.dtype

    if neuron_config and neuron_config.is_sequence_parallel and not return_all_outputs and not neuron_config.use_1d_query:
        # For sequence parallel norm, we need to gather the hidden but we don't need the full hidden (which creates a large 
        # tensor). Since we only need last_token_id, we can first pick out last_token_id % seq_shard_len from each tp rank,
        # then all-gather only this part, and then pick out the last_token_id // seq_shard_len to pick out the hidden coming
        # from the correct rank.

        # We do not enable for use_1d_query case since the logic is more involved.
        last_token_id_pos_in_shard = hlo.remainder(last_token_id, n_active_tokens)
        hidden = _dynamic_logits_slice(hidden, last_token_id_pos_in_shard, neuron_config)
        hidden = hlo.all_gather(hidden, 1, tp_degree)
        last_token_id_shard_idx = hlo.divide(last_token_id, n_active_tokens)
        hidden = _dynamic_logits_slice(hidden, last_token_id_shard_idx, neuron_config)
        n_active_tokens = 1
    else:
        if neuron_config and neuron_config.is_sequence_parallel:
            hidden = hlo.all_gather(hidden, 1, tp_degree)

        # Check and perform slicing if needed
        if not return_all_outputs:
            hidden = _dynamic_logits_slice(hidden, last_token_id, neuron_config)
            n_active_tokens = 1

    rms_hidden = hlo.rms_norm(hidden, rms_weight, eps, neuron_config=None, tp_degree=tp_degree) if is_bsh else hlo.rms_norm(hidden, rms_weight, eps, dim=0, neuron_config=None, tp_degree=tp_degree)

    if is_bsh:
        rms_hidden = hlo.transpose210(rms_hidden)
    rms_hidden = hlo.reshape(rms_hidden, (hidden_size, n_active_tokens*batch_size))
    logits = hlo.dot00(lm_head_weight, rms_hidden)
    if lm_head_bias is not None:
        lm_head_bias = dtype[logits.sizes].Broadcast(lm_head_bias, dimensions=[0])
        logits = dtype[logits.sizes].Add(logits, lm_head_bias)
    vocab_size, _ = logits.sizes
    result = hlo.reshape(logits, (vocab_size, n_active_tokens, batch_size))

    if neuron_config and tp_degree != neuron_config.get_local_tp(tp_degree) and not neuron_config.on_device_generation:
        result = hlo.all_gather(result, 0, tp_degree)

    return result


def _dynamic_logits_slice(hidden, last_token_id, neuron_config=None):
    is_bsh = neuron_config and neuron_config.attention_layout == LAYOUT_BSH
    if is_bsh:
        batch_size, n_active_tokens, hidden_size = hidden.sizes
    else:
        hidden_size, n_active_tokens, batch_size = hidden.sizes
    if neuron_config and neuron_config.lhs_aligned:
        if not is_bsh:
            hidden = hlo.transpose210(hidden)
        hidden = hlo.reshape(hidden, (batch_size*n_active_tokens, hidden_size))

        # [6,3,9] -> [(0,6),(1,3),(2,9)] -> [6+0*128,3+1*128,9+2*128] -> [6,131,265]
        # last_token_id + iota * n_active_tokens
        if neuron_config and neuron_config.use_1d_query:
            # The input is expected to be a list of prompt lengths
            # Here, we transform prompt_lens to last_token_id with following algorithm
            # >   last_token_id = max(cumsum(prompt_lens) - 1, 0)
            last_token_id = hlo.cumsum(last_token_id, dim=0)
            last_token_id = hlo.subtract(last_token_id, 1)
            last_token_id = hlo.maximum(last_token_id, 0)
        else:
            assert last_token_id.sizes[0] == batch_size, \
                f"vectorized last_token_id length ({last_token_id.sizes[0]}) is expected to equal to batch size ({batch_size})"
            offset = hlo.iota(last_token_id.dtype, last_token_id.sizes, [0])
            offset = hlo.multiply(offset, n_active_tokens)
            last_token_id = hlo.add(last_token_id, offset)

        hidden = hlo.index_select(hidden, dim=0, index=last_token_id)
        hidden = hlo.reshape(hidden, (last_token_id.sizes[0], 1, hidden_size))
        if not is_bsh:
            hidden = hlo.transpose210(hidden)
    else:
        hidden = hlo.transpose102(hidden)
        hidden = hlo.index_select(hidden, dim=0, index=last_token_id)
        hidden = hlo.transpose102(hidden)
    return hidden
