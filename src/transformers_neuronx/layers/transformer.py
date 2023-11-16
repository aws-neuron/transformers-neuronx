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


def inputs(scribe, dtype, batch_size, n_active_tokens, hidden_size, neuron_config=None):
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

    if neuron_config and neuron_config.attention_layout == LAYOUT_BSH:
        hidden_sizes = batch_size, n_active_tokens, hidden_size
    else:
        hidden_sizes = hidden_size, n_active_tokens, batch_size

    hidden = dtype[hidden_sizes].Parameter(parameter_number=0)
    cache_2d = neuron_config and neuron_config.use_2d_cache_ids
    if cache_2d:
        position_sizes = batch_size, n_active_tokens
        cache_ids = s32[position_sizes].Parameter(parameter_number=1)   # 2d cache_ids
    else:
        cache_ids = s32[n_active_tokens].Parameter(parameter_number=1)  # 1d cache_ids

    start_ids = s32[batch_size].Parameter(parameter_number=2)
    last_token_id = s32.Parameter(parameter_number=3)

    sequence_slice_dimensions = (
        1,                     # hidden        | In both HSB/BSH the sequence dim is 1
        1 if cache_2d else 0,  # cache_ids     | Sequence dim varies based on alignment
        None,                  # start_ids     | Offset is per batch, no slicing required
        None                   # last_token_id | Scalar, no slicing required
    )

    return hidden, cache_ids, start_ids, last_token_id, sequence_slice_dimensions


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
    dtype = hidden.dtype

    # Check and perform slicing if needed
    if not return_all_outputs:
        hidden = hlo.dynamic_slice_along(hidden, dim=1, start=last_token_id, size= 1)
        n_active_tokens = 1

    ln_hidden = hlo.layer_norm_bsh(hidden, ln_f_weight, ln_f_bias) if is_bsh else hlo.layer_norm(hidden, ln_f_weight, ln_f_bias)
    if is_bsh:
        ln_hidden = hlo.transpose210(ln_hidden)
    ln_hidden = dtype[hidden_size,n_active_tokens*batch_size].Reshape(ln_hidden)
    logits = hlo.dot00(lm_head_weight, ln_hidden)
    if lm_head_bias is not None:
        lm_head_bias = dtype[logits.sizes].Broadcast(lm_head_bias, dimensions=[0])
        logits = dtype[logits.sizes].Add(logits, lm_head_bias)
    vocab_size, _ = logits.sizes
    result = dtype[vocab_size,n_active_tokens,batch_size].Reshape(logits)

    if neuron_config and tp_degree != neuron_config.get_local_tp(tp_degree):
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
    dtype = hidden.dtype

    # Check and perform slicing if needed
    if not return_all_outputs:
        hidden = hlo.dynamic_slice_along(hidden, dim=1, start=last_token_id, size= 1)
        n_active_tokens = 1

    rms_hidden = hlo.rms_norm(hidden, rms_weight, eps) if is_bsh else hlo.rms_norm(hidden, rms_weight, eps, dim=0)
    rms_hidden = dtype[hidden_size, n_active_tokens*batch_size].Reshape(rms_hidden)
    logits = hlo.dot00(lm_head_weight, rms_hidden)
    if lm_head_bias is not None:
        lm_head_bias = dtype[logits.sizes].Broadcast(lm_head_bias, dimensions=[0])
        logits = dtype[logits.sizes].Add(logits, lm_head_bias)
    vocab_size, _ = logits.sizes
    result = dtype[vocab_size,n_active_tokens,batch_size].Reshape(logits)

    if neuron_config and tp_degree != neuron_config.get_local_tp(tp_degree):
        result = hlo.all_gather(result, 0, tp_degree)

    return result
