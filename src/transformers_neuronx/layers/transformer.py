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


def ln_lm_head(hidden, last_token_id, ln_f_weight, ln_f_bias, lm_head_weight, lm_head_bias, return_all_outputs=True, neuron_config=None):
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
    ln_hidden = dtype[hidden_size,n_active_tokens*batch_size].Reshape(ln_hidden)
    logits = hlo.dot00(lm_head_weight, ln_hidden)
    if lm_head_bias is not None:
        lm_head_bias = dtype[logits.sizes].Broadcast(lm_head_bias, dimensions=[0])
        logits = dtype[logits.sizes].Add(logits, lm_head_bias)
    vocab_size, _ = logits.sizes
    result = dtype[vocab_size,n_active_tokens,batch_size].Reshape(logits)
    return result


def rms_lm_head(hidden, last_token_id, rms_weight, lm_head_weight, lm_head_bias, return_all_outputs=True, eps=1e-6, neuron_config=None):
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
    return result
