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


def ln_lm_head(hidden, ln_f_weight, ln_f_bias, lm_head_weight, lm_head_bias):
    """
    Language model head with layer normalization.

    This slices the hidden input to compute output for a single output token
    rather than `n_active_tokens`. During context encoding this means that
    the next token logits will be computed *only* for the last context token.

    Models: GPT2, OPT, GPT-J, GPTNeoX, BLOOM.

    logits = (layer_norm(H) @ W) + B
    """
    hidden_size, n_active_tokens, batch_size = hidden.sizes
    dtype = hidden.dtype
    if n_active_tokens > 1:
        slice_dimensions = [
            dict(start=0, limit=hidden_size, stride=1),
            dict(start=n_active_tokens - 1, limit=n_active_tokens, stride=1),
            dict(start=0, limit=batch_size, stride=1),
        ]
        n_active_tokens = 1
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


def rms_lm_head(hidden, rms_weight, lm_head_weight, lm_head_bias, eps=1e-6):
    """
    Language model head with rms normalization.

    This slices the hidden input to compute output for a single output token
    rather than `n_active_tokens`. During context encoding this means that
    the next token logits will be computed *only* for the last context token.

    Models: LLaMa.

    logits = (rms_norm(H) @ W) + B
    """
    hidden_size, n_active_tokens, batch_size = hidden.sizes
    dtype = hidden.dtype
    if n_active_tokens > 1:
        slice_dimensions = [
            dict(start=0, limit=hidden_size, stride=1),
            dict(start=n_active_tokens - 1, limit=n_active_tokens, stride=1),
            dict(start=0, limit=batch_size, stride=1),
        ]
        n_active_tokens = 1
        sizes = hidden_size, n_active_tokens, batch_size
        hidden = dtype[sizes].Slice(hidden, slice_dimensions=slice_dimensions)
    rms_hidden = hlo.rms_norm(hidden, rms_weight, eps)
    rms_hidden = dtype[hidden_size, n_active_tokens * batch_size].Reshape(rms_hidden)
    logits = hlo.dot00(lm_head_weight, rms_hidden)
    if lm_head_bias is not None:
        lm_head_bias = dtype[logits.sizes].Broadcast(lm_head_bias, dimensions=[0])
        logits = dtype[logits.sizes].Add(logits, lm_head_bias)
    vocab_size, _ = logits.sizes
    result = dtype[vocab_size,n_active_tokens,batch_size].Reshape(logits)
    return result
