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
from transformers_neuronx import hlo, config


def generate(logits, logits_indices, config: config.GenerationConfig, tp_degree=1):

    logits = mask_logits(logits, logits_indices, config.vocab_size)
    if config.do_sample:
        return sample(
            logits,
            top_k=config.top_k,
            top_p=config.top_p,
            temperature=config.temperature,
            top_p_min_tokens=config.top_p_min_tokens,
            tp_degree=tp_degree
        )
    else:
        return greedy_search(logits, tp_degree=tp_degree)


def mask_logits(logits, indices, model_vocab_size):
    vocab_size, n_active_tokens, _ = logits.sizes
    indices_br = hlo.broadcast(indices, (logits.sizes), broadcast_dimensions=(0,))
    mask = hlo.less(indices_br, model_vocab_size)
    logits = hlo.masked_select(mask, logits, float('-inf'))
    return logits


def greedy_search(logits, *, tp_degree=1):
    vocab_size, n_active_tokens, batch_size = logits.sizes
    result = hlo.argmax(logits, 0, tp_degree=tp_degree) # shape: n_active_tokens, batch_size
    return hlo.transpose(result, 0, 1) # shape: batch_size, n_active_tokens


def sample(logits, *, top_k=50, top_p=1.0, top_p_min_tokens=1, temperature=None, tp_degree=1, deterministic=False, dynamic=False):
    vocab_size, n_active_tokens, batch_size = logits.sizes

    # NOTE: Compiler failures can occur when batch != 1
    if top_k == 1 and batch_size == 1:
        return greedy_search(logits, tp_degree=tp_degree)

    if temperature is not None and temperature != 1.0:
        logits = hlo.divide(logits, temperature)

    indices = None

    # Perform Top-K
    if top_k is not None:
        if dynamic:
            logits, index, indices = hlo.topk_masked(logits, k=top_k, dim=0, tp_degree=tp_degree)
        else:
            logits, indices = hlo.topk(logits, k=top_k, dim=0, tp_degree=tp_degree)

    # Perform Top-P
    if top_p is not None and top_p < 1.0:
        logits, indices = hlo.topp(logits, top_p=top_p, top_p_min_tokens=top_p_min_tokens, tp_degree=tp_degree, indices=indices)

    probs = hlo.softmax(logits, dim=0)

    # Final sample after filtering TopP/TopK
    samples = hlo.multinomial(probs, dim=0, deterministic=deterministic)
    tokens = hlo.gather(indices, 0, samples)
    tokens = hlo.squeeze(tokens, 0)
    return hlo.transpose(tokens, 0, 1)
