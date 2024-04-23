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
    if config.dynamic or config.do_sample:
        return sample(
            logits,
            top_k=config.top_k,
            top_p=config.top_p,
            temperature=config.temperature,
            top_p_min_tokens=config.top_p_min_tokens,
            tp_degree=tp_degree,
            dynamic=config.dynamic,
            deterministic=config.deterministic
        )
    else:
        return greedy_search(logits, tp_degree=tp_degree)


def mask_logits(logits, indices, model_vocab_size):
    vocab_size, n_active_tokens, _ = logits.sizes
    indices_br = hlo.broadcast(indices, (logits.sizes), broadcast_dimensions=(0,))
    mask = hlo.less(indices_br, model_vocab_size)
    logits = hlo.masked_select(mask, logits, float('-inf'))
    return logits


def greedy_search(logits, *, tp_degree=1, permute=True):
    if permute:
        logits = hlo.permute(logits, (2, 1, 0))
    batch_size, n_active_tokens, vocab_size = logits.sizes
    return hlo.argmax(logits, 2, tp_degree=tp_degree) # shape: batch_size, n_active_tokens


def sample(logits, *, top_k=50, top_p=1.0, top_p_min_tokens=1, temperature=None, tp_degree=1, dynamic=False, deterministic=False):

    logits = hlo.permute(logits, (2, 1, 0))
    batch_size, n_active_tokens, vocab_size = logits.sizes

    # NOTE: Compiler failures can occur when batch != 1
    if top_k == 1 and batch_size == 1:
        return greedy_search(logits, tp_degree=tp_degree, permute=False)

    if temperature is not None and temperature != 1.0:
        if hlo._is_hlo_scalar(temperature):
            temperature = hlo.cast(temperature, logits.dtype)
        logits = hlo.divide(logits, temperature)

    indices = None

    # Perform Top-K
    if top_k is not None:
        if dynamic:
            logits, index, indices = hlo.topk_masked(logits, k=top_k, dim=2, tp_degree=tp_degree)
        else:
            logits, indices = hlo.topk(logits, k=top_k, dim=2, tp_degree=tp_degree)

    # Perform Top-P
    if dynamic or top_p is not None and top_p < 1.0:
        logits, indices = hlo.topp(logits, top_p=top_p, top_p_min_tokens=top_p_min_tokens, tp_degree=tp_degree, indices=indices, dim=2)

    probs = hlo.softmax(logits, dim=2)

    # Final sample after filtering TopP/TopK
    samples = hlo.multinomial(probs, dim=2, deterministic=deterministic)
    tokens = hlo.gather(indices, 2, samples)
    return hlo.squeeze(tokens, 2)
