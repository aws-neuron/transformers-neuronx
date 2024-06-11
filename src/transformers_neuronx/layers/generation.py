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
    if not config.per_batch_line:
        return sample(
            logits,
            top_k=config.top_k,
            top_p=config.top_p,
            temperature=config.temperature,
            top_p_min_tokens=config.top_p_min_tokens,
            global_top_k=config.global_top_k,
            tp_degree=tp_degree,
            dynamic=config.dynamic,
            deterministic=config.deterministic
        )

    if not config.dynamic and not config.do_sample:
        return greedy_search(logits, tp_degree=tp_degree)

    if config.global_top_k is not None:
        assert config.dynamic is True, "Dynamic on device generation must be enabled when global_top_k is set."

    logits = hlo.permute(logits, (2, 1, 0))
    batch_size, n_active_tokens, vocab_size = logits.sizes

    indices = None
    # Perform global top-k
    if config.global_top_k is not None:
        logits, indices = hlo.topk(logits, k=config.global_top_k, dim=2, tp_degree=tp_degree)

    tokens = []
    for batch_line in range(batch_size):
        logits_slice = hlo.slice_along(logits, 0, start=batch_line, limit=batch_line+1)
        indices_slice = None if indices is None else hlo.slice_along(indices, 0, start=batch_line, limit=batch_line+1)
      
        batch_line_top_k = config.top_k if hlo._is_hlo_scalar(config.top_k) else hlo.get_hlo_scalar_by_index(config.top_k, batch_line)
        batch_line_top_p = config.top_p if hlo._is_hlo_scalar(config.top_p) else hlo.get_hlo_scalar_by_index(config.top_p, batch_line)
        batch_line_temperature = config.temperature if hlo._is_hlo_scalar(config.temperature) else hlo.get_hlo_scalar_by_index(config.temperature, batch_line)
        batch_line_top_p_min_tokens =  config.top_p_min_tokens if hlo._is_hlo_scalar(config.top_p_min_tokens) else  hlo.get_hlo_scalar_by_index(config.top_p_min_tokens, batch_line)

        token = sample(
            logits_slice,
            indices=indices_slice,
            top_k=batch_line_top_k,
            top_p=batch_line_top_p,
            temperature=batch_line_temperature,
            top_p_min_tokens=batch_line_top_p_min_tokens,
            global_top_k=None, # we already performed global_top_k
            permute=False, # we already permuted
            tp_degree=tp_degree,
            dynamic=config.dynamic,
            deterministic=config.deterministic,                        
        )    
        tokens.append(token)
    returned_tokens = hlo.concatenate(tokens, dimension=0)
    return returned_tokens

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


def sample(logits, *, top_k=50, top_p=1.0, top_p_min_tokens=1, temperature=None, global_top_k=None, tp_degree=1, dynamic=False, deterministic=False, indices=None, permute=True):

    if global_top_k is not None:
        assert dynamic is True, "Dynamic on device generation must be enabled when global_top_k is set."

    if permute:
        logits = hlo.permute(logits, (2, 1, 0))

    if global_top_k is not None:
        logits, indices = hlo.topk(logits, k=global_top_k, dim=2, tp_degree=tp_degree)

    batch_size, n_active_tokens, vocab_size = logits.sizes

    # NOTE: Compiler failures can occur when batch != 1
    if top_k == 1 and batch_size == 1 and indices is None:
        return greedy_search(logits, tp_degree=tp_degree, permute=False)

    if temperature is not None and temperature != 1.0:
        if hlo._is_hlo_scalar(temperature):
            temperature = hlo.cast(temperature, logits.dtype)
        logits = hlo.divide(logits, temperature)

    # Perform Top-K
    if top_k is not None:
        if dynamic:
            logits, index, indices = hlo.topk_masked(logits, k=top_k, dim=2, tp_degree=tp_degree, indices=indices)
        else:
            logits, indices = hlo.topk(logits, k=top_k, dim=2, tp_degree=tp_degree)

    # Perform Top-P
    if dynamic or top_p is not None and top_p < 1.0:
        logits, indices = hlo.topp(logits, top_p=top_p, top_p_min_tokens=top_p_min_tokens, tp_degree=tp_degree, indices=indices, dim=2)

    if indices is None:
        if tp_degree > 1:
            logits = hlo.all_gather(logits, dim=2, tp_degree=tp_degree)

    probs = hlo.softmax(logits, dim=2)

    # Final sample after filtering TopP/TopK
    samples = hlo.multinomial(probs, dim=2, deterministic=deterministic)
    if indices is not None:
        tokens = hlo.gather(indices, 2, samples)
    else:
        tokens = samples
    return hlo.squeeze(tokens, 2)
