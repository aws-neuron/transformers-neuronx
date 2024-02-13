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


def generate(logits, logits_indices, config: config.GenerationConfig, tp_degree=1, eos_token_id=None):
    logits = mask_logits(logits, logits_indices, config.vocab_size, eos_token_id, config)
    if config.do_sample:
        return sample(
            logits,
            k=config.top_k,
            temperature=config.temperature,
            tp_degree=tp_degree
        )
    else:
        return greedy_search(logits, tp_degree=tp_degree)


def mask_logits(logits, indices, model_vocab_size, eos_token_id, config):
    vocab_size, n_active_tokens, _ = logits.sizes
    assert n_active_tokens == 1
    indices_br = hlo.broadcast(indices, (logits.sizes), broadcast_dimensions=(0,))
    max_index = hlo.full(value=model_vocab_size, dtype=indices.dtype, sizes=logits.sizes)
    mask = hlo.less(indices_br, max_index)
    min_value = hlo.full(value=float('-inf'), dtype=logits.dtype, sizes=logits.sizes)
    logits = hlo.masked_select(mask, logits, min_value)
    if config.eos_token_id is None and eos_token_id is not None:
        eos_token = hlo.full(value=eos_token_id, dtype=indices_br.dtype, sizes=logits.sizes)
        mask = hlo.equal(indices_br, eos_token)
        logits = hlo.masked_select(mask, min_value, logits)
    return logits


def greedy_search(logits, *, tp_degree=1):
    vocab_size, n_active_tokens, batch_size = logits.sizes
    assert n_active_tokens == 1
    result = hlo.argmax(logits, 0, tp_degree=tp_degree)
    return result.dtype[batch_size, 1].Reshape(result)


def sample(logits, *, k=50, temperature=None, tp_degree=1):
    vocab_size, n_active_tokens, batch_size = logits.sizes
    assert n_active_tokens == 1

    if k == 1 and batch_size == 1:
        return greedy_search(logits, tp_degree=tp_degree)

    logits = hlo.reshape(logits, (vocab_size, batch_size))
    logits = hlo.transpose(logits, src=0, dst=1)

    topk_logits, topk_indices = hlo.topk(logits, k=k, dim=1, tp_degree=tp_degree)

    if temperature is not None and temperature != 1.0:
        temperature = hlo.full_like(topk_logits, temperature)
        topk_logits = hlo.divide(topk_logits, temperature)

    probs = hlo.softmax(topk_logits, dim=1)
    samples = hlo.multinomial(probs, dim=1)
    result = hlo.select(topk_indices, dim=1, index=samples)
    return hlo.reshape(result, (batch_size, 1))
