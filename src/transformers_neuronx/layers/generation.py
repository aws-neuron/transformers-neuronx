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


def generate(logits, config: config.GenerationConfig, tp_degree=1):

    if config.do_sample:
        return sample(
            logits,
            k=config.top_k,
            temperature=config.temperature,
            tp_degree=tp_degree
        )
    else:
        return greedy_search(logits, tp_degree=tp_degree)


def greedy_search(logits, *, tp_degree=1):
    vocab_size, batch_size, n_active_tokens = logits.sizes
    assert n_active_tokens == 1
    result = hlo.argmax(logits, 0, tp_degree=tp_degree)
    return result.dtype[batch_size, 1].Reshape(result)


def sample(logits, *, k=50, temperature=None, tp_degree=1):
    vocab_size, batch_size, n_active_tokens = logits.sizes
    assert n_active_tokens == 1

    if k == 1:
        return greedy_search(logits, tp_degree=tp_degree)

    logits = hlo.reshape(logits, (batch_size, vocab_size))
    topk_logits, topk_indices = hlo.topk(logits, k=k, dim=1, tp_degree=tp_degree)

    if temperature is not None and temperature != 1.0:
        temperature = hlo.full_like(topk_logits, temperature)
        topk_logits = hlo.divide(topk_logits, temperature)

    probs = hlo.softmax(topk_logits, dim=1)
    samples = hlo.multinomial(probs, dim=1)
    result = hlo.select(topk_indices, dim=1, index=samples)
    return hlo.reshape(result, (batch_size, 1))
