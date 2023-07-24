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


def greedy_search(logits, tp_degree=1):
    vocab_size, n_active_tokens, batch_size = logits.sizes
    assert n_active_tokens == 1
    result = hlo.argmax(logits, 0, tp_degree=tp_degree)
    return result.dtype[batch_size, 1].Reshape(result)


def sample(logits, k=50, tp_degree=1):
    vocab_size, n_active_tokens, batch_size = logits.sizes
    assert n_active_tokens == 1
    if k == 1:
        return greedy_search(logits)
    logits = logits.dtype[vocab_size, batch_size].Reshape(logits)
    logits = hlo.transpose(logits, 1, 0)
    topk_logits, topk_indices = hlo.topk(logits, k=k, dim=1, tp_degree=tp_degree)
    probs = hlo.softmax_new(topk_logits, dim=1)
    samples = hlo.multinomial(probs, dim=1)
    result = hlo.select(topk_indices, dim=1, index=samples)
    return result.dtype[batch_size, 1].Reshape(result)
