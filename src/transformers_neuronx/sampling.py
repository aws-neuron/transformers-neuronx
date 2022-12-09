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
import torch


@torch.no_grad()
def simple_sample(model, input_ids, sequence_length, max_sequence_length, eos_token_id=2, top_k=50):
    model.reset()
    start = input_ids.shape[1]

    # populate key/value caches according to the prompt text
    cache_offset = torch.arange(start, dtype=torch.int32)
    next_token_scores = model(input_ids, cache_offset)

    # auto-regressive generation
    tokens = [input_ids]
    for cur_len in range(start, sequence_length):
        next_len = cur_len + 1

        # don't sample EOS
        next_token_scores[:, eos_token_id] = -float('inf')

        # Remove all tokens with a probability less than the last token of the top-k
        topk_values, topk_indices = torch.topk(next_token_scores, top_k)

        # sample
        probs = torch.nn.functional.softmax(topk_values, dim=-1)
        inputs_in_topk = torch.multinomial(probs, num_samples=1, replacement=True)
        inputs = torch.gather(topk_indices, 1, inputs_in_topk)
        tokens.append(inputs)

        # forward pass to get next token
        cache_offset = torch.as_tensor([cur_len], dtype=torch.int32)
        next_token_scores = model(inputs, cache_offset)
    return torch.cat(tokens, dim=-1)
