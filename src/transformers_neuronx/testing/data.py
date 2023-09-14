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
# =============================================================================
import torch


def batch_varying_lengths(vocab_size, batch_size, context_length):
    """
    Creates a batch with varying length context attention masks

    This is useful for testing that mixed/unordered context lengths produce the
    correct results.

    Example:
        attention_mask = [
            [0, 0, 1],
            [0, 1, 1],
            [0, 0, 1],
            [1, 1, 1],
        ]
    """
    inputs = torch.randint(0, vocab_size, [batch_size, context_length])
    ones = torch.ones(batch_size, context_length)
    starts = torch.randint(0, context_length, [batch_size])
    mask = (starts.view(-1, 1) < ones).int()  # Ensure at least 1 item is set in mask
    return inputs, mask, starts


def batch_all_lengths(vocab_size, context_length):
    """
    Creates a batch with 1 of each context attention mask sizes

    This is useful for exhaustively checking if each context length is properly
    respected in the model.

    Example:
        attention_mask = [
            [1, 1, 1],
            [0, 1, 1],
            [0, 0, 1],
        ]
    """
    inputs = torch.randint(0, vocab_size, [context_length, context_length])
    mask = torch.ones(context_length, context_length).triu().int()
    starts = torch.arange(context_length)
    inputs = inputs * mask
    return inputs, mask, starts


def batch_full_lengths(vocab_size, batch_size, context_length):
    """
    Creates a batch with identical full-length context attention masks

    This is only useful as a sanity check to see that batching works.

    Example:
        attention_mask = [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
        ]
    """
    inputs = torch.randint(0, vocab_size, [batch_size, context_length])
    mask = torch.ones(batch_size, context_length)
    starts = torch.zeros(batch_size)
    return inputs, mask, starts
