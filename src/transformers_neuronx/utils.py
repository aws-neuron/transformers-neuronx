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
import math
import itertools
import random

import torch
import torch.nn.functional as F


def get_closest_pow2_bucket_size(size):
    # Lets assume bucket-size = n where 2^k < n < 2^(k+1), should we use 2^k or 2^(k+1)?
    # Elapsed time for these 2 cases:
    #   2^k bucket:       parallel_time + (n - 2^k) * serial_time
    #   2^(k+1) bucket:   2 * parallel_time
    # Approximate: parallel_time ~ 40 x serial_time for 2048
    # switching criteria: n - 2^k = 40
    criteria = 1 - 40 / 2048
    size = 2 ** math.ceil(math.log(criteria * size, 2))
    return size


def maybe_override_attributes(self, kwargs):
    for key, value in kwargs.items():
        if not hasattr(self, key):
            raise KeyError(f'Found invalid key "{key}".')
        if value is not None:
            setattr(self, key, value)


def power_of_two_bucket_sizes(min_bucket_size, max_bucket_size):
    sizes = []
    bucket_size = min_bucket_size
    while bucket_size < max_bucket_size:
        sizes.append(bucket_size)
        bucket_size *= 2
    sizes.append(max_bucket_size)
    return sizes


def pad_sizes(shape, dims, sizes, left=False):
    if isinstance(dims, int):
        dims = (dims,)
    if isinstance(sizes, int):
        sizes = (sizes,) * len(dims)
    lhs = [0] * len(shape)
    rhs = [0] * len(shape)
    side = lhs if left else rhs
    for dim, size in zip(dims, sizes):
        side[dim] = size - shape[dim]
    sizes = tuple(itertools.chain(*zip(reversed(lhs), reversed(rhs))))
    if sum(sizes) == 0:
        return None
    return sizes


def pad(tensor, dims, sizes, left=False):
    if tensor is None:
        return tensor
    padding = pad_sizes(tensor.shape, dims, sizes, left=left)
    if padding is not None:
        if isinstance(tensor, torch.nn.Parameter):
            tensor = tensor.detach()
        return F.pad(tensor, padding)
    return tensor


def round_up_to_divisor(value, divisor):
    return math.ceil(value / divisor) * divisor


def pad_vocab_size(vocab_size, divisor):
    return ((vocab_size // divisor + 1) * divisor - vocab_size) % divisor


def amp_is_u8(amp):
    return '-u8-' in amp


def parse_amp(amp):
    if amp_is_u8(amp):
        return amp.split('-')
    return amp, None, None


def u8_encode(tensor):
    tensor = tensor.to(torch.float32)
    tensor_min = tensor.min().item()
    tensor_max = tensor.max().item()
    tensor = tensor - tensor_min
    tensor *= 255.0 / (tensor_max - tensor_min)
    tensor = tensor.round().to(torch.uint8)
    return tensor, tensor_min, tensor_max

# Sparse attention related, put here for now
def create_blk_mask(blks_q, blks_kv, num_global_blks=0, num_local_blks=1, num_random_blks=0, causal=False):
    """
    Create a block mask given the configs, and the number of blocks in each dimension.
    Assume all heads use the same mask.
    """
    blk_mask = torch.zeros((blks_q, blks_kv), dtype=torch.int32)
    # Add global blocks
    if num_global_blks > 0:
        blk_mask[:num_global_blks, :] = 1
        blk_mask[:, :num_global_blks] = 1
    # Add local blocks
    if num_local_blks > 0:
        width = num_local_blks // 2
        for row in range(blks_q):
            start = max(0, row - width)
            end = min(row + width + 1, blks_kv)
            blk_mask[row, start:end] = 1
    # Add random blocks
    if num_random_blks > 0:
        assert blks_kv > num_random_blks, "Number of random blocks must be smaller than total number of col blocks!"
        for row in range(blks_q):
            # If causal, only draw blocks from the lower-triangular part of the matrix
            pool = list(range(0, blks_kv)) if not causal else list(range(0, row+1))
            selected = pool if len(pool) <= num_random_blks else random.sample(pool, num_random_blks)
            for col in selected:
                blk_mask[row, col] = 1

    if causal:
        blk_mask = torch.tril(blk_mask)

    return blk_mask

def build_dense_mask(q_seq_len, k_seq_len, mask, blk_size=128, causal=False):
    row_blks = (q_seq_len + blk_size - 1) // blk_size
    col_blks = (k_seq_len + blk_size - 1) // blk_size
    assert tuple(mask.shape) == (row_blks, col_blks), f'Mask must have shape (q_seq_len // blk_size, k_seq_len // blk_size)'
    dense_mask = torch.zeros((q_seq_len, k_seq_len), dtype=torch.int32)

    for row_id in range(row_blks):
        for col_id in range(col_blks):
            if int(mask[row_id, col_id]) == 1:
                last_row = min(q_seq_len, (row_id+1)*blk_size)
                last_col = min(k_seq_len, (col_id+1)*blk_size)
                dense_mask[row_id*blk_size:last_row, col_id*blk_size:last_col] = 1
    if causal:
        dense_mask = torch.tril(dense_mask)
    return dense_mask


def batch_tokenize(tokenizer_left_padded, input_texts, pad_token=None):
    """
    Tokenize a list of texts with different lengths.

    Args:
        tokenizer_left_padded (tokenizer): Tokenzier with padding_side='left'. For example: AutoTokenizer.from_pretrained('gpt2', padding_side='left')
        input_texts (list of strings): List of input texts. Texts can have different lengths
        pad_token (int, optional): pad token

    Returns:
        tuple: input_ids, start_ids used as arguments for model.sample
    """
    if pad_token is not None:
        tokenizer_left_padded.pad_token = pad_token
    if not hasattr(tokenizer_left_padded, "pad_token") or tokenizer_left_padded.pad_token is None:
        tokenizer_left_padded.pad_token = tokenizer_left_padded.eos_token
    tok = tokenizer_left_padded(input_texts, return_tensors='pt', padding=True)
    _, start_ids = tok.attention_mask.max(axis=1)
    if (start_ids == 0).all():
        start_ids = None
    input_ids = tok.input_ids
    return input_ids, start_ids