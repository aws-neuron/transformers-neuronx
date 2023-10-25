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


def batch_tokenize(tokenizer_left_padded, input_texts, pad_token=None):
    """
    Tokenize a list of texts with different lengths.

    Args:
        tokenizer_left_padded (tokenizer): Tokenzier with padding_side='left'. For example: AutoTokenizer.from_pretrained('gpt2', padding_side='left')
        input_texts (list of strings): List of input texts. Texts can have different lengths
        pad_token (int, optional): pad token

    Returns:
        tuple: input_ids, start_ids used as arguments for model.sample function
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


def interleave_qkv(q, k, v, tp_degree, dim=1):
    """
    Create a merged QKV Tensor with weights arranged for sharding.
    
    Args:
        q(torch.Tensor): Weights/Bias for Q
        k(torch.Tensor): Weights/Bias for K
        v(torch.Tensor): Weights/Bias for V
        tp_degree(int): tp_degree used for sharding
        dim(int): Dimension to concatenate tensors
    Returns:
        tensor: Concatenated QKV tensor with interleaved weights
    """
    if q.shape[dim] == k.shape[dim]:
        size = q.shape[dim]
        shard_size = q.shape[dim] // tp_degree
        slices = [slice(None) for _ in q.shape]
        is_single_dim = len(q.shape) == 1
        if is_single_dim:
            tensor = torch.zeros((q.shape[0] * 3), dtype=q.dtype)
        else:
            tensor = torch.zeros((q.shape[0], q.shape[1] * 3), dtype=q.dtype)
        for idx, start in enumerate(range(0, size, shard_size)):    
            slices[dim] = slice(start, start+shard_size, 1)
            q_shard = q[tuple(slices)].contiguous()
            k_shard = k[tuple(slices)].contiguous()
            v_shard = v[tuple(slices)].contiguous()
            shard = torch.cat((q_shard, k_shard, v_shard), dim=dim).contiguous()
            if is_single_dim:
                tensor[(idx)*shard.shape[dim]:(idx+1)*shard.shape[dim]] = shard
            else:
                tensor[:, (idx)*shard.shape[dim]:(idx+1)*shard.shape[dim]] = shard
    else:
        tensor = torch.zeros((q.shape[0], q.shape[1] + k.shape[1] * 2), dtype=q.dtype)
        q_shards = []
        kv_shards = []
        q_size = q.shape[dim]
        q_shard_size = q.shape[1] // tp_degree
        q_slices = [slice(None) for _ in q.shape]
        for start in range(0, q_size, q_shard_size):
            q_slices[dim] = slice(start, start + q_shard_size, 1)
            q_shard = q[tuple(q_slices)].contiguous()    
            q_shards.append(q_shard)

        kv_size = k.shape[dim]
        kv_shard_size = k.shape[1] // tp_degree
        kv_slices = [slice(None) for _ in q.shape]
        for start in range(0, kv_size, kv_shard_size):    
            kv_slices[dim] = slice(start, start + kv_shard_size, 1)
            k_shard = k[tuple(kv_slices)].contiguous()    
            v_shard = v[tuple(kv_slices)].contiguous()
            kv_shard = torch.concat((k_shard, v_shard), dim=dim).contiguous()
            kv_shards.append(kv_shard)

        for idx, (q_shard, kv_shard) in enumerate(zip(q_shards, kv_shards)):
            shard = torch.cat((q_shard, kv_shard), dim=dim).contiguous()
            tensor[:, (idx)*shard.shape[1]:(idx+1)*shard.shape[1]] = shard
    return tensor
