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

from transformers_neuronx.constants import FUSED_QKV_TP_FACTOR


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


def pad_interleaved(tensor, dim, size, source_len_per_group, pad_len_per_group):
    """
    Pad the selected `dim` of `tensor`, up to `size`. Put zeros interleavedly base on `source_len_per_group`
    and `pad_len_per_group`

        [
            # group i # ? ... x source_len_per_group ... ?, 0, ... x pad_len_per_group ... , 0]
        ]

    For example:
        pad_interleaved([1,2,3]], dim=0, size=9, source_len_per_group=1, pad_len_per_group=2)
        each group becomes [?, 0, 0], with number of group as 3, we have
        result: [1, 0, 0, 2, 0, 0, 3, 0, 0]
    """
    assert isinstance(dim, int), "pad_interleaved now only supports single dim"

    assert isinstance(size, int), "pad_interleaved now only supports single dim of size"

    padded_shape = list(tensor.shape)

    padded_shape[dim] = size

    padded_tensor = torch.zeros(padded_shape, dtype=tensor.dtype)

    num_groups = size // (source_len_per_group + pad_len_per_group)

    src_indices = [slice(0, None) for _ in range(len(padded_shape))]
    target_indices = [slice(0, None, None) for _ in range(len(padded_shape))]
    s_src = 0
    s_target = 0
    for _ in range(num_groups):
        target_indices[dim] = slice(s_target, s_target+source_len_per_group)
        src_indices[dim] = slice(s_src, s_src+source_len_per_group)
        padded_tensor[target_indices] = tensor[src_indices]
        s_src += source_len_per_group
        s_target += (source_len_per_group + pad_len_per_group)

    return padded_tensor


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


def get_pad_size(vocab_size, divisor):
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
    def get_slice_params(tensor, dim, tp_degree):
        size = tensor.shape[dim]
        shard_size = size // tp_degree
        slices = [slice(None) for _ in tensor.shape]
        return size, shard_size, slices

    def get_shard_tensors(tensors, size, shard_size, slices, dim):
        for idx, start in enumerate(range(0, size, shard_size)):
            slices[dim] = slice(start, start+shard_size, 1)
            shard_tensors = [t[tuple(slices)].contiguous() for t in tensors]
            yield idx, shard_tensors

    if q.shape[dim] == k.shape[dim]:
        size, shard_size, slices = get_slice_params(q, dim, tp_degree)
        is_single_dim = len(q.shape) == 1
        if is_single_dim:
            tensor = torch.zeros((size * FUSED_QKV_TP_FACTOR), dtype=q.dtype)
        else:
            hidden_dim, interleave_dim = q.shape
            tensor = torch.zeros((hidden_dim, interleave_dim * FUSED_QKV_TP_FACTOR), dtype=q.dtype)
        for idx, shard_tensors in get_shard_tensors((q, k, v), size, shard_size, slices, dim):
            shard = torch.cat(shard_tensors, dim=dim).contiguous()
            if is_single_dim:
                tensor[(idx)*shard.shape[dim]:(idx+1)*shard.shape[dim]] = shard
            else:
                tensor[:, (idx)*shard.shape[dim]:(idx+1)*shard.shape[dim]] = shard
    else:
        q_hidden_dim, q_interleave_dim = q.shape
        _, kv_interleave_dim = k.shape
        tensor = torch.zeros((q_hidden_dim, q_interleave_dim + kv_interleave_dim * 2), dtype=q.dtype)
        q_size, q_shard_size, q_slices = get_slice_params(q, dim, tp_degree)
        kv_size, kv_shard_size, kv_slices = get_slice_params(k, dim, tp_degree)
        for idx, ((_, q_shard_tensors), (_, kv_shard_tensors)) in enumerate(zip(
            get_shard_tensors((q,), q_size, q_shard_size, q_slices, dim),
            get_shard_tensors((k,v), kv_size, kv_shard_size, kv_slices, dim)
        )):
            q_shard = q_shard_tensors[0]
            kv_shard = torch.concat(kv_shard_tensors, dim=dim).contiguous()
            shard = torch.cat((q_shard, kv_shard), dim=dim).contiguous()
            tensor[:, (idx)*shard.shape[1]:(idx+1)*shard.shape[1]] = shard
    return tensor


def build_replica_groups(num_groups, group_size):
    """
    Construct replica_groups to handle "intra-group" reduce operations.

    Each nested list represents the ids of the cores within the same group.

    Examples:

        group_size = 2
        num_groups = 3
        replica_groups = [[0, 1], [2, 3], [4, 5]]

        group_size = 3
        num_groups = 2
        replica_groups = [[0, 1, 2], [3, 4, 5]]
    """
    replica_groups = [
        [nc for nc in range(group_size * group, group_size * group + group_size)]
        for group in range(num_groups)
    ]
    return replica_groups

