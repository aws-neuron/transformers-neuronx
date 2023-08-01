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
import functools
import operator

import torch

from transformers_neuronx import activations
from transformers_neuronx.config import NeuronConfig
from transformers_neuronx import utils
from transformers_neuronx import compiler


def ax_plus_by(a, x, b, y):
    """
    Calculates a * x + b * y
    """
    ax = a.dtype[a.sizes].Multiply(a, x)
    by = b.dtype[b.sizes].Multiply(b, y)
    ax_by = ax.dtype[ax.sizes].Add(ax, by)
    return ax_by

def ax_minus_by(a, x, b, y):
    """
    Calculates a * x - b * y
    """
    ax = a.dtype[a.sizes].Multiply(a, x)
    by = b.dtype[b.sizes].Multiply(b, y)
    ax_by = ax.dtype[ax.sizes].Subtract(ax, by)
    return ax_by


def layer_norm(hidden, weight, bias):
    scribe = hidden.scribe
    dtype = hidden.dtype
    f32 = scribe.f32
    hidden_size, n_active_tokens, batch_size = input_sizes = hidden.sizes
    norm_size = n_active_tokens * batch_size
    sizes = hidden_size, norm_size
    hidden = dtype[sizes].Reshape(hidden)
    hidden = f32[sizes].Convert(hidden)
    one = f32.Constant(constant_value=1)
    scale = f32[norm_size].Broadcast(one, dimensions=[])
    zero = f32.Constant(constant_value=0)
    offset = f32[norm_size].Broadcast(zero, dimensions=[])
    shape = scribe.tuple(f32[sizes], f32[norm_size], f32[norm_size])
    bn_tuple = shape.BatchNormTraining(hidden, scale, offset, epsilon=1e-5, feature_index=1)
    bn_output = f32[sizes].GetTupleElement(bn_tuple, tuple_index=0)
    weight_br = f32[sizes].Broadcast(weight, dimensions=[0])
    output = f32[sizes].Multiply(bn_output, weight_br)
    bias_br = f32[sizes].Broadcast(bias, dimensions=[0])
    output = f32[sizes].Add(output, bias_br)
    output = dtype[sizes].Convert(output)
    output = dtype[input_sizes].Reshape(output)
    return output


def layer_norm_bsh(hidden, weight, bias):
    scribe = hidden.scribe
    dtype = hidden.dtype
    f32 = scribe.f32
    batch_size, n_active_tokens, hidden_size = input_sizes = hidden.sizes
    norm_size = n_active_tokens * batch_size
    sizes = norm_size, hidden_size
    hidden = dtype[sizes].Reshape(hidden)
    hidden = f32[sizes].Convert(hidden)
    one = f32.Constant(constant_value=1)
    scale = f32[norm_size].Broadcast(one, dimensions=[])
    zero = f32.Constant(constant_value=0)
    offset = f32[norm_size].Broadcast(zero, dimensions=[])
    shape = scribe.tuple(f32[sizes], f32[norm_size], f32[norm_size])
    bn_tuple = shape.BatchNormTraining(hidden, scale, offset, epsilon=1e-5, feature_index=0)
    bn_output = f32[sizes].GetTupleElement(bn_tuple, tuple_index=0)
    weight_br = f32[sizes].Broadcast(weight, dimensions=[1])
    output = f32[sizes].Multiply(bn_output, weight_br)
    bias_br = f32[sizes].Broadcast(bias, dimensions=[1])
    output = f32[sizes].Add(output, bias_br)
    output = dtype[sizes].Convert(output)
    output = dtype[input_sizes].Reshape(output)
    return output


def rms_norm(hidden, weight, eps=1e-6):
    # Reference: https://github.com/huggingface/transformers/blob/v4.29.2/src/transformers/models/t5/modeling_t5.py#L238-L260

    batch_size, n_active_tokens, hidden_size = size = hidden.sizes
    dtype = hidden.dtype
    scribe = hidden.scribe
    f32 = scribe.f32

    hidden = cast(hidden, f32)

    # PERF: Is it better to use BatchNormTraining operation here?
    square = f32[hidden.sizes].Multiply(hidden, hidden)
    variance = reduce_mean(square, 2)
    eps = f32.Constant(constant_value=eps)
    eps_br = f32[variance.sizes].Broadcast(eps, dimensions=[])
    mean_eps = f32[variance.sizes].Add(variance, eps_br)
    rsqrt = f32[variance.sizes].Rsqrt(mean_eps)
    rsqrt_br = f32[size].Broadcast(rsqrt, dimensions=[0, 1])
    scaled = f32[size].Multiply(hidden, rsqrt_br)

    if weight is None:
        scaled = cast(scaled, dtype)
        return scaled

    weight = cast(weight, f32)
    weight_br = f32[size].Broadcast(weight, dimensions=[2])
    result = f32[size].Multiply(scaled, weight_br)
    result = cast(result, dtype)

    return result


def dot00(lhs, rhs):
    dtype = lhs.dtype
    _, lhs_size = lhs.sizes
    _, rhs_size = rhs.sizes
    dot_dims = dict(lhs_contracting_dimensions=[0], rhs_contracting_dimensions=[0])
    return dtype[lhs_size, rhs_size].Dot(lhs, rhs, dot_dimension_numbers=dot_dims)


def dot01(lhs, rhs):
    dtype = lhs.dtype
    _, lhs_size = lhs.sizes
    rhs_size, _ = rhs.sizes
    dot_dims = dict(lhs_contracting_dimensions=[0], rhs_contracting_dimensions=[1])
    return dtype[lhs_size, rhs_size].Dot(lhs, rhs, dot_dimension_numbers=dot_dims)


def dot00_add0(lhs, rhs, bias, scales=None, neuron_config=None):
    dtype = bias.dtype if bias is not None else rhs.dtype
    enable_quantize = neuron_config and neuron_config.quant
    if enable_quantize:
        if rhs.dtype != dtype:
            rhs = dtype[rhs.sizes].Convert(rhs)
        if lhs.dtype != dtype:
            lhs = dtype[lhs.sizes].Convert(lhs)
    _, lhs_size = lhs.sizes
    _, rhs_size = rhs.sizes
    dot_dims = dict(lhs_contracting_dimensions=[0], rhs_contracting_dimensions=[0])
    dot = dtype[lhs_size, rhs_size].Dot(lhs, rhs, dot_dimension_numbers=dot_dims)
    if enable_quantize:
        dot = dequantize(dot, scales, neuron_config, 0)
    if bias is None:
        return dot
    bias = dtype[lhs_size, rhs_size].Broadcast(bias, dimensions=[0])
    return dtype[lhs_size, rhs_size].Add(dot, bias)


def dot00_add1(lhs, rhs, bias, scales=None, neuron_config=None):
    dtype = bias.dtype if bias is not None else rhs.dtype
    enable_quantize = neuron_config and neuron_config.quant
    if enable_quantize:
        if rhs.dtype != dtype:
            rhs = dtype[rhs.sizes].Convert(rhs)
        if lhs.dtype != dtype:
            lhs = dtype[lhs.sizes].Convert(lhs)
    _, lhs_size = lhs.sizes
    _, rhs_size = rhs.sizes
    dot_dims = dict(lhs_contracting_dimensions=[0], rhs_contracting_dimensions=[0])
    dot = dtype[lhs_size, rhs_size].Dot(lhs, rhs, dot_dimension_numbers=dot_dims)
    if enable_quantize:
        dot = dequantize(dot, scales, neuron_config, 1)
    if bias is None:
        return dot
    bias = dtype[lhs_size, rhs_size].Broadcast(bias, dimensions=[1])
    return dtype[lhs_size, rhs_size].Add(dot, bias)

def dot10_add1(lhs, rhs, bias, scales=None, neuron_config=None):
    dtype = bias.dtype if bias is not None else rhs.dtype
    enable_quantize = neuron_config and neuron_config.quant
    if enable_quantize:
        if rhs.dtype != dtype:
            rhs = dtype[rhs.sizes].Convert(rhs)
        if lhs.dtype != dtype:
            lhs = dtype[lhs.sizes].Convert(lhs)
    lhs_size, _ = lhs.sizes
    _, rhs_size = rhs.sizes
    dot_dims = dict(lhs_contracting_dimensions=[1], rhs_contracting_dimensions=[0])
    dot = dtype[lhs_size, rhs_size].Dot(lhs, rhs, dot_dimension_numbers=dot_dims)
    if enable_quantize:
        dot = dequantize(dot, scales, neuron_config, 1)
    if bias is None:
        return dot
    bias = dtype[lhs_size, rhs_size].Broadcast(bias, dimensions=[1])
    return dtype[lhs_size, rhs_size].Add(dot, bias)

def gen_add_func(dtype):

    def add_func(scribe):
        p0 = dtype.Parameter(parameter_number=0)
        p1 = dtype.Parameter(parameter_number=1)
        return dtype.Add(p0, p1)

    return add_func


def gen_assign_func(dtype):

    def assign_func(scribe):
        p0 = dtype.Parameter(parameter_number=0)
        p1 = dtype.Parameter(parameter_number=1)
        return p1

    return assign_func


def gen_max_func(dtype):

    def max_func(scribe):
        p0 = dtype.Parameter(parameter_number=0)
        p1 = dtype.Parameter(parameter_number=1)
        return dtype.Maximum(p0, p1)

    return max_func


def mlp(hidden, in_weight, in_bias, out_weight, out_bias, activation_function, tp_degree,
        dequant_dtype=None, u8_bounds=None, in_scales=None, out_scales=None, neuron_config=None):
    # single:
    #   hidden: [h, a, b]
    #   in_weight: [h, 4h]
    #   in_bias: [4h]
    #   out_weight: [4h, h]
    #   out_bias: [h]
    # t-way tp:
    #   hidden: [h, a, b]
    #   in_weight: [h, 4h/t]
    #   in_bias: [4h/t]
    #   out_weight: [4h/t, h]
    #   out_bias: [h]
    dtype = hidden.dtype
    if u8_bounds is not None:
        f32 = hidden.scribe.f32
        *_, in_min, in_max, out_min, out_max = u8_bounds
        in_weight = u8_decode(dtype, dequant_dtype, in_weight, in_min, in_max)
        out_weight = u8_decode(dtype, dequant_dtype, out_weight, out_min, out_max)
    hidden_size, n_active_tokens, batch_size = hidden_sizes = hidden.sizes
    hidden_r_sizes = hidden_size, n_active_tokens * batch_size
    hidden = hidden.dtype[hidden_r_sizes].Reshape(hidden)
    hidden = dot00_add1(hidden, in_weight, in_bias, in_scales, neuron_config)
    hidden = getattr(activations, activation_function)(hidden)
    hidden = dot10_add1(hidden, out_weight, out_bias, out_scales, neuron_config)
    hidden = dtype[hidden_r_sizes].Transpose(hidden, dimensions=[1, 0])
    hidden = dtype[hidden_sizes].Reshape(hidden)
    if tp_degree == 1:
        return hidden
    replica_groups = [list(range(tp_degree))]
    add_func = gen_add_func(dtype)
    hidden = dtype[hidden_sizes].AllReduce(hidden, replica_groups=replica_groups, to_apply=add_func)
    return hidden


def mlp_bsh(hidden, in_weight, in_bias, out_weight, out_bias, activation_function, tp_degree,
            dequant_dtype=None, u8_bounds=None, in_scales=None, out_scales=None, neuron_config=None):
    # single:
    #   hidden: [b, a, h]
    #   in_weight: [h, 4h]
    #   in_bias: [4h]
    #   out_weight: [4h, h]
    #   out_bias: [h]
    # t-way tp:
    #   hidden: [b, a, h]
    #   in_weight: [h, 4h/t]
    #   in_bias: [4h/t]
    #   out_weight: [4h/t, h]
    #   out_bias: [h]
    dtype = hidden.dtype
    if u8_bounds is not None:
        f32 = hidden.scribe.f32
        *_, in_min, in_max, out_min, out_max = u8_bounds
        in_weight = u8_decode(dtype, dequant_dtype, in_weight, in_min, in_max)
        out_weight = u8_decode(dtype, dequant_dtype, out_weight, out_min, out_max)
    batch_size, n_active_tokens, hidden_size = hidden_sizes = hidden.sizes
    hidden_r_sizes = batch_size * n_active_tokens, hidden_size
    hidden = hidden.dtype[hidden_r_sizes].Reshape(hidden)
    hidden = dot10_add1(hidden, in_weight, in_bias, in_scales, neuron_config)
    hidden = getattr(activations, activation_function)(hidden)
    hidden = dot10_add1(hidden, out_weight, out_bias, out_scales, neuron_config)
    hidden = dtype[hidden_sizes].Reshape(hidden)
    if tp_degree == 1:
        return hidden
    replica_groups = [list(range(tp_degree))]
    add_func = gen_add_func(dtype)
    hidden = dtype[hidden_sizes].AllReduce(hidden, replica_groups=replica_groups, to_apply=add_func)
    return hidden


def gated_mlp(
    hidden,
    in0_weight,
    in1_weight,
    out_weight,
    in0_bias=None,
    in1_bias=None,
    out_bias=None,
    activation_function='silu',
    tp_degree=1
):
    """
    An attention MLP using 2 input projections as found in LLama.

    Reference: https://github.com/huggingface/transformers/blob/v4.29.2/src/transformers/models/llama/modeling_llama.py#L144

    TODO: Support quantization

    Sizes:
        hidden:     [b, a, h]
        in0_weight: [h, n / tp]
        in1_weight: [h, n / tp]
        out_weight: [n / tp, h]
        in0_bias:   [n / tp]
        in1_bias:   [n / tp]
        out_bias:   [h]
        result:     [b, a, h]
    """

    dtype = hidden.dtype
    batch_size, n_active_tokens, hidden_size = hidden_sizes = hidden.sizes
    hidden_r_sizes = batch_size * n_active_tokens, hidden_size

    hidden = hidden.dtype[hidden_r_sizes].Reshape(hidden)

    hidden_active = dot10_add1(hidden, in0_weight, in0_bias)
    hidden_active = getattr(activations, activation_function)(hidden_active)
    hidden_linear = dot10_add1(hidden, in1_weight, in1_bias)
    hidden_states = dtype[hidden_linear.sizes].Multiply(hidden_active, hidden_linear)

    result = dot10_add1(hidden_states, out_weight, out_bias)
    result = dtype[hidden_sizes].Reshape(result)

    if tp_degree != 1:
        result = all_reduce_sum(result, tp_degree)

    return result


def u8_decode(dtype, dequant_dtype, weight, min_value, max_value):
    sizes = weight.sizes
    weight = dequant_dtype[sizes].Convert(weight)
    factor = (max_value - min_value) / 255.0
    factor = dequant_dtype.Constant(constant_value=factor)
    factor = dequant_dtype[sizes].Broadcast(factor, dimensions=[])
    min_value = dequant_dtype.Constant(constant_value=min_value)
    min_value = dequant_dtype[sizes].Broadcast(min_value, dimensions=[])
    weight = dequant_dtype[sizes].Multiply(weight, factor)
    weight = dequant_dtype[sizes].Add(weight, min_value)
    return dtype[sizes].Convert(weight)

def softmax_new(logits, dim=None):
    rank = len(logits.sizes)
    if dim is None:
        dim = rank - 1
    shape = logits.sizes
    dtype = logits.dtype
    backend_config = str(dim).encode()
    return dtype[shape].CustomCall(logits, custom_call_target="AwsNeuronSoftmax", backend_config=backend_config,)


def softmax(logits, dim=None):
    rank = len(logits.sizes)
    if dim is None:
        dim = rank - 1
    br_dims = [di for di in range(rank) if di != dim]
    dtype = logits.dtype
    constant_2 = dtype.Constant(constant_value=float('-inf'))
    reduce_sizes = [logits.sizes[di] for di in br_dims]
    max_func = gen_max_func(dtype)
    reduce_7 = dtype[reduce_sizes].Reduce(logits, constant_2, dimensions=[dim], to_apply=max_func)
    broadcast_8 = dtype[logits.sizes].Broadcast(reduce_7, dimensions=br_dims)
    subtract_9 = dtype[logits.sizes].Subtract(logits, broadcast_8)
    exp = dtype[logits.sizes].Exp(subtract_9)
    constant_11 = dtype.Constant(constant_value=0)
    add_func = gen_add_func(dtype)
    reduce_16 = dtype[reduce_sizes].Reduce(exp, constant_11, dimensions=[dim], to_apply=add_func)
    broadcast_17 = dtype[logits.sizes].Broadcast(reduce_16, dimensions=br_dims)
    return dtype[logits.sizes].Divide(exp, broadcast_17)


def transfer_with_static_ring(shape):
    custom_call_target = 'AwsNeuronTransferWithStaticRing'
    return shape.dtype[shape.sizes].CustomCall(shape, custom_call_target=custom_call_target)


def decoder_attention_mask(start_ids, position_ids, n_positions, triu_comparison='LE',
                           allow_kv_dot_prefetch=False, start_mask=True):

    batch_size, = start_ids.sizes
    n_active_tokens, = position_ids.sizes
    triu_sizes = n_active_tokens, n_positions
    int_dtype = position_ids.dtype
    pred = position_ids.scribe.pred
    iota1 = int_dtype[n_positions].Iota(dimensions=[0])
    iota1t = int_dtype[triu_sizes].Broadcast(iota1, dimensions=[1])
    position_ids_br = int_dtype[triu_sizes].Broadcast(position_ids, dimensions=[0])
    mask_triu = pred[triu_sizes].Compare(iota1t, position_ids_br, comparison_direction=triu_comparison)
    if not start_mask:
        return mask_triu, None
    start_sizes = batch_size, n_positions
    iota1s = int_dtype[start_sizes].Broadcast(iota1, dimensions=[1])
    start_ids_br = int_dtype[start_sizes].Broadcast(start_ids, dimensions=[0])
    mask_start = pred[start_sizes].Compare(iota1s, start_ids_br, comparison_direction='GE')
    mask_sizes = batch_size, n_active_tokens, n_positions
    mask_triu = pred[mask_sizes].Broadcast(mask_triu, dimensions=[1, 2])
    mask_start = pred[mask_sizes].Broadcast(mask_start, dimensions=[0, 2])
    mask = pred[mask_sizes].And(mask_triu, mask_start)
    if not allow_kv_dot_prefetch:
        return mask, None
    sizes = batch_size, n_active_tokens
    start_ids_br = int_dtype[sizes].Broadcast(start_ids, dimensions=[0])
    position_ids_br = int_dtype[sizes].Broadcast(position_ids, dimensions=[1])
    active_mask = pred[sizes].Compare(position_ids_br, start_ids_br, comparison_direction='GE')
    return mask, active_mask

class ParameterBuilder:

    def __init__(self, dtype):
        self.dtype = dtype
        self.parameter_number = 0

    def __call__(self, shape, dtype=None):
        if dtype is None:
            dtype = self.dtype
        param = dtype[tuple(shape)].Parameter(parameter_number=self.parameter_number)
        self.parameter_number += 1
        return param


def decoder_attention_mask_legacy(position_ids, dtype, n_positions):
    n_active_tokens, = position_ids.sizes
    sizes = n_active_tokens, n_positions
    int_dtype = position_ids.dtype
    pred = position_ids.scribe.pred
    iota0 = int_dtype[sizes].Iota(dimensions=[0])
    iota1 = int_dtype[sizes].Iota(dimensions=[1])
    triu = pred[sizes].Compare(iota0, iota1, comparison_direction='GE')
    triu = dtype[sizes].Convert(triu)
    position_ids = int_dtype[sizes].Broadcast(position_ids, dimensions=[0])
    mask = pred[sizes].Compare(iota1, position_ids, comparison_direction='LE')
    mask = dtype[sizes].Convert(mask)
    return dtype[sizes].Multiply(mask, triu)


def legalize_cache_ids(cache_ids):
    """
    Updates cache ids with valid indices and returns the final non-padding
    position.

    This function allows the `cache_id` tensor to be 0 padded without
    updating incorrect cache lines. This does so by inserting linear ids
    into the pad values. This also computes the last non-padding cache
    id position so that the index can be used during hidden state selection.

    This function assumes that the non-padded portion of the `cache_ids` input
    begins at tensor position 0 and is linearly increasing.

    Examples:

    | Scenario                 | input          | cache_ids        | index |
    |--------------------------|----------------|------------------|-------|
    | Padded Prompt Encoding   | [14, 15, 0, 0] | [14, 15, 16, 17] | 1     |
    | Unpadded Prompt Encoding | [5, 6, 7]      | [5, 6, 7]        | 2     |
    | Token Generation         | [29]           | [29]             | 0     |

    Arguments:
        cache_ids: The cache ids to update the cache for (may be padded)

    Returns:
        cache_ids: The original cache_ids with padding removed
        index: The index of the maximum valid cache id
    """
    dtype = cache_ids.dtype
    sizes = cache_ids.sizes

    # During token generation, do not compute the end index
    if sizes[0] == 1:
        return cache_ids, dtype.Constant(constant_value=0)

    value = reduce_max(cache_ids, 0)
    index = argmax(cache_ids, 0)
    index = cast(index, dtype)

    positions = dtype[sizes].Iota(dimensions=[0])
    value_br = dtype[sizes].Broadcast(value, dimensions=[])
    index_br = dtype[sizes].Broadcast(index, dimensions=[])

    offset = dtype[sizes].Subtract(positions, index_br)
    cache_ids = dtype[sizes].Add(value_br, offset)

    return cache_ids, index


def dtype_minimum(dtype):
    scribe = dtype.scribe
    minimums = {
         scribe.s64: -2 ** 63,
         scribe.s32: -2 ** 31,
         scribe.s16: -2 ** 15,
         scribe.s8: -2 ** 7,
         scribe.u64: 0,
         scribe.u32: 0,
         scribe.u16: 0,
         scribe.u8: 0,
         scribe.pred: False,
    }
    return minimums.get(dtype, float('-inf'))


def reduce_max(tensor, dim, keepdim=False):

    dtype = tensor.dtype
    reduce_shape = list(tensor.sizes)
    reduce_shape.pop(dim)

    def reducer(scribe):
        p0 = dtype.Parameter(parameter_number=0)
        p1 = dtype.Parameter(parameter_number=1)
        return dtype.Maximum(p0, p1)

    minimum = dtype.Constant(constant_value=dtype_minimum(dtype))
    value = dtype[reduce_shape].Reduce(tensor, minimum, dimensions=[dim], to_apply=reducer)

    if keepdim:
        keepdim_shape = list(tensor.sizes)
        keepdim_shape[dim] = 1
        value = dtype[keepdim_shape].Reshape(value)

    return value


def reduce_sum(tensor, dim, keepdim=False):

    dtype = tensor.dtype
    reduce_shape = list(tensor.sizes)
    reduce_shape.pop(dim)

    def reducer(scribe):
        p0 = dtype.Parameter(parameter_number=0)
        p1 = dtype.Parameter(parameter_number=1)
        return dtype.Add(p0, p1)

    minimum = dtype.Constant(constant_value=0)
    value = dtype[reduce_shape].Reduce(tensor, minimum, dimensions=[dim], to_apply=reducer)

    if keepdim:
        keepdim_shape = list(tensor.sizes)
        keepdim_shape[dim] = 1
        value = dtype[keepdim_shape].Reshape(value)

    return value


def all_gather(tensor, dim, tp_degree):
    shape = list(tensor.sizes)
    shape[dim] *= tp_degree
    dtype = tensor.dtype
    return dtype[shape].AllGather(
        tensor,
        dimensions=[dim],
        replica_groups=[list(range(tp_degree))],
    )


def all_reduce_sum(tensor, tp_degree):
    size = tensor.sizes
    dtype = tensor.dtype

    def reducer(scribe):
        p0 = dtype.Parameter(parameter_number=0)
        p1 = dtype.Parameter(parameter_number=1)
        return dtype.Add(p0, p1)

    return dtype[size].AllReduce(
        tensor,
        replica_groups=[list(range(tp_degree))],
        to_apply=reducer
    )


def squeeze(tensor, dim):
    assert tensor.sizes[dim] == 1
    dtype = tensor.dtype
    size = list(tensor.sizes)
    size.pop(dim)
    return dtype[size].Reshape(tensor)


def unsqueeze(tensor, dim):
    size = list(tensor.sizes)
    dim %= len(size) + 1  # Handle negative sizes
    size.insert(dim, 1)
    dtype = tensor.dtype
    return dtype[size].Reshape(tensor)


def gather(tensor, dim, index):
    """
    Gather elements from a `tensor` along `dim` at the given `index`

    Provides similar functionality to `torch.gather`. The `tensor` and `index`
    tensors must have the same rank.
    """
    assert dim <= len(tensor.sizes)

    # Must have the same rank
    tensor_sizes = list(tensor.sizes)
    index_sizes = list(index.sizes)
    assert len(tensor_sizes) == len(index_sizes)

    # Must have same dimensions in non-`dim` dimension
    tensor_sizes.pop(dim)
    index_sizes.pop(dim)
    assert tensor_sizes == index_sizes

    dims = len(tensor.sizes)
    final_size = index.sizes

    # Usqueeze the index to concatenate with linear indices
    index = unsqueeze(index, -1)

    index_size = index.sizes
    dtype = tensor.dtype

    # Build linear indexers for non-`dim` dimensions
    indices = list()
    for i in range(dims):
        if i == dim:
            indices.append(index)
        else:
            indices.append(index.dtype[index_size].Iota(dimensions=[i]))

    # Concatenate indices into a single dense indexing tensor
    concat_size = list(index_size)
    concat_size[-1] = dims
    index = index.dtype[concat_size].Concatenate(*indices, dimensions=[dims])

    # Gather using dense index
    result = dtype[final_size].Gather(
        tensor,
        index,
        gather_dimension_numbers=dict(
            collapsed_slice_dims=list(range(dims)),
            start_index_map=list(range(dims)),
            index_vector_dim=dims,
        ),
        gather_slice_sizes=[1] * dims,
    )

    return result


def _argmax(tensor, dim, keepdim=False, return_values=False):
    """
    Performs argmax on a single partition
    """
    backend_config = str(dim).encode()

    scribe = tensor.scribe
    u32 = scribe.u32
    reduce_shape = list(tensor.sizes)
    reduce_shape.pop(dim)

    index = u32[reduce_shape].CustomCall(
        tensor, custom_call_target='AwsNeuronArgMax', backend_config=backend_config,
    )

    if keepdim:
        keepdim_shape = list(tensor.sizes)
        keepdim_shape[dim] = 1
        index = u32[keepdim_shape].Reshape(index)

    if return_values:
        return reduce_max(tensor, dim, keepdim), index
    return index


def argmax(tensor, dim, keepdim=False, return_values=False, tp_degree=1):

    if tp_degree == 1:
        return _argmax(tensor, dim, keepdim, return_values=return_values)

    scribe = tensor.scribe

    # Initially reduce on each replica for replica-local result
    index = _argmax(tensor, dim, keepdim=True)
    value = reduce_max(tensor, dim, keepdim=True)

    # Synchronize replica-local results across all replicas (Much smaller after argmax)
    index = all_gather(index, dim, tp_degree)
    value = all_gather(value, dim, tp_degree)

    dtype = index.dtype
    sizes = index.sizes

    # Fix concatenated replica-local indices. Offset by (replica_id * replica_size)
    replica_size = dtype.Constant(constant_value=tensor.sizes[dim])
    replica_size = dtype[sizes].Broadcast(replica_size)
    replica_ids = dtype[sizes].Iota(dimensions=[dim])
    offset = dtype[sizes].Multiply(replica_ids, replica_size)
    index = dtype[sizes].Add(index, offset)

    # Find replica with globally maximum value
    replica_index = _argmax(value, dim, keepdim=True)

    # Final masked reduction
    dimensions = list(range(len(replica_index.sizes) + 1))
    dimensions.pop(dim)

    rs_size = list(replica_index.sizes)
    rs_size[dim] *= tp_degree
    br_size = list(replica_index.sizes)
    br_size.insert(dim, tp_degree)
    replica_index = dtype[br_size].Broadcast(replica_index, dimensions=dimensions)
    replica_index = dtype[rs_size].Reshape(replica_index)

    mask = scribe.pred[sizes].Compare(replica_index, replica_ids, comparison_direction='EQ')
    mask_index = index.dtype[mask.sizes].Convert(mask)
    masked = dtype[sizes].Multiply(mask, mask_index)
    index = reduce_sum(masked, dim=dim, keepdim=keepdim)

    if return_values:
        value = reduce_max(value, dim=dim, keepdim=keepdim)
        return value, index
    return index

def _embedding(weight, index):
    """
    Performs embedding on a single partition
    """
    assert len(weight.sizes) == 2, (
        f'Expected rank 2 embedding weights but found shape: {weight.sizes}'
    )

    n_embedding, embedding_dim = weight.sizes
    dtype = weight.dtype

    # Linearize index tensor to gather from 0th dimension
    n_index = functools.reduce(operator.mul, index.sizes, 1)
    linear_index = index.dtype[n_index].Reshape(index)

    # Gather
    result = dtype[n_index, embedding_dim].Gather(
        weight,
        linear_index,
        gather_dimension_numbers=dict(
            offset_dims=[1],
            collapsed_slice_dims=[0],
            start_index_map=[0],
            index_vector_dim=1,
        ),
        gather_slice_sizes=[1, embedding_dim],
    )

    # Reshape embedding tensor to look like the original index shape
    return dtype[(*index.sizes, embedding_dim)].Reshape(result)


def embedding(weight, index, tp_degree=1, dim=1):
    """
    An embedding operation analogous to torch.nn.Embedding

    When `tp_degree` == 1, this assumes that each program has its own
    embedding data that will be used exclusively within that partition. In a
    program that uses multiple nodes, this can be useful if the embedding
    data is replicated across all nodes.

    When `tp_degree` > 1, this function assumes that the index is identical
    across replicas and the embedding data is partitioned across them. This
    allows each partition to gather from their embedding weight matrices
    independently and the results can be combined with a collective compute
    operation. The combination strategy is based on how the embedding was
    partitioned:
    - When `dim` == 0, this function assumes that the embedding has been
      partitioned with distinct vocabulary tokens on each device. This uses
      AllReduce to combine results with a masked summation.
    - When `dim` == 1, this function assumes that each partition has the all
      vocabulary tokens but only a portion of the embedding. This uses
      AllGather to combine results with concatenation.
    """
    partition_size, embed_size = weight.sizes

    # Use (index % partition_size) with partitioned vocabulary
    offset = index
    if tp_degree > 1 and dim == 0:
        const = index.dtype.Constant(constant_value=partition_size)
        const_br = index.dtype[index.sizes].Broadcast(const, dimensions=[])
        offset = index.dtype[index.sizes].Remainder(index, const_br)

    # Replica-local embedding
    result = _embedding(weight, offset)

    # Case 1: Early exit if not combining results from multiple replicas
    if tp_degree == 1:
        return result

    # Case 2: Partitioned vocabulary - Sum masked embeddings
    if dim == 0:

        raise NotImplementedError(
            f'Embedding `dim` may not be 0. ReplicaId instruction unsupported'
        )

        pred = index.scribe.pred

        # Compute embedding mask
        replica_id = index.dtype.ReplicaId() # XXX: Unsupported
        vocab_size = index.dtype.Constant(constant_value=partition_size)
        one = index.dtype.Constant(constant_value=1)

        minimum = index.dtype.Multiply(replica_id, vocab_size)
        next_replica_id = index.dtype.Add(replica_id, one)
        maximum = index.dtype.Multiply(next_replica_id, vocab_size)

        minimum_br = index.dtype[index.sizes].Broadcast(minimum, dimensions=[])
        maximum_br = index.dtype[index.sizes].Broadcast(maximum, dimensions=[])

        mask_min = pred[index.sizes].Compare(index, minimum_br, comparison_direction='GE')
        mask_max = pred[index.sizes].Compare(index, maximum_br, comparison_direction='LT')

        mask = pred[index.sizes].And(mask_min, mask_max)
        dims = range(len(result.sizes))[:-1] # All but the embedding dimension
        mask_br = pred[result.sizes].Broadcast(mask, dimensions=dims)

        # Zero out embeddings which are not contained in this partition
        zero = result.dtype.Constant(constant_value=0)
        zero_br = result.dtype[result.sizes].Broadcast(zero, dimensions=[])
        masked_result = result.dtype[result.sizes].Select(mask_br, result, zero_br)

        # Combine embeddings from all partitions
        return all_reduce_sum(masked_result, tp_degree=tp_degree)

    # Case 3: Partitioned embedding: Concatenate embedding pieces
    if dim == 1:
        # Using BSH, concatenate along the last dim
        return all_gather(result, 2, tp_degree=tp_degree)

    raise NotImplementedError(
        f'Embedding operation does not support dim={dim}'
    )


def cache_broadcast(n_positions, from_batch_size, to_batch_size, n_heads_tp, d_head, amp, n_layer):
    if to_batch_size % from_batch_size:
        raise ValueError(f'to_batch_size={to_batch_size} is not multiples of from_batch_size={from_batch_size}')

    def cache_broadcast_impl(scribe):
        dtype = getattr(scribe, amp)
        sizes = n_positions, from_batch_size, n_heads_tp, d_head
        sources = [dtype[sizes].Parameter(parameter_number=pn) for pn in range(n_layer * 2)]
        num_repeat = to_batch_size // from_batch_size
        outputs = []
        for source in sources:
            operands = [source for _ in range(num_repeat)]
            sizes = n_positions, to_batch_size, n_heads_tp, d_head
            outputs.append(dtype[sizes].Concatenate(*operands, dimensions=[1]))
        root_shapes = [shape.dtype[shape.sizes] for shape in outputs]
        return scribe.tuple(*root_shapes).Tuple(*outputs)

    return cache_broadcast_impl


def quantize(tensor, neuron_config: NeuronConfig, scales_dim):
    scribe = tensor.scribe
    quant_dtype = getattr(scribe, neuron_config.quant.quant_dtype)
    dtype = tensor.dtype
    abs_tensor = dtype[tensor.sizes].Abs(tensor)
    max_vals = reduce_max(abs_tensor, dim=scales_dim)
    constant = dtype.Constant(constant_value=127.0)
    broadcast0 = dtype[max_vals.sizes].Broadcast(constant, dimensions=[])
    scales = dtype[max_vals.sizes].Divide(max_vals, broadcast0)
    bdim = list(range(0, len(tensor.sizes)))
    bdim.remove(scales_dim)
    broadcast1 = dtype[tensor.sizes].Broadcast(scales, dimensions=bdim)
    quantized_tensor = dtype[tensor.sizes].Divide(tensor, broadcast1)
    clamp_upper_bound = dtype[tensor.sizes].Broadcast(dtype.Constant(constant_value=127.0), dimensions=[])
    clamp_lower_bound = dtype[tensor.sizes].Broadcast(dtype.Constant(constant_value=-128.0), dimensions=[])
    quantized_tensor = dtype[tensor.sizes].Clamp(clamp_lower_bound, quantized_tensor, clamp_upper_bound)
    quantized_tensor = quant_dtype[tensor.sizes].Convert(quantized_tensor)
    return quantized_tensor, scales


def dequantize(tensor, scales, neuron_config: NeuronConfig, scales_dim):
    scribe = tensor.scribe
    f32 = scribe.f32
    dtype = getattr(scribe, neuron_config.quant.dequant_dtype)
    tensor = f32[tensor.sizes].Convert(tensor)
    scales = f32[tensor.sizes].Broadcast(scales, dimensions=[scales_dim])
    tensor = f32[tensor.sizes].Multiply(tensor, scales)
    tensor = dtype[tensor.sizes].Convert(tensor)
    return tensor


def reduce_mean(tensor, dims, keepdim=False):

    dtype = tensor.dtype

    if dims is None:
        dims = list(range(len(tensor.sizes)))

    if isinstance(dims, int):
        dims = [dims]

    elements = 1
    reduce_shape = list(tensor.sizes)
    for dim in sorted(dims, reverse=True):
        elements *= reduce_shape[dim]
        reduce_shape.pop(dim)

    def reducer(scribe):
        p0 = dtype.Parameter(parameter_number=0)
        p1 = dtype.Parameter(parameter_number=1)
        return dtype.Add(p0, p1)

    minimum = dtype.Constant(constant_value=0)
    value = dtype[reduce_shape].Reduce(tensor, minimum, dimensions=dims, to_apply=reducer)
    divisor = dtype.Constant(constant_value=1.0 / elements)
    divisor_br = dtype[reduce_shape].Broadcast(divisor)
    value = dtype[reduce_shape].Multiply(value, divisor_br)

    if keepdim:
        keepdim_shape = list(tensor.sizes)
        for dim in dims:
            keepdim_shape[dim] = 1
        value = dtype[keepdim_shape].Reshape(value)

    return value


def cumsum(tensor, dim):

    dtype = tensor.dtype

    init = dtype.Constant(constant_value=0)

    def reducer(scribe):
        p0 = dtype.Parameter(parameter_number=0)
        p1 = dtype.Parameter(parameter_number=1)
        return dtype.Add(p0, p1)

    sizes = [1] * len(tensor.sizes)
    pads = [0] * len(tensor.sizes)
    sizes[dim] = tensor.sizes[dim]
    pads[dim] = tensor.sizes[dim] - 1

    return dtype[tensor.sizes].ReduceWindow(
        tensor,
        init,
        to_apply=reducer,
        window=dict(
            dimensions=[
                dict(
                    size=size,
                    stride=1,
                    padding_low=pad,
                    window_dilation=1,
                    base_dilation=1,
                )
                for (size, pad) in zip(sizes, pads)
            ],
        ),
    )


def cast(value, dtype):
    if value.dtype != dtype:
        return dtype[value.sizes].Convert(value)
    return value


def slice_along(tensor, dim, limit, start=0):
    """
    Slice along a dimension.
    """
    dimensions = [
        dict(start=0, limit=size, stride=1) for size in tensor.sizes
    ]
    dimensions[dim] = dict(start=start, limit=limit, stride=1)

    sizes = list(tensor.sizes)
    sizes[dim] = limit - start

    return tensor.dtype[sizes].Slice(
        tensor,
        slice_dimensions=dimensions
    )


def pad(tensor, dim, size, value=0):
    rank = len(tensor.sizes)
    dtype = tensor.dtype

    dimensions = [dict(edge_padding_low=0, edge_padding_high=0, interior_padding=0)] * rank
    dimensions[dim] = dict(edge_padding_low=0, edge_padding_high=size, interior_padding=0)

    sizes = list(tensor.sizes)
    sizes[dim] += size

    padding = dtype.Constant(constant_value=value)
    return dtype[sizes].Pad(tensor, padding, padding_config=dict(dimensions=dimensions))


def transpose(tensor, src, dst):
    size = list(tensor.sizes)
    size[src] = tensor.sizes[dst]
    size[dst] = tensor.sizes[src]
    dimensions = list(range(len(size)))
    dimensions[src] = dst
    dimensions[dst] = src
    return tensor.dtype[size].Transpose(tensor, dimensions=dimensions)


def _topk(tensor, k):
    """
    Performs top-k on a single partition on the last dimension
    """
    scribe = tensor.scribe
    u32 = scribe.u32
    dtype = tensor.dtype

    sizes = list(tensor.sizes)
    sizes[-1] = k

    assert k <= tensor.sizes[-1], f'Cannot perform topk when k ({k}) is larger than the tensor size ({tensor.sizes[-1]})'

    # Compiler requirement: Ensure input top-k dim is a factor of 8
    size = tensor.sizes[-1]
    padded_size = utils.round_up_to_divisor(size, 8)
    if padded_size != size:
        padding = padded_size - size
        tensor = pad(tensor, -1, padding, value=dtype_minimum(tensor.dtype))

    results = scribe.tuple(dtype[sizes], u32.dtype[sizes]).CustomCall(
        tensor,
        custom_call_target='AwsNeuronTopK',
        backend_config=str(k).encode(),
    )

    value = dtype[sizes].GetTupleElement(results, tuple_index=0)
    index = u32[sizes].GetTupleElement(results, tuple_index=1)

    return value, index


def full_like(tensor, value):
    dtype = tensor.dtype
    size = tensor.sizes
    result = dtype.Constant(constant_value=value)
    result = dtype[size].Broadcast(result, dimensions=[])
    return result


def all_reduce_max(tensor, tp_degree=1):
    size = tensor.sizes
    dtype = tensor.dtype

    def reducer(scribe):
        p0 = dtype.Parameter(parameter_number=0)
        p1 = dtype.Parameter(parameter_number=1)
        return dtype.Maximum(p0, p1)

    return dtype[size].AllReduce(
        tensor,
        replica_groups=[list(range(tp_degree))],
        to_apply=reducer
    )


def all_reduce_max_with_indices(tensor, index, tp_degree=1):
    """
    Select the maximum value and its associated index across ranks.

    NOTE: This does NOT automatically correct the index tensor to be globally
          valid. This must be done in the calling code since we do not know
          which dimension the index refers to here.
    """
    size = tensor.sizes
    scribe = tensor.scribe
    pred = scribe.pred

    assert tensor.sizes == index.sizes

    # Find maximum value across ranks
    maximum = all_reduce_max(tensor, tp_degree)

    # Zero out all rank-local indices which do not correspond global maximum
    mask = pred[size].Compare(tensor, maximum, comparison_direction='EQ')
    zero = full_like(tensor, 0)
    index = index.dtype[index.sizes].Select(mask, index, zero)

    # Reduce masked indices from across ranks (Note: Max instead of sum due to potential duplicate values)
    index = all_reduce_max(index, tp_degree)

    return maximum, index


def topk(tensor, dim, k=50, tp_degree=1):
    """
    Get the top-k values and indices along a dimension.

    Implementation Notes:
    ---------------------
    - The `k` value may not be larger than tensor.shape[dim]. The dimension size
      may be smaller than anticipated with large tensor parallel degrees.
    - Top-k instruction may only be invoked on the final dimension.
    - The input `tensor` size along dimension `dim` must be a multiple of 8
    - The input `tensor` may not be 1d. The compiler will fail.
    """

    if k == 1:
        return argmax(tensor, dim, return_values=True, keepdim=True, tp_degree=tp_degree)

    scribe = tensor.scribe
    f32 = scribe.f32

    rank = len(tensor.sizes)
    if dim < 0:
        dim %= rank

    original = None
    if dim != rank - 1:
        original = dim
        dim = rank - 1

    # Helper function to reformat outputs
    def output(value, index):
        if original is not None:
            value = transpose(value, dim, original)
            index = transpose(index, dim, original)
        return value, index

    # Compiler may only perform top-k on the last dimension
    if original is not None:
        tensor = transpose(tensor, dim, original)

    # Heuristic: When tensor sizes are small, gather first
    if k > tensor.sizes[dim] and tp_degree > 1:
        tensor = all_gather(tensor, dim=1, tp_degree=tp_degree)
        tp_degree = 1

    # Initial reduction
    value, index = _topk(tensor, k)

    # Early exit if not doing a tensor parallel computation
    if tp_degree == 1:
        return output(value, index)

    rank_size = f32.Constant(constant_value=tensor.sizes[dim])

    # Combine initial reduction from all ranks
    value = all_gather(value, dim=dim, tp_degree=tp_degree)
    index = all_gather(index, dim=dim, tp_degree=tp_degree)

    # Correct index so that it is global. Note: Use f32 computation to avoid compiler issue
    dtype = index.dtype
    sizes = index.sizes
    rank_size_br = f32[sizes].Broadcast(rank_size, dimensions=[])
    k_size = f32.Constant(constant_value=k)
    k_size_br = f32[sizes].Broadcast(k_size, dimensions=[])
    iota = f32[sizes].Iota(dimensions=[dim])
    group_id = f32[sizes].Divide(iota, k_size_br)
    offset = f32[sizes].Multiply(group_id, rank_size_br)
    offset = dtype[sizes].Convert(offset)
    index = dtype[sizes].Add(index, offset)

    # Find the final global index
    value, replica_index = _topk(value, k)

    # Get the real indices
    index = gather(index, dim, replica_index)
    return output(value, index)


def multinomial(probabilities, dim):
    """
    Single sample multinomial selection along a dimension
    """
    scribe = probabilities.scribe
    dtype = probabilities.dtype
    pred = scribe.pred
    u32 = scribe.u32

    cumprob = cumsum(probabilities, dim)

    minimum = dtype.Constant(constant_value=0)
    maximum = dtype.Constant(constant_value=1)
    sizes = list(probabilities.sizes)
    sizes.pop(dim)
    uniform = dtype[sizes].Rng(minimum, maximum, distribution=1) # Uniform distribution

    dims = list(range(len(probabilities.sizes)))
    dims.pop(dim)
    uniform = dtype[probabilities.sizes].Broadcast(uniform, dimensions=dims)

    cmp = pred[probabilities.sizes].Compare(uniform, cumprob, comparison_direction='GT')
    result = cast(cmp, u32)
    summation = reduce_sum(result, dim, keepdim=True)
    return summation


def full(value, dtype, sizes):
    result = dtype.Constant(constant_value=value)
    result = dtype[sizes].Broadcast(result, dimensions=[])
    return result


def literal(dtype, tensor):

    accessors = {
        # https://github.com/tensorflow/tensorflow/blob/v2.8.0/tensorflow/compiler/xla/xla_data.proto#L401
        torch.bool: "preds",
        torch.int8: "s8s",
        torch.uint8: "u8s",
        torch.int32: "s32s",
        torch.int64: "s64s",
        torch.float32: "f32s",
        torch.float64: "f64s",

        torch.complex64: "c64s", # Stored as interleaved real, imag floats.
        torch.complex128: "c128s", # Stored as interleaved real, imag doubles.

        # The F16s, BF16s, U16s and S16s are encoded in little endian byte order
        torch.float16: "f16s",     # Stored as bytes
        torch.bfloat16: "bf16s",   # Stored as bytes
        torch.int16: "s16s",       # Stored as bytes
    }

    converter = compiler.DataTypeConverter()

    # Convert tensor data to expected HLO data type
    torch_dtype = converter.hlo2torch(dtype.shape_proto.element_type)
    if tensor.dtype != torch_dtype:
        tensor = tensor.to(torch_dtype)

    data = tensor.data.numpy().ravel()
    if tensor.dtype in [torch.float16, torch.bfloat16, torch.int16]:
        data = data.tobytes()

    accessor = accessors[tensor.dtype]
    element_type = converter.torch2hlo(tensor.dtype)
    sizes = list(tensor.shape)
    result = dtype[sizes].Constant(
        literal={
            accessor: data,
            'shape': dict(
                dimensions=sizes,
                element_type=element_type,
                is_dynamic_dimension=[False] * len(sizes),
                layout=dict(
                    minor_to_major=reversed(range(len(sizes))),
                    memory_space=1,
                ),
            ),
        },
    )
    return result


def select(tensor, dim, index, keepdim=False):
    """
    Selects a value for a single index along a dimension.
    """
    assert index.sizes[dim] == 1
    assert len(tensor.sizes) == len(index.sizes)

    scribe = tensor.scribe
    pred = scribe.pred
    size = tensor.sizes
    dtype = tensor.dtype

    iota = index.dtype[size].Iota(dimensions=[dim])
    index_br = index.dtype[size].Broadcast(index, dimensions=list(range(len(index.sizes))))
    mask = pred[size].Compare(iota, index_br, comparison_direction='EQ')
    mask = cast(mask, dtype)

    masked = dtype[size].Multiply(mask, tensor)
    result = reduce_sum(masked, dim)
    if keepdim:
        result = unsqueeze(result, dim)
    return result


def index_select(tensor, dim, index):
    dtype = tensor.dtype
    n_index, = index.sizes

    sizes = list(tensor.sizes)
    sizes[dim] = n_index
    offset_dims = list(range(len(tensor.sizes)))
    offset_dims.pop(dim)
    gather_slice_sizes = list(tensor.sizes)
    gather_slice_sizes[dim] = 1

    result = dtype[sizes].Gather(
        tensor,
        index,
        gather_dimension_numbers=dict(
            offset_dims=offset_dims,
            collapsed_slice_dims=[dim],
            start_index_map=[dim],
            index_vector_dim=1,
        ),
        gather_slice_sizes=gather_slice_sizes,
    )
    return result


def add(a, b):
    assert a.sizes == b.sizes
    assert a.dtype == b.dtype
    return a.dtype[a.sizes].Add(a, b)


def divide(a, b):
    assert a.sizes == b.sizes
    assert a.dtype == b.dtype
    return a.dtype[a.sizes].Divide(a, b)


def reshape(tensor, shape):
    if shape == tensor.sizes:
        return tensor
    dst_numel = functools.reduce(operator.mul, shape)
    src_numel = functools.reduce(operator.mul, tensor.sizes)
    assert dst_numel == src_numel
    return tensor.dtype[shape].Reshape(tensor)
