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
from transformers_neuronx import activations


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


def dot00(lhs, rhs):
    dtype = lhs.dtype
    _, lhs_size = lhs.sizes
    _, rhs_size = rhs.sizes
    dot_dims = dict(lhs_contracting_dimensions=[0], rhs_contracting_dimensions=[0])
    return dtype[lhs_size, rhs_size].Dot(lhs, rhs, dot_dimension_numbers=dot_dims)


def dot00_add0(lhs, rhs, bias):
    dtype = lhs.dtype
    _, lhs_size = lhs.sizes
    _, rhs_size = rhs.sizes
    dot_dims = dict(lhs_contracting_dimensions=[0], rhs_contracting_dimensions=[0])
    dot = dtype[lhs_size, rhs_size].Dot(lhs, rhs, dot_dimension_numbers=dot_dims)
    bias = dtype[lhs_size, rhs_size].Broadcast(bias, dimensions=[0])
    return dtype[lhs_size, rhs_size].Add(dot, bias)


def dot00_add1(lhs, rhs, bias):
    dtype = lhs.dtype
    _, lhs_size = lhs.sizes
    _, rhs_size = rhs.sizes
    dot_dims = dict(lhs_contracting_dimensions=[0], rhs_contracting_dimensions=[0])
    dot = dtype[lhs_size, rhs_size].Dot(lhs, rhs, dot_dimension_numbers=dot_dims)
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
        dequant_dtype=None, u8_bounds=None):
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
    hidden = dot00_add0(in_weight, hidden, in_bias)
    hidden = getattr(activations, activation_function)(hidden)
    hidden = dot00_add0(out_weight, hidden, out_bias)
    hidden = dtype[hidden_sizes].Reshape(hidden)
    if tp_degree == 1:
        return hidden
    replica_groups = [list(range(tp_degree))]
    add_func = gen_add_func(dtype)
    hidden = dtype[hidden_sizes].AllReduce(hidden, replica_groups=replica_groups, to_apply=add_func)
    return hidden


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


def decoder_attention_mask(position_ids, dtype, n_positions):
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
