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


def new_gelu(hidden):
    dtype = hidden.dtype
    sizes = hidden.sizes
    input_input = dtype[sizes].Multiply(hidden, hidden)
    input_pow_3 = dtype[sizes].Multiply(input_input, hidden)
    scale = dtype.Constant(constant_value=0.044715)
    scale_br = dtype[sizes].Broadcast(scale, dimensions=[])
    mul = dtype[sizes].Multiply(input_pow_3, scale_br)
    add = dtype[sizes].Add(mul, hidden)
    sqrt_2_over_pi = dtype.Constant(constant_value=math.sqrt(2.0 / math.pi))
    sqrt_2_over_pi_br = dtype[sizes].Broadcast(sqrt_2_over_pi, dimensions=[])
    mul2 = dtype[sizes].Multiply(add, sqrt_2_over_pi_br)
    tanh = dtype[sizes].Tanh(mul2)
    one = dtype.Constant(constant_value=1.0)
    one_br = dtype[sizes].Broadcast(one, dimensions=[])
    add1 = dtype[sizes].Add(tanh, one_br)
    mul3 = dtype[sizes].Multiply(add1, hidden)
    half = dtype.Constant(constant_value=0.5)
    half_br = dtype[sizes].Broadcast(half, dimensions=[])
    output = dtype[sizes].Multiply(mul3, half_br)
    return output


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


def mlp(hidden, in_weight, in_bias, out_weight, out_bias, tp_degree):
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
    hidden_size, n_active_tokens, batch_size = hidden_sizes = hidden.sizes
    hidden_r_sizes = hidden_size, n_active_tokens * batch_size
    hidden = hidden.dtype[hidden_r_sizes].Reshape(hidden)
    hidden = dot00_add0(in_weight, hidden, in_bias)
    hidden = new_gelu(hidden)
    hidden = dot00_add0(out_weight, hidden, out_bias)
    hidden = dtype[hidden_sizes].Reshape(hidden)
    replica_groups = [list(range(tp_degree))]
    add_func = gen_add_func(dtype)
    hidden = dtype[hidden_sizes].AllReduce(hidden, replica_groups=replica_groups, to_apply=add_func)
    return hidden


def softmax(logits):
    dtype = logits.dtype
    constant_2 = dtype.Constant(constant_value=float('-inf'))
    *reduce_sizes, _ = logits.sizes
    max_func = gen_max_func(dtype)
    reduce_7 = dtype[reduce_sizes].Reduce(logits, constant_2, dimensions=[3], to_apply=max_func)
    broadcast_8 = dtype[logits.sizes].Broadcast(reduce_7, dimensions=[0,1,2])
    subtract_9 = dtype[logits.sizes].Subtract(logits, broadcast_8)
    exp = dtype[logits.sizes].Exp(subtract_9)
    constant_11 = dtype.Constant(constant_value=0)
    add_func = gen_add_func(dtype)
    reduce_16 = dtype[reduce_sizes].Reduce(exp, constant_11, dimensions=[3], to_apply=add_func)
    broadcast_17 = dtype[logits.sizes].Broadcast(reduce_16, dimensions=[0,1,2])
    return dtype[logits.sizes].Divide(exp, broadcast_17)


def transfer_with_static_ring(shape):
    custom_call_target = 'AwsNeuronTransferWithStaticRing'
    return shape.dtype[shape.sizes].CustomCall(shape, custom_call_target=custom_call_target)


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
