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

def gelu_new(hidden):
    return hidden.dtype[hidden.sizes].CustomCall(hidden, custom_call_target="AwsNeuronGelu")

def gelu_new_legacy(hidden):
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


def relu(hidden):
    dtype = hidden.dtype
    sizes = hidden.sizes
    zero = dtype.Constant(constant_value=0.0)
    zero_br = dtype[sizes].Broadcast(zero, dimensions=[])
    return dtype[sizes].Maximum(hidden, zero_br)

def solu(hidden):
    dtype = hidden.dtype
    sizes = hidden.sizes
    softmax_hidden = dtype[sizes].CustomCall(hidden, custom_call_target="AwsNeuronSoftmax")
    output = dtype[sizes].Multiply(hidden,softmax_hidden)
    return output

def sigmoid(tensor):
    return tensor.dtype[tensor.sizes].Logistic(tensor)


def silu(tensor):
    logistic = sigmoid(tensor)
    return tensor.dtype[tensor.sizes].Multiply(tensor, logistic)
