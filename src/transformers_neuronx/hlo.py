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
import os
import functools
import operator
from typing import Optional, List, Callable, Union

import torch
import numpy as np

from transformers_neuronx import activations
from transformers_neuronx.config import NeuronConfig
from transformers_neuronx.constants import LAYOUT_BSH
from transformers_neuronx import utils
from transformers_neuronx import compiler
from transformers_neuronx import constants
from transformers_neuronx import dtypes
from transformers_neuronx.nki.compile import nki_call

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


def batch_norm(tensor, feature_index, epsilon=1e-5):
    """
    Normalizes an array across batch and spatial dimensions.

    For each feature in the feature dimension (`feature_index` is the index
    for the feature dimension in tensor), the operation calculates the mean
    and variance across all the other dimensions and uses the mean and variance
    to normalize each element in tensor.

    Reference: https://www.tensorflow.org/xla/operation_semantics#batchnormtraining

    Arguments:
        tensor: N dimensional tensor to be normalized.
        feature_index: Index to feature dimension in operand.
        epsilon: Value used in normalization calculation to avoid dividing by 0.

    Returns:
        bn_tuple: Tuple of (tensor, batch_mean, batch_var), where
            tensor is the tensor normalized by the mean and variance
            batch_mean (norm_size) mean that's used to normalize tensor.
            var (norm_size) var that's used to normalize tensor.
    """
    scribe = tensor.scribe
    dtype = tensor.dtype
    sizes = tensor.sizes
    num_features = sizes[feature_index]
    scale = full(1, dtype, num_features)
    offset = full(0, dtype, num_features)
    shape = scribe.tuple(dtype[sizes], dtype[num_features], dtype[num_features])
    bn_tuple = shape.BatchNormTraining(tensor, scale, offset, epsilon=epsilon, feature_index=feature_index)
    return bn_tuple


def layer_norm(hidden, weight, bias, neuron_config=None, tp_degree=None):
    scribe = hidden.scribe
    dtype = hidden.dtype
    f32 = scribe.f32
    hidden_size, n_active_tokens, batch_size = input_sizes = hidden.sizes
    norm_size = n_active_tokens * batch_size
    sizes = hidden_size, norm_size
    hidden = reshape(hidden, sizes)
    hidden = cast(hidden, f32)
    bn_tuple = batch_norm(hidden, feature_index=1)
    bn_output = get_tuple_element(bn_tuple, tuple_index=0)
    weight_br = broadcast(weight, sizes, [0])
    output = multiply(bn_output, weight_br)
    bias_br = broadcast(bias, sizes, [0])
    output = add(output, bias_br)
    output = cast(output, dtype)
    output = reshape(output, input_sizes)

    if neuron_config and neuron_config.is_sequence_parallel:
        return all_gather(output, 1, tp_degree, replica_groups=None)

    return output


def layer_norm_bsh(hidden, weight, bias, neuron_config=None, tp_degree=None):
    scribe = hidden.scribe
    dtype = hidden.dtype
    f32 = scribe.f32
    batch_size, n_active_tokens, hidden_size = input_sizes = hidden.sizes
    norm_size = n_active_tokens * batch_size
    sizes = norm_size, hidden_size
    hidden = reshape(hidden, sizes)
    hidden = cast(hidden, f32)
    bn_tuple = batch_norm(hidden, feature_index=0)
    bn_output = get_tuple_element(bn_tuple, tuple_index=0)
    weight_br = broadcast(weight, sizes, [1])
    output = multiply(bn_output, weight_br)
    bias_br = broadcast(bias, sizes, [1])
    output = add(output, bias_br)
    output = cast(output, dtype)
    output = reshape(output, input_sizes)

    if neuron_config and neuron_config.is_sequence_parallel:
        return all_gather(output, 0, tp_degree, replica_groups=None)

    return output


def group_norm(hidden, weight, bias, num_groups=1, neuron_config=None, tp_degree=None):
    """
    Perform GroupNorm on input with shape (H, S, B).
    """
    dtype = hidden.dtype
    scribe = hidden.scribe
    f32 = scribe.f32

    hidden_size, n_active_tokens, batch_size = input_sizes = hidden.sizes

    if hidden_size % num_groups!= 0:
        raise ValueError(f'Hidden dim {hidden_size} must be divisible by num_groups {num_groups}')

    # Permute HSB->BSH
    hidden = permute(hidden, [2, 1, 0])
    # Reshape hidden to (B, S, g, H // g)
    group_size = hidden_size // num_groups
    bsh_unpack_sizes = batch_size, n_active_tokens, num_groups, group_size
    hidden = reshape(hidden, bsh_unpack_sizes)
    # Reshape to (B*S*g, H//g)
    norm_size = batch_size * n_active_tokens * num_groups
    sizes = norm_size, group_size
    hidden = reshape(hidden, sizes)
    hidden = cast(hidden, f32)

    # Batchnorm
    bn_tuple = batch_norm(hidden, feature_index=0)
    bn_output = get_tuple_element(bn_tuple, tuple_index=0)

    # Reshape back to (B, S, g, H // g)
    bn_output = reshape(bn_output, bsh_unpack_sizes)
    # Reshape to BSH
    bsh_shape = batch_size, n_active_tokens, hidden_size
    bn_output = reshape(bn_output, bsh_shape)
    # Permute back to HSB
    bn_output = permute(bn_output, [2, 1, 0])

    # Apply weight and bias
    weight_br = broadcast(weight, input_sizes, broadcast_dimensions=[0])
    output = multiply(bn_output, weight_br)
    bias_br = broadcast(bias, input_sizes, broadcast_dimensions=[0])
    output = add(output, bias_br)
    output = cast(output, dtype)

    if neuron_config and neuron_config.is_sequence_parallel:
        return all_gather(output, 1, tp_degree, replica_groups=None)

    return output


def group_norm_shb(hidden, weight, bias, num_groups):
    """
    Perform GroupNorm on input with shape (S, H, B).
    """
    raise NotImplementedError("SHB GroupNorm is not currently implemented, use HSB")


def group_norm_bsh(hidden, weight, bias, num_groups):
    """
    Perform GroupNorm on input with shape (B, S, H).
    """
    raise NotImplementedError("BSH GroupNorm is not currently implemented, use HSB")


def rms_norm_legacy(hidden, weight, eps=1e-6, dim=2, neuron_config=None, tp_degree=None):
    # Reference: https://github.com/huggingface/transformers/blob/v4.29.2/src/transformers/models/t5/modeling_t5.py#L238-L260

    size = hidden.sizes
    batch_dims = list(range(len(size)))
    batch_dims.pop(dim)
    batch_shapes = list(size)
    batch_shapes.pop(dim)

    dtype = hidden.dtype
    scribe = hidden.scribe
    f32 = scribe.f32

    hidden = cast(hidden, f32)

    square = multiply(hidden, hidden)
    variance = reduce_mean(square, dim)
    eps = full(eps, f32, batch_shapes)
    mean_eps = add(variance, eps)
    mean_rsqrt = rsqrt(mean_eps)
    rsqrt_br = broadcast(mean_rsqrt, size, batch_dims)
    scaled = multiply(hidden, rsqrt_br)

    if weight is None:
        scaled = cast(scaled, dtype)
        return scaled

    weight = cast(weight, f32)
    weight_br = broadcast(weight, size, [dim])
    result = multiply(scaled, weight_br)
    result = cast(result, dtype)

    if neuron_config and neuron_config.is_sequence_parallel:
        result = all_gather(result, 1, tp_degree, replica_groups=None)

    return result


def rms_norm(hidden, weight, eps=1e-6, dim=2, neuron_config=None, tp_degree=None):

    dtype = hidden.dtype
    shape = hidden.sizes
    # Fallback on generic HLO implementation when norm dimension is 1
    if shape[dim] == 1:
        return rms_norm_legacy(hidden, weight, eps, dim, neuron_config, tp_degree)
    scribe = hidden.scribe
    backend_config = str(dim).encode()
    eps = hidden.scribe.f32.Constant(constant_value=eps)
    f32 = scribe.f32
    hidden = cast(hidden, f32)

    result = dtype[shape].CustomCall(hidden, weight, eps, custom_call_target="AwsNeuronRmsNorm", backend_config=backend_config,)
    if neuron_config and neuron_config.is_sequence_parallel:
        result = all_gather(result, 1, tp_degree, replica_groups=None)

    return result


def dot_general(lhs, rhs, dimension_numbers):
    # Reference: https://www.tensorflow.org/xla/operation_semantics#dotgeneral

    dtype = lhs.dtype
    lhs_sizes = lhs.sizes
    rhs_sizes = rhs.sizes
    dot_dims = dict(lhs_contracting_dimensions=dimension_numbers.get("lhs_contracting_dimensions", [0]),
                    lhs_batch_dimensions=dimension_numbers.get("lhs_batch_dimensions", []),
                    rhs_contracting_dimensions=dimension_numbers.get("rhs_contracting_dimensions", [0]),
                    rhs_batch_dimensions=dimension_numbers.get("rhs_batch_dimensions", []))
    lhs_free_dimensions = list(filter(lambda x: x not in dot_dims["lhs_batch_dimensions"] and \
                                      x not in dot_dims["lhs_contracting_dimensions"],
                                      list(range(len(lhs_sizes)))))
    rhs_free_dimensions = list(filter(lambda x: x not in dot_dims["rhs_batch_dimensions"] and \
                                      x not in dot_dims["rhs_contracting_dimensions"],
                                      list(range(len(rhs_sizes)))))

    # Calculate batch/contracting/free sizes
    lhs_batch_sizes = [lhs_sizes[idx] for idx in dot_dims["lhs_batch_dimensions"]]
    rhs_batch_sizes = [rhs_sizes[idx] for idx in dot_dims["rhs_batch_dimensions"]]
    assert lhs_batch_sizes == rhs_batch_sizes, f"unmatched batch_sizes ({lhs_batch_sizes}) vs ({rhs_batch_sizes})"
    lhs_contracting_sizes = [lhs_sizes[idx] for idx in dot_dims["lhs_contracting_dimensions"]]
    rhs_contracting_sizes = [rhs_sizes[idx] for idx in dot_dims["rhs_contracting_dimensions"]]
    assert lhs_contracting_sizes == rhs_contracting_sizes, \
        f"unmatched contracting_sizes ({lhs_contracting_sizes}) vs ({rhs_contracting_sizes})"
    lhs_free_sizes = [lhs_sizes[idx] for idx in lhs_free_dimensions]
    rhs_free_sizes = [rhs_sizes[idx] for idx in rhs_free_dimensions]

    dot_sizes = lhs_batch_sizes + lhs_free_sizes + rhs_free_sizes
    output_dot = dtype[dot_sizes].Dot(lhs, rhs, dot_dimension_numbers=dot_dims)
    return output_dot


def blockwise_qk_matmul(query, keys, neuron_config):
    output_dot = hlo.full(0, )
    return output_dot


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


def canonicalize_lhs_rhs_dtype(lhs, rhs, neuron_config):
    enable_quantize = neuron_config is not None \
                        and neuron_config.quant is not None
    scribe = lhs.scribe
    if enable_quantize:
        if lhs.dtype == getattr(scribe, neuron_config.quant.quant_dtype):
            lhs = rhs.dtype[lhs.sizes].Convert(lhs)
        if rhs.dtype == getattr(scribe, neuron_config.quant.quant_dtype):
            rhs = lhs.dtype[rhs.sizes].Convert(rhs)
    dtype = lhs.dtype
    return lhs, rhs, dtype


def dot_add(
    lhs: 'HloShape',
    rhs: 'HloShape',
    bias: 'HloShape' = None,
    lhs_contracting_dimension: List[int] = 0,
    rhs_contracting_dimension: List[int] = 0,
    bias_dimension: int = 0,
    scales: Optional['HloShape'] = None,
    neuron_config: Optional[NeuronConfig] = None,
):
    """
    Perform tensor product and optional addition of bias.

    A tensor product is performed on the lhs and rhs tensors,
    contracting over the dimensions specified in
    lhs_contracting_dimension and rhs_contracting_dimension.

    If provided, the bias tensor is broadcast to the shape of
    the dot product and added to the result.

    The result is optionally dequantized before returning.

    Arguments:
        lhs: Left-hand side tensor (HloShape).
        rhs: Right-hand side tensor (HloShape).
        bias: Bias tensor to be added to the result.
        lhs_contracting_dimension: Contracting dimension(s) for the left-hand side tensor.
        rhs_contracting_dimension: Contracting dimension(s) for the right-hand side tensor.
        bias_dimension: Dimension to broadcast the bias tensor.
        scales: Scales to "rescale" the quantized tensor after it's multiplied,
            where W_f = W_q * scales.
        neuron_config: NeuronConfig object that specifies the quantization dtype
            and method.

    Returns:
        tensor: Result of tensor product and optional addition.
    """

    lhs, rhs, _ = canonicalize_lhs_rhs_dtype(lhs, rhs, neuron_config)
    enable_quantize = neuron_config is not None and neuron_config.quant is not None

    if not isinstance(lhs_contracting_dimension, list):
        lhs_contracting_dimension = [lhs_contracting_dimension]
    if not isinstance(rhs_contracting_dimension, list):
        rhs_contracting_dimension = [rhs_contracting_dimension]
    dimension_numbers = dict(
        lhs_contracting_dimensions=lhs_contracting_dimension,
        rhs_contracting_dimensions=rhs_contracting_dimension,
    )
    dot = dot_general(lhs, rhs, dimension_numbers)
    if enable_quantize:
        dot = dequantize(dot, scales, neuron_config, bias_dimension)
    if bias is None:
        return dot
    bias = broadcast(bias, dot.sizes, broadcast_dimensions=[bias_dimension])
    return add(dot, bias)


def dot00_add0(lhs, rhs, bias, scales=None, neuron_config=None):
    return dot_add(lhs, rhs, bias, 0, 0, 0, scales, neuron_config)


def dot00_add1(lhs, rhs, bias, scales=None, neuron_config=None):
    return dot_add(lhs, rhs, bias, 0, 0, 1, scales, neuron_config)


def dot10_add1(lhs, rhs, bias, scales=None, neuron_config=None):
    return dot_add(lhs, rhs, bias, 1, 0, 1, scales, neuron_config)


def dot11_add1(lhs, rhs, bias, scales=None, neuron_config=None):
    return dot_add(lhs, rhs, bias, 1, 1, 1, scales, neuron_config)


def dot_with_tiled_weight_add(lhs, rhs, bias,
                              lhs_contracting_dimensions,
                              rhs_contracting_dimensions,
                              bias_dimension=0,
                              scales=None, neuron_config=None):
    lhs, rhs, dtype = canonicalize_lhs_rhs_dtype(lhs, rhs, neuron_config)
    enable_quantize = neuron_config and neuron_config.quant
    dot_result_lhs_dims = list(filter(lambda x: x not in lhs_contracting_dimensions,
                                       list(range(len(lhs.sizes)))))
    dot_result_lhs_sizes = [lhs.sizes[i] for i in dot_result_lhs_dims]
    dot_result_rhs_dims = list(filter(lambda x: x not in rhs_contracting_dimensions,
                                       list(range(len(rhs.sizes)))))
    dot_result_rhs_sizes = [rhs.sizes[i] for i in dot_result_rhs_dims]
    dot_dims = dict(lhs_contracting_dimensions=lhs_contracting_dimensions,
                    rhs_contracting_dimensions=rhs_contracting_dimensions)
    dot_result_sizes = dot_result_lhs_sizes + dot_result_rhs_sizes
    dot = dtype[dot_result_sizes].Dot(lhs, rhs, dot_dimension_numbers=dot_dims)
    lhs_size = np.product(dot_result_lhs_sizes)
    rhs_size = np.product(dot_result_rhs_sizes)
    dot = reshape(dot, (lhs_size, rhs_size))

    if enable_quantize:
        dot = dequantize(dot, scales, neuron_config, bias_dimension)
    if bias is None:
        return dot
    bias = broadcast(bias, (lhs_size, rhs_size), broadcast_dimensions=[bias_dimension])
    result = add(dot, bias)
    return result


def dot_1220_add1(lhs, rhs, bias, scales=None, neuron_config=None):
    return dot_with_tiled_weight_add(lhs, rhs, bias,
                                     lhs_contracting_dimensions=[1, 2],
                                     rhs_contracting_dimensions=[2, 0],
                                     bias_dimension=1,
                                     scales=scales, neuron_config=neuron_config)

def dot_0120_add1(lhs, rhs, bias, scales=None, neuron_config=None):
    return dot_with_tiled_weight_add(lhs, rhs, bias,
                                     lhs_contracting_dimensions=[0, 1],
                                     rhs_contracting_dimensions=[2, 0],
                                     bias_dimension=1,
                                     scales=scales, neuron_config=neuron_config)


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


def get_activation(activation_function: Union[str, Callable]) -> Callable:
    """
    Returns an activation function if it's a callable. Otherwise returns
    the mapping to the named function in activations.py.

    Arguments:
        activation_function: Callable or name of a function in activations.py.

    Returns:
        activation: A callable activation function.
    """
    if callable(activation_function):
        return activation_function
    assert hasattr(activations, activation_function), f"{activation_function} is not defined in activations.py"
    activation = getattr(activations, activation_function)
    assert callable(activation), f"Expected a callable activation function but recieved a {type(activation)}"
    return activation


def mlp(hidden, in_weight, in_bias, out_weight, out_bias,
        activation_function, tp_degree,
        dequant_dtype=None, u8_bounds=None, in_scales=None, out_scales=None,
        neuron_config=None, transposed=False,
):
    # single:
    #   hidden: [h, a, b]
    #   in_weight: [h, 4h]
    #   in_bias: [4h]
    #   out_weight: [4h, h] or [h, 4h] when transposed
    #   out_bias: [h]
    # t-way tp:
    #   hidden: [h, a, b]
    #   in_weight: [h, 4h/t]
    #   in_bias: [4h/t]
    #   out_weight: [4h/t, h] or [h, 4h/t] when transposed
    #   out_bias: [h]
    dtype = hidden.dtype
    if u8_bounds is not None:
        *_, in_min, in_max, out_min, out_max = u8_bounds
        in_weight = u8_decode(dtype, dequant_dtype, in_weight, in_min, in_max)
        out_weight = u8_decode(dtype, dequant_dtype, out_weight, out_min, out_max)
    hidden_size, n_active_tokens, batch_size = hidden_sizes = hidden.sizes
    hidden_r_sizes = hidden_size, n_active_tokens * batch_size
    hidden = reshape(hidden, hidden_r_sizes)

    if neuron_config is not None and neuron_config.weight_tiling:
        assert hidden_size % constants.TILE_SIZE == 0, \
            f"hidden size needs to be divisible by {constants.TILE_SIZE}" \
            f"in order to use weight tiling."

        bias_dimension = 1

        hidden_tiled_sizes = hidden_size // constants.TILE_SIZE, constants.TILE_SIZE, batch_size * n_active_tokens,
        hidden = reshape(hidden, hidden_tiled_sizes)
        hidden = dot_with_tiled_weight_add(
            hidden, in_weight, in_bias, [0,1],
            [neuron_config.mlp_in_weight_tiling_permute_order.index(0), neuron_config.mlp_in_weight_tiling_permute_order.index(1)],
            bias_dimension, in_scales, neuron_config)
        hidden_tiled_sizes = hidden.sizes[0], hidden.sizes[1] // constants.TILE_SIZE, constants.TILE_SIZE
        hidden = reshape(hidden, hidden_tiled_sizes)
        hidden = get_activation(activation_function)(hidden)
        hidden = dot_with_tiled_weight_add(hidden, out_weight, out_bias, [1,2],
            [neuron_config.mlp_out_weight_tiling_permute_order.index(0), neuron_config.mlp_out_weight_tiling_permute_order.index(1)],
            bias_dimension, out_scales, neuron_config)
    else:
        # (h, b * s) @ (h, i) contract=(0, 0) => (b * s, i)
        hidden = dot00_add1(hidden, in_weight, in_bias, in_scales, neuron_config)
        hidden = get_activation(activation_function)(hidden)

        if transposed:
            # (b * s, i) @ (h, i) contract=(1, 1) => (b * s, h)
            hidden = dot11_add1(hidden, out_weight, out_bias, out_scales, neuron_config)
        else:
            # (b * s, i) @ (i, h) contract=(1, 0) => (b * s, h)
            hidden = dot10_add1(hidden, out_weight, out_bias, out_scales, neuron_config)

    is_bsh = neuron_config and neuron_config.collectives_layout == LAYOUT_BSH
    if is_bsh:
        # (b * s, h) => (b, s, h)
        hidden = reshape(hidden, (batch_size, n_active_tokens, hidden_size))
    else:
        # (b * s, h) = > (h, s, b)
        hidden = transpose(hidden, 0, 1)
        hidden = reshape(hidden, hidden_sizes)

    dtype, replica_groups = utils.parse_dtype_replica_groups(neuron_config, tp_degree)
    if neuron_config is not None and neuron_config.is_sequence_parallel:
        hidden = reduce_scatter_sum(hidden, tp_degree=tp_degree, dim=1, replica_groups=replica_groups, dtype=dtype)
    else:
        hidden = all_reduce_sum(hidden, tp_degree, dtype=dtype, replica_groups=replica_groups)

    # Transpose back to HSB if applicable
    return permute(hidden, (2, 1, 0)) if is_bsh else hidden


def mlp_bsh(hidden, in_weight, in_bias, out_weight, out_bias,
            activation_function, tp_degree,
            dequant_dtype=None, u8_bounds=None, in_scales=None, out_scales=None,
            neuron_config=None, transposed=False,
):
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
        *_, in_min, in_max, out_min, out_max = u8_bounds
        in_weight = u8_decode(dtype, dequant_dtype, in_weight, in_min, in_max)
        out_weight = u8_decode(dtype, dequant_dtype, out_weight, out_min, out_max)
    batch_size, n_active_tokens, hidden_size = hidden_sizes = hidden.sizes
    hidden_r_sizes = batch_size * n_active_tokens, hidden_size
    hidden = reshape(hidden, hidden_r_sizes)

    if neuron_config is not None and neuron_config.weight_tiling:
        assert hidden_size % constants.TILE_SIZE == 0, \
            f"hidden size needs to be divisible by {constants.TILE_SIZE}" \
            f"in order to use weight tiling."

        lhs_contracting_dimensions = [1,2]

        bias_dimension = 1
        hidden_tiled_sizes = batch_size * n_active_tokens, hidden_size // constants.TILE_SIZE, constants.TILE_SIZE
        hidden = reshape(hidden, hidden_tiled_sizes)
        hidden = dot_with_tiled_weight_add(hidden, in_weight, in_bias,
            lhs_contracting_dimensions,
            [neuron_config.mlp_in_weight_tiling_permute_order.index(0), neuron_config.mlp_in_weight_tiling_permute_order.index(1)],
            bias_dimension, in_scales, neuron_config)
        hidden_tiled_sizes = hidden.sizes[0], hidden.sizes[1] // constants.TILE_SIZE, constants.TILE_SIZE
        hidden = reshape(hidden, hidden_tiled_sizes)
        hidden = get_activation(activation_function)(hidden)
        hidden = dot_with_tiled_weight_add(hidden, out_weight, out_bias,
            lhs_contracting_dimensions,
            [neuron_config.mlp_out_weight_tiling_permute_order.index(0), neuron_config.mlp_out_weight_tiling_permute_order.index(1)],
            bias_dimension, out_scales, neuron_config)
    else:
        hidden = dot10_add1(hidden, in_weight, in_bias, in_scales, neuron_config)
        hidden = get_activation(activation_function)(hidden)
        hidden = dot10_add1(hidden, out_weight, out_bias, out_scales, neuron_config)
    hidden = reshape(hidden, hidden_sizes)

    dtype, replica_groups = utils.parse_dtype_replica_groups(neuron_config, tp_degree)
    if neuron_config is not None and neuron_config.is_sequence_parallel:
        hidden = reduce_scatter_sum(hidden, tp_degree=tp_degree, dim=1, replica_groups=replica_groups, dtype=dtype)
    else:
        hidden = all_reduce_sum(hidden, tp_degree, dtype=dtype, replica_groups=replica_groups)

    return hidden


def gated_mlp_bsh(
    hidden,
    in0_weight,
    in1_weight,
    out_weight,
    in0_scales=None,
    in1_scales=None,
    out_scales=None,
    in0_bias=None,
    in1_bias=None,
    out_bias=None,
    activation_function='silu',
    tp_degree=1,
    neuron_config=None,
    return_partial=False,
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

    hidden = reshape(hidden, hidden_r_sizes)
    if neuron_config is not None and neuron_config.weight_tiling:
        assert hidden_size % constants.TILE_SIZE == 0, \
            f"hidden size needs to be divisible by {constants.TILE_SIZE}" \
            f"in order to use weight tiling."
        hidden_tiled_sizes = batch_size * n_active_tokens, hidden_size // constants.TILE_SIZE, constants.TILE_SIZE
        hidden = reshape(hidden, hidden_tiled_sizes)
        hidden_active = dot_1220_add1(hidden, in0_weight, in0_bias, in0_scales, neuron_config)
        if neuron_config and neuron_config.fuse_mlp:
            hidden_tiled_sizes = hidden_active.sizes[0], 2, hidden_active.sizes[1] // (2 * constants.TILE_SIZE), \
                                 constants.TILE_SIZE
        else:
            hidden_tiled_sizes = hidden_active.sizes[0], hidden_active.sizes[1] // constants.TILE_SIZE, constants.TILE_SIZE
        hidden_active = reshape(hidden_active, hidden_tiled_sizes)
        if neuron_config and neuron_config.fuse_mlp:
            hidden_gate = slice_along(hidden_active, 1, limit=1, start=0)
            hidden_linear = slice_along(hidden_active, 1, limit=2, start=1)
            hidden_gate = squeeze(hidden_gate, dim=1)
            hidden_linear = squeeze(hidden_linear, dim=1)
            hidden_active = get_activation(activation_function)(hidden_gate)
        else:
            hidden_active = get_activation(activation_function)(hidden_active)
            hidden_linear = dot_1220_add1(hidden, in1_weight, in1_bias, in1_scales, neuron_config)
            hidden_linear = reshape(hidden_linear, hidden_tiled_sizes)
        hidden_states = multiply(hidden_active, hidden_linear)
        result = dot_1220_add1(hidden_states, out_weight, out_bias,
                        scales=out_scales, neuron_config=neuron_config)
    else:
        hidden_active = dot10_add1(hidden, in0_weight, in0_bias,
                               scales=in0_scales, neuron_config=neuron_config)
        if neuron_config and neuron_config.fuse_mlp:
            size = hidden_active.sizes[1]//2
            hidden_gate = slice_along(hidden_active, 1, limit=size, start=0)
            hidden_linear = slice_along(hidden_active, 1, limit=2*size, start=size)
            hidden_active = get_activation(activation_function)(hidden_gate)
        else:
            hidden_active = get_activation(activation_function)(hidden_active)
            hidden_linear = dot10_add1(hidden, in1_weight, in1_bias,
                               scales=in1_scales, neuron_config=neuron_config)
        hidden_states = multiply(hidden_active, hidden_linear)
        if neuron_config is not None and neuron_config.mlp_out_weight_transpose:
            result = dot10_add1(hidden_states, out_weight, out_bias,
                            scales=out_scales, neuron_config=neuron_config)
        else:
            result = dot11_add1(hidden_states, out_weight, out_bias,
                            scales=out_scales, neuron_config=neuron_config)
    result = reshape(result, hidden_sizes)

    if not return_partial:
        dtype, replica_groups = utils.parse_dtype_replica_groups(neuron_config, tp_degree)
        if neuron_config is not None and neuron_config.is_sequence_parallel:
            result = reduce_scatter_sum(result, tp_degree=tp_degree, dim=1, replica_groups=replica_groups, dtype=dtype)
        else:
            result = all_reduce_sum(result, tp_degree, dtype=dtype, replica_groups=replica_groups)
    return result


def gated_mlp(
    hidden,
    in0_weight,
    in1_weight,
    out_weight,
    in0_scales=None,
    in1_scales=None,
    out_scales=None,
    in0_bias=None,
    in1_bias=None,
    out_bias=None,
    activation_function='silu',
    tp_degree=1,
    neuron_config=None,
    return_partial=False,
):
    """
    An attention MLP using 2 input projections as found in LLama.

    Reference: https://github.com/huggingface/transformers/blob/v4.29.2/src/transformers/models/llama/modeling_llama.py#L144

    Sizes:

        i = n / tp

        hidden:     [h, s, b]
        in0_weight: [h, i]
        in1_weight: [h, i]
        out_weight: [h, i]
        in0_bias:   [i]
        in1_bias:   [i]
        out_bias:   [h]
        result:     [h, s, b]
    """

    dtype = hidden.dtype
    hidden_size, n_active_tokens, batch_size = hidden_sizes = hidden.sizes
    hidden_r_sizes = hidden_size, n_active_tokens * batch_size
    hidden = reshape(hidden, hidden_r_sizes)

    if len(in0_weight.sizes) == 4:
        assert hidden_size % constants.TILE_SIZE == 0, \
                f"hidden size needs to be divisible by {constants.TILE_SIZE}" \
                f"in order to use weight tiling."
        hidden_tiled_sizes = hidden_size // constants.TILE_SIZE, constants.TILE_SIZE, batch_size * n_active_tokens,
        hidden = reshape(hidden, hidden_tiled_sizes)

        hidden_active = dot_0120_add1(hidden, in0_weight, in0_bias,
                                      scales=in0_scales, neuron_config=neuron_config)
        if neuron_config and neuron_config.fuse_mlp:
            hidden_tiled_sizes = n_active_tokens * batch_size, 2, hidden_active.sizes[1] // (2 * constants.TILE_SIZE), \
                                 constants.TILE_SIZE
            hidden_active = reshape(hidden_active, hidden_tiled_sizes)
            hidden_gate = slice_along(hidden_active, 1, limit=1, start=0)
            hidden_linear = slice_along(hidden_active, 1, limit=2, start=1)
            hidden_gate = squeeze(hidden_gate, dim=1)
            hidden_linear = squeeze(hidden_linear, dim=1)
            hidden_active = get_activation(activation_function)(hidden_gate)
            hidden_states = multiply(hidden_active, hidden_linear)
        else:
            hidden_active = get_activation(activation_function)(hidden_active)
            hidden_linear = dot_0120_add1(hidden, in1_weight, in1_bias,
                                          scales=in1_scales, neuron_config=neuron_config)

            hidden_states = multiply(hidden_active, hidden_linear)
            hidden_states_tiled_sizes = hidden_states.sizes[0], hidden_states.sizes[1] // constants.TILE_SIZE, constants.TILE_SIZE
            hidden_states = reshape(hidden_states, hidden_states_tiled_sizes)

        result = dot_1220_add1(hidden_states, out_weight, out_bias,
                               scales=out_scales, neuron_config=neuron_config)
    else:
        # (h, b * s) @ (h, i) contract=(0, 0) => (b * s, i)
        hidden_active = dot00_add1(hidden, in0_weight, in0_bias, scales=in0_scales, neuron_config=neuron_config)
        if neuron_config and neuron_config.fuse_mlp:
            size = hidden_active.sizes[1]//2
            hidden_gate = slice_along(hidden_active, 1, limit=size, start=0)
            hidden_linear = slice_along(hidden_active, 1, limit=2*size, start=size)
            hidden_active = get_activation(activation_function)(hidden_gate)
        else:
            hidden_active = get_activation(activation_function)(hidden_active)

            # (h, b * s) @ (h, i) contract=(0, 0) => (b * s, i)
            hidden_linear = dot00_add1(hidden, in1_weight, in1_bias, scales=in1_scales, neuron_config=neuron_config)
        hidden_states = multiply(hidden_active, hidden_linear)

        if neuron_config is not None and neuron_config.mlp_out_weight_transpose:
            # (b * s, i) @ (i, h) contract=(1, 0) => (b * s, h)
            result = dot10_add1(hidden_states, out_weight, out_bias, scales=out_scales, neuron_config=neuron_config)
        else:
            # (b * s, i) @ (h, i) contract=(1, 1) => (b * s, h)
            result = dot11_add1(hidden_states, out_weight, out_bias, scales=out_scales, neuron_config=neuron_config)

    is_bsh = neuron_config and neuron_config.collectives_layout == LAYOUT_BSH

    if is_bsh:
        # (b * s, h) => (b, s, h)
        result = reshape(result, (batch_size, n_active_tokens, hidden_size))
    else:
        # (b * s, h) = > (h, s, b)
        result = transpose(result, 0, 1)
        result = reshape(result, hidden_sizes)

    if not return_partial:
        dtype, replica_groups = utils.parse_dtype_replica_groups(neuron_config, tp_degree)
        if neuron_config is not None and neuron_config.is_sequence_parallel:
            result = reduce_scatter_sum(result, tp_degree=tp_degree, dim=1, replica_groups=replica_groups, dtype=dtype)
        else:
            result = all_reduce_sum(result, tp_degree, dtype=dtype, replica_groups=replica_groups)

    # Transpose back to HSB if applicable
    return permute(result, (2, 1, 0)) if is_bsh else result


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


def softmax(logits, dim=None, tp_degree=1):
    """
    When tp_degree > 1 this function assumes that the softmax operation is
    sharded over the `dim` dimension.
    """
    rank = len(logits.sizes)
    if dim is None:
        dim = rank - 1
    dims = list(range(rank))
    dims.pop(dim)

    maximum = reduce_max(logits, dim)
    if tp_degree > 1:
        maximum = all_reduce_max(maximum, tp_degree=tp_degree)
    maximum = broadcast(maximum, logits.sizes, dims)

    difference = subtract(logits, maximum)
    exponential = exp(difference)

    denominator = reduce_sum(exponential, dim)
    if tp_degree > 1:
        denominator = all_reduce_sum(denominator, tp_degree=tp_degree)
    denominator = broadcast(denominator, logits.sizes, dims)

    return divide(exponential, denominator)


def transfer_with_static_ring(shape):
    custom_call_target = 'AwsNeuronTransferWithStaticRing'
    return shape.dtype[shape.sizes].CustomCall(shape, custom_call_target=custom_call_target)

def token_tree_attention_mask(full_token_tree_attention_mask, active_mask):
    """
    The full_token_tree_attention_mask is loaded as a weight of the speculation model and contains
    the attention mask for the whole tree structure.
    This method is used to extract the active attention mask corresponding to a tree structure
    that is a sub tree of the full token tree but only till a given level.
    So this is extracting a top left prefix of the full_token_tree_attention_mask based on the
    number of nodes in the sub tree which is determined by the size of active_mask.

    Eg:

       Full token tree : {0:[1], 1:[2], 2:[3]}
       Corresponding full_token_tree_attention_mask : [[1, 0, 0, 0],[1, 1, 0, 0],[1, 1, 1, 0], [1, 1, 1, 1]]
       sub tree as part of draft speculation loop : {0:[1], 1:[2]}
       So the shape of the active_mask will be (B = 1, SL = 3, SL = 3)
       Final mask returned will then be [[[True, False, False],[True, True, False],[True, True, True]]]
       of shape (B=1, SL=3, SL=3) which is the upper left 3x3 matrix of the full_token_tree_attention_mask matrix
    """
    batch, seq_len, *_ = active_mask.sizes
    # Keep required rows
    row_sliced_token_tree_active_mask = slice_along(full_token_tree_attention_mask, 0, seq_len)
    # Keep required columns
    col_sliced_token_tree_active_mask = slice_along(row_sliced_token_tree_active_mask, 1, seq_len)
    # Broadcast to accomodate correct shape of the mask
    token_tree_active_mask = broadcast(col_sliced_token_tree_active_mask, active_mask.sizes, [1,2])
    # Convert to pred
    pred_token_tree_active_mask = equal(token_tree_active_mask, 1)
    return pred_token_tree_active_mask


def attention_mask(cache_ids, start_ids, n_positions, last_token_id=None, num_active_blocks=None, neuron_config=None):
    """
    Create decomposed prior/active attention masks.

    The attention mask generated depends on the padding/alignment style used
    and which mode the model is being used for:
    - Single token generation
    - Parallel context encoding (prefill)
    - Speculative decoding
    - Windowed context encoding (prefill)
    The required mask(s) will be derived from the input parameters.

    Arguments:
        cache_ids: The positions to update in the KV cache.
        start_ids: The padding/batch offset for each batch line.
        n_positions: The total size of the KV cache to consider. This is
            equal to the size of the bucket.

    Returns:
        prior_mask: The attention mask to apply to the K/V cache
        active_mask: The attention mask to apply to the active tokens.

    Implementation Notes:
    ---------------------
    The goal of creating multiple masks for both the prior state and the
    active state is to enable the calculation of the attention score
    with fewer data dependencies than a traditional attention mechanism.

    Traditionally the active K/V is inserted (scatter) into the K/V cache prior
    to computing the attention score/context. This means that the computations
    have a data dependency on both the large K/V cache and the small active K/V
    (exactly 1 during auto-regressive token generation).

    Using a decomposed attention calculation is significantly faster
    since it allows the different portions of the attention to be
    scheduled more flexibly and does not introduce a data dependency on a
    relatively slow operation (scatter).
    """
    if len(cache_ids.sizes) == 2:
        # TODO: support lhs_aligned flag and the related attention mask
        return decoder_attention_mask_lhs_aligned(
            cache_ids,
            n_positions,
            last_token_id=last_token_id,
            num_active_blocks=num_active_blocks,
            neuron_config=neuron_config,
        )
    else:
        n_active_tokens, = cache_ids.sizes
        use_prefetch = n_active_tokens != n_positions
        triu_comparison = 'LT' if use_prefetch else 'LE'
        return decoder_attention_mask(
            start_ids,
            cache_ids,
            n_positions,
            triu_comparison=triu_comparison,
            allow_kv_dot_prefetch=use_prefetch,
            start_mask=True,
        )


def decoder_attention_mask(start_ids, position_ids, n_positions, triu_comparison='LE',
                           allow_kv_dot_prefetch=False, start_mask=True):
    batch_size, = start_ids.sizes
    int_dtype = position_ids.dtype
    use_2d_cache_ids = len(position_ids.sizes) > 1
    if use_2d_cache_ids:
        _, n_active_tokens = position_ids.sizes  # 2d position_ids

        # TODO: fix the following broadcast for 2D position_ids
        position_ids = int_dtype[n_active_tokens].Iota(dimensions=[0])
    else:
        n_active_tokens,  = position_ids.sizes  # 1d position_ids
    triu_sizes = n_active_tokens, n_positions
    pred = position_ids.scribe.pred

    # Windowed & speculative attention
    if n_active_tokens > 1 and allow_kv_dot_prefetch:
        return decoder_attention_mask_window(position_ids, start_ids, n_positions)

    if batch_size == 1 and n_active_tokens > 1 and n_positions == n_active_tokens:
        position_ids = int_dtype[n_active_tokens].Iota(dimensions=[0])
        start_ids = int_dtype[1].Iota(dimensions=[0])
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


def all_reduce(tensor, replica_groups, to_apply, dtype=None):
    size = tensor.sizes
    tensor_dtype = tensor.dtype
    scribe = tensor.scribe

    if dtype is None:
        all_reduce_dtype = tensor_dtype
    elif isinstance(dtype, str):
        all_reduce_dtype = dtypes.to_pyhlo_type(scribe, dtype)
    else:
        all_reduce_dtype = dtype

    tensor = cast(tensor, all_reduce_dtype)

    result = all_reduce_dtype[size].AllReduce(
        tensor,
        replica_groups=replica_groups,
        to_apply=to_apply
    )

    result = cast(result, tensor_dtype)

    return result


def all_gather(tensor, dim, tp_degree, replica_groups=None):
    shape = list(tensor.sizes)
    dtype = tensor.dtype

    if replica_groups is None:
        replica_groups = [list(range(tp_degree))]
        shape[dim] *= tp_degree
    else:
        shape[dim] *= len(replica_groups[0])

    return dtype[shape].AllGather(
        tensor,
        dimensions=[dim],
        replica_groups=replica_groups,
    )


def all_reduce_sum(tensor, tp_degree, dtype=None, replica_groups=None):

    if tp_degree == 1:
        return tensor

    scribe = tensor.scribe

    if dtype is None:
        all_reduce_dtype =  tensor.dtype
    elif isinstance(dtype, str):
        all_reduce_dtype = dtypes.to_pyhlo_type(scribe, dtype)
    else:
        all_reduce_dtype = dtype

    if replica_groups is None:
        replica_groups = [list(range(tp_degree))]

    def reducer(scribe):
        p0 = all_reduce_dtype.Parameter(parameter_number=0)
        p1 = all_reduce_dtype.Parameter(parameter_number=1)
        return all_reduce_dtype.Add(p0, p1)

    return all_reduce(
        tensor,
        replica_groups=replica_groups,
        to_apply=reducer,
        dtype=all_reduce_dtype
    )


def _all_to_all(tensor, split_dim, concat_dim, tp_degree):
    # Add extra reshape (and potentially transpose) before and after all-to-all CC, due to
    # 1) hlo_to_mlir_hlo would assume split_dim == concat_dim
    # 2) split_count is taken from first replica_group
    # 3) we can only split/concat outer most dimension
    shape = list(tensor.sizes)
    dtype = tensor.dtype
    assert (split_dim == 0 and concat_dim == 1) or (split_dim == 1 and concat_dim == 0), \
        f"invalid input dimensions: split_dim={split_dim}, concat_dim={concat_dim}"
    assert shape[split_dim] >= tp_degree and shape[split_dim] % tp_degree == 0, \
        f"invalid split size ({shape[split_dim]}) and tp_degree ({tp_degree})"
    shape_flat = shape[1:].copy()
    shape_flat[0] *= shape[0]

    # Assume input_shape = [N,M]
    # split_dim == 0 and concat_dim == 1, new shape = [M*tp_degree, N/tp_degree]
    # concat_dim == 0 and split_dim == 1, new shape = [N*tp_degree, M/tp_degree]
    shape_new = shape.copy()
    shape_new[split_dim] = shape_new[split_dim] // tp_degree
    shape_new[concat_dim] *= tp_degree
    if split_dim == 0:
        assert concat_dim == 1
        shape_new[1], shape_new[0] = shape_new[0], shape_new[1]

    if split_dim == 1:
        assert concat_dim == 0
        tensor = transpose(tensor, 0, 1)
    tensor = reshape(tensor, shape_flat)
    tensor = dtype[shape_flat].AllToAll(
        tensor,
        dimensions=[0],
        replica_groups=[list(range(tp_degree))],
    )
    tensor = reshape(tensor, shape_new)
    if split_dim == 0:
        assert concat_dim == 1
        tensor = transpose(tensor, 0, 1)
    return tensor


def all_to_all(tensor, split_dim, concat_dim, tp_degree):
    # Handle the case when split_dim==0 and concat_dim=0
    if split_dim == 0 and concat_dim == 0:
        tensor = hlo.unsqueeze(tensor, dim=0)
        tensor = _all_to_all(tensor, split_dim=1, concat_dim=0, tp_degree=tp_degree)
        tensor = hlo.squeeze(tensor, dim=1)
    else:
        tensor = _all_to_all(tensor, split_dim, concat_dim, tp_degree)
    return tensor


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


def gather_select_reduce(tensor, dim, index):
    """
    Gather elements from a `tensor` along `dim` at the given `index`
    using select and reduce op.
    """
    assert dim <= len(tensor.sizes)

    # Must have the same rank
    tensor_sizes = list(tensor.sizes)
    index_sizes = list(index.sizes)
    assert len(tensor_sizes) == len(index_sizes)

    index_broadcast_dims = list()
    input_broadcast_dims = list()
    sizes = list()

    for i in range(0, len(index_sizes)):
        if i < dim:
            input_broadcast_dims.append(i)
            index_broadcast_dims.append(i)
        elif i == dim:
            sizes.append(tensor_sizes[i])
            input_broadcast_dims.append(i)
            index_broadcast_dims.append(i + 1)
        else:
            input_broadcast_dims.append(i + 1)
            index_broadcast_dims.append(i + 1)
        sizes.append(index_sizes[i])

    cast_index = broadcast(index, sizes, index_broadcast_dims)
    ref_iota = iota(index.dtype, sizes, dim)
    mask = equal(cast_index, ref_iota)

    cast_input = broadcast(tensor, sizes, input_broadcast_dims)
    zero_full = full(0, cast_input.dtype, cast_input.sizes)
    masked_input = cast_input.dtype[cast_input.sizes].Select(mask, cast_input, zero_full)

    def reducer(scribe):
        p0 = dtype.Parameter(parameter_number=0)
        p1 = dtype.Parameter(parameter_number=1)
        return dtype.Add(p0, p1)

    dtype = masked_input.dtype
    zero = dtype.Constant(constant_value=0)
    result = dtype[tensor_sizes].Reduce(masked_input, zero, dimensions=[dim], to_apply=reducer)
    return result


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
        index = reshape(index, keepdim_shape)
        index = cast(index, u32)

    if return_values:
        return reduce_max(tensor, dim, keepdim), index
    return index


def argmax(tensor, dim, keepdim=False, return_values=False, tp_degree=1):

    if tp_degree == 1:
        return _argmax(tensor, dim, keepdim, return_values=return_values)

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
    replica_size = broadcast(replica_size, sizes, [])
    replica_ids = iota(dtype, sizes, dim)
    offset = multiply(replica_ids, replica_size)
    index = add(index, offset)

    # Find replica with globally maximum value
    replica_index = _argmax(value, dim, keepdim=True)

    # Final masked reduction
    dimensions = list(range(len(replica_index.sizes) + 1))
    dimensions.pop(dim)

    rs_size = list(replica_index.sizes)
    rs_size[dim] *= tp_degree
    br_size = list(replica_index.sizes)
    br_size.insert(dim, tp_degree)
    replica_index = broadcast(replica_index, br_size, dimensions)
    replica_index = reshape(replica_index, rs_size)

    mask = compare(replica_index, replica_ids, direction='EQ')
    mask_index = cast(mask, index.dtype)
    masked = multiply(index, mask_index)
    index = reduce_sum(masked, dim=dim, keepdim=keepdim)

    if return_values:
        value = reduce_max(value, dim=dim, keepdim=keepdim)
        return value, index
    return index


def _embedding(weight, index, dtype=None):
    """
    Performs embedding on a single partition
    """
    assert len(weight.sizes) == 2, (
        f'Expected rank 2 embedding weights but found shape: {weight.sizes}'
    )

    n_embedding, embedding_dim = weight.sizes

    # Linearize index tensor to gather from 0th dimension
    n_index = functools.reduce(operator.mul, index.sizes, 1)
    linear_index = reshape(index, n_index)

    # Gather
    result = weight.dtype[n_index, embedding_dim].Gather(
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
    if dtype != result.dtype:
        result = cast(result, dtype)

    # Reshape embedding tensor to look like the original index shape
    return reshape(result, (*index.sizes, embedding_dim))


def embedding(weight, index, tp_degree=1, dim=1, dtype=None):
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
    result = _embedding(weight, offset, dtype)

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


def concatenate(operands, dimension):
    # Concatenates a sequence of arrays along dimension.
    dtype = operands[0].dtype
    sizes = list(operands[0].sizes)
    for op_idx in range(1, len(operands)):
        for dim_idx in range(len(sizes)):
            if dim_idx != dimension:
                assert sizes[dim_idx] == operands[op_idx].sizes[dim_idx], \
                    "All tensors must have the same shape (except in the concatenating dimension)."
        sizes[dimension] = sizes[dimension] + operands[op_idx].sizes[dimension]
    output = dtype[sizes].Concatenate(*operands, dimensions=[dimension])
    return output


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

def quantize_kv_cache_direct_cast(tensor, neuron_config: NeuronConfig):
    scribe = tensor.scribe
    quant_dtype = getattr(scribe, neuron_config.kv_cache_quant.quant_dtype)
    quantized_tensor = quant_dtype[tensor.sizes].Convert(tensor)
    return quantized_tensor
  
def dequantize_kv_cache_direct_cast(tensor, neuron_config: NeuronConfig):
    scribe = tensor.scribe
    dequant_dtype = getattr(scribe, neuron_config.kv_cache_quant.dequant_dtype)
    dequantized_tensor = dequant_dtype[tensor.sizes].Convert(tensor)
    return dequantized_tensor

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


def all_reduce_mean(tensor, tp_degree, dtype=None, replica_groups=None):
    """
    Calculate the "global" mean for elements in a group where each group's
    global_mean = (intermediate_mean0 + intermediate_mean1 + ...) / group_size for intermediate_meani in each group

    Example 1:

        tp_degree = 2
        replica_groups = [[0, 1]]
        sharding=(0,)
        inputs = torch.tensor([
            [1, 2, 3],  # Rank 0
            [5, 6, 7],  # Rank 1
        ])
        all_reduce_mean on NC 0 = torch.tensor([
            [3, 4, 5]
        ])

    Example 2:

        tp_degree = 2
        replica_groups = [[0], [1]]
        sharding=(0,)
        inputs = torch.tensor([
            [1, 2, 3],  # Rank 0
            [5, 6, 7],  # Rank 1
        ])
        all_reduce_mean on NC 0 = torch.tensor([
            [1, 2, 3]
        ])
    """

    if tp_degree == 1:
        return tensor

    scribe = tensor.scribe
    size = tensor.sizes

    if dtype is None:
        all_reduce_dtype = tensor.dtype
    elif isinstance(dtype, str):
        all_reduce_dtype = dtypes.to_pyhlo_type(scribe, dtype)
    else:
        all_reduce_dtype = dtype

    if replica_groups is None:
        replica_groups = [list(range(tp_degree))]

    group_size = len(replica_groups[0])
    assert all(
        len(group) == group_size for group in replica_groups
    ), "All groups must have the same size"

    mean_sum = all_reduce_sum(
        tensor, tp_degree, dtype=all_reduce_dtype, replica_groups=replica_groups
    )
    divisor = full(group_size, all_reduce_dtype, size)
    global_mean = divide(mean_sum, divisor)

    return global_mean


def cumsum(tensor, dim):

    if is_floating_point(tensor):
        return _cumsum_fast(tensor, dim)

    scribe = tensor.scribe
    s32 = scribe.s32
    pred = scribe.pred

    last = len(tensor.sizes) - 1
    dtype = tensor.dtype

    if dim < 0:
        dim %= len(tensor.sizes)

    if dim != last:
        tensor = transpose(tensor, dim, last)

    size = tensor.sizes[last]
    sizes = (size, size)

    # Build triu mask
    a = s32[sizes].Iota(dimensions=[0])
    b = s32[sizes].Iota(dimensions=[1])
    triu = pred[sizes].Compare(a, b, comparison_direction='LE')
    triu = dtype[sizes].Convert(triu)

    # Cumulative sum along final dimension
    result = dtype[tensor.sizes].Dot(tensor, triu, dot_dimension_numbers=dict(
        lhs_contracting_dimensions=[last],
        rhs_contracting_dimensions=[0]
    ))
    if dim != last:
        result = transpose(result, dim, last)

    return result


def _cumsum_fast(tensor, dim):

    try:
        from neuronxcc.nki._private_kernels.cumsum import cumsum as nki_cumsum
    except ImportError:
        from neuronxcc.nki.kernels.cumsum import cumsum as nki_cumsum

    scribe = tensor.scribe
    last = len(tensor.sizes) - 1
    dtype = tensor.dtype
    f32 = scribe.f32

    if dim < 0:
        dim %= len(tensor.sizes)

    # Note: NKI kernel must do accumulation on the last dim
    if dim != last:
        tensor = transpose(tensor, dim, last)

    # Note: NKI kernel only supports 2d tensors
    reshaped = False
    shape = tensor.sizes
    if len(shape) != 2:
        sizes = list(tensor.sizes)
        sizes.pop(last)
        elements = functools.reduce(operator.mul, sizes, 1)
        tensor = reshape(tensor, (elements, tensor.sizes[last]))
        reshaped = True

    def _cumsum(inputs, output):
        return nki_cumsum(inputs, output, axis=1)
    result = nki_call(_cumsum, tensor, output_HloShapes=tensor.dtype[tensor.sizes])

    if reshaped:
        result = reshape(result, shape)

    if dim != last:
        result = transpose(result, dim, last)

    return result


def _cumsum_reduce_window(tensor, dim):
    # PERF: Scales poorly with large tensors

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


def slice_along(tensor, dim, limit, start=0, stride=1):
    """
    Slice along a dimension.
    """
    dimensions = [
        dict(start=0, limit=size, stride=1) for size in tensor.sizes
    ]
    dimensions[dim] = dict(start=start, limit=limit, stride=stride)

    sizes = list(tensor.sizes)
    sizes[dim] = (limit - start + stride - 1)//stride

    return tensor.dtype[sizes].Slice(
        tensor,
        slice_dimensions=dimensions
    )


def dynamic_slice_along(tensor, dim, start, size):

    scribe = tensor.scribe
    s32 = scribe.s32
    u32 = scribe.u32
    s64 = scribe.s64
    u64 = scribe.u64
    dtype = tensor.dtype

    assert isinstance(size, int), (
        f"Parameter 'size' must be an integer. Found type={type(size)}"
    )
    assert not isinstance(start, int), (
        f"Parameter 'start' must be a tensor. Found type={type(size)}"
    )
    assert len(start.sizes) == 0, (
        f"Parameter 'start' must be a scalar. Found shape={start.sizes}"
    )
    assert start.dtype in (s32, u32, s64, u64), (
        f"Parameter 'start' must be an integer type."
    )

    sizes = list(tensor.sizes)
    assert size <= sizes[dim], (
        f"Parameter 'size' ({size}) must less/equal to {sizes[dim]}. (dim={dim}, shape={sizes})"
    )
    sizes[dim] = size

    start = cast(start, s32)
    zero = s32.Constant(constant_value=0)
    starts = [zero] * len(sizes)
    starts[dim] = start

    return dtype[sizes].DynamicSlice(
        tensor,
        *starts,
        dynamic_slice_sizes=sizes,
    )


def dynamic_update_slice(tensor, update, start_indices):

    assert len(tensor.sizes) == len(update.sizes), (
        f"Parameter tensor and update must have same dim, get {len(tensor.sizes)} and {len(update.sizes)}"
    )

    assert len(start_indices) == len(tensor.sizes), (
        f"Parameter tensor and start_indices must have same dims, get {len(start_indices)} and {tensor.sizes}"
    )

    assert isinstance(start_indices, list), (
        f"Parameter 'start_indices' must be an list. Found type={type(start_indices)}"
    )

    if isinstance(start_indices[0], int):
        start_indices = [tensor.scribe.u32.Constant(constant_value=i) for i in start_indices]

    return tensor.dtype[tensor.sizes].DynamicUpdateSlice(tensor, update, *start_indices)


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


def permute(tensor, dimensions):
    size = list(tensor.sizes)
    permuted_size = [size[dim] for dim in dimensions]
    return tensor.dtype[permuted_size].Transpose(tensor, dimensions=dimensions)


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

    value = get_tuple_element(results, tuple_index=0)
    index = get_tuple_element(results, tuple_index=1)

    return value, index


def full_like(tensor, value):
    dtype = tensor.dtype
    size = tensor.sizes
    return full(value, dtype, size)


def all_reduce_max(tensor, tp_degree=1, dtype=None, replica_groups=None):
    if tp_degree == 1:
        return tensor


    scribe = tensor.scribe

    if dtype is None:
        all_reduce_dtype = tensor.dtype
    elif isinstance(dtype, str):
        all_reduce_dtype = dtypes.to_pyhlo_type(scribe, dtype)
    else:
        all_reduce_dtype = dtype

    if replica_groups is None:
        replica_groups = [list(range(tp_degree))]

    def reducer(scribe):
        p0 = all_reduce_dtype.Parameter(parameter_number=0)
        p1 = all_reduce_dtype.Parameter(parameter_number=1)
        return all_reduce_dtype.Maximum(p0, p1)

    return all_reduce(
        tensor,
        replica_groups=replica_groups,
        to_apply=reducer,
        dtype=all_reduce_dtype
    )


def all_reduce_max_with_indices(tensor, index, tp_degree=1, dtype=None, replica_groups=None):
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
    maximum = all_reduce_max(tensor, tp_degree, dtype=dtype, replica_groups=replica_groups)

    # Zero out all rank-local indices which do not correspond global maximum
    mask = pred[size].Compare(tensor, maximum, comparison_direction='EQ')
    zero = full_like(tensor, 0)
    index = index.dtype[index.sizes].Select(mask, index, zero)

    # Reduce masked indices from across ranks (Note: Max instead of sum due to potential duplicate values)
    index = all_reduce_max(index, tp_degree, dtype=dtype, replica_groups=replica_groups)

    return maximum, index


def topk(tensor, dim, k=50, tp_degree=1):
    """
    Get the top-k values and indices along a dimension.

    When using a `tp_degree > 1`, this function assumes that the sharding
    dimension is the same as the reduction dimension. In this mode, the
    returned index is transformed into a global index by adding an offset
    of `tensor.sizes[dim] * rank_id`.

    Implementation Notes:
    - The `k` value may not be larger than tensor.shape[dim]. The dimension
      size may be smaller than anticipated with large tensor parallel degrees.
    - The Top-k custom call may only be invoked on the final dimension.
    - The input `tensor` size along dimension `dim` must be a multiple of 8.

    Arguments:
        tensor: The values to select the top-k from.
        dim: The dimension along which to select the values from.
        k: The number of values to select.
        tp_degree: The number of ranks to collect across.

    Returns:
        value: The top-k values in the tensor.
        index: The indices of the top-k values in the tensor.
    """

    if not isinstance(k, int) or (k < 1):
        raise ValueError(f"`k` has to be a positive integer, but is {k}")

    if k == 1:
        return argmax(tensor, dim, return_values=True, keepdim=True, tp_degree=tp_degree)

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
        tensor = all_gather(tensor, dim=dim, tp_degree=tp_degree)
        tp_degree = 1

    # Initial reduction to find rank-local top-k
    value, index = _topk(tensor, k)

    # Early exit if not doing a tensor parallel computation
    if tp_degree == 1:
        return output(value, index)

    # Combine initial reduction from all ranks
    value = all_gather(value, dim=dim, tp_degree=tp_degree)
    index = all_gather(index, dim=dim, tp_degree=tp_degree)

    # Add shard size offset to rank-local index to make it global
    dtype = index.dtype
    sizes = list(index.sizes)
    assert sizes[dim] == tp_degree * k, (
        f"Expected input dimension {dim} to be size {tp_degree * k} but found {sizes[dim]} (shape={sizes})"
    )
    sizes.insert(dim + 1, k)
    sizes[dim] = tp_degree
    rank_id = iota(dtype, sizes, dim)
    rank_id = reshape(rank_id, index.sizes)
    shard_size = full(tensor.sizes[dim], dtype, index.sizes)
    offset = multiply(rank_id, shard_size)
    offset = cast(offset, dtype)
    index = add(index, offset)

    # Final global reduction to find true top-k values
    value, replica_index = _topk(value, k)

    # Gather global index from the initial offset rank-local index
    index = gather(index, dim, replica_index)
    return output(value, index)


def topk_masked(tensor, dim, k=50, tp_degree=1, indices=None):
    """
    Get the top-k values and indices along a dimension. All other values
    in the tensor are "masked out" and set to -30000.

    When using a `tp_degree > 1`, this function assumes that the sharding
    dimension is the same as the reduction dimension. In this mode, the
    returned index is transformed into a global index by adding an offset
    of `tensor.sizes[dim] * rank_id`.

    Arguments:
        tensor: The values to select the top-k from.
        dim: The dimension along which to select the values from.
        k: The number of values to select.
        tp_degree: The number of ranks to collect across.
        indices: Optional indices of the pre-sorted tensor in descending order.
            If `indices` is provided it is assumed `tensor` is also sorted
            in descending order.

    Returns:
        value: The top-k values in the tensor.
        index: The indices of the top-k values in the tensor.
        indices: Indices of the pre-sorted tensor in descending order.
    """
    scribe = tensor.scribe
    s32 = scribe.s32

    k_is_hlo_scalar = _is_hlo_scalar(k)

    if not k_is_hlo_scalar:
        if not isinstance(k, int) or (k < 1):
            raise ValueError(f"`k` has to be a positive integer, but is {k}")
    else:
        converter = compiler.DataTypeConverter()
        k_torch_dtype = converter.hlo2torch(k.dtype.shape_proto.element_type)
        assert k_torch_dtype in [
            torch.int32,
            torch.int64,
        ], f"Expected `k` to be an integer, but `k` is {k_torch_dtype}"
        k = cast(k, tensor.dtype)

    if indices is None:
        if tp_degree > 1:
            tensor = all_gather(tensor, dim=dim, tp_degree=tp_degree)
        tensor, indices = sort_with_indices(tensor, dim=dim, descending=True)

    indices_dtype = indices.dtype
    indices = cast(indices, s32) # Cast to signed int32 to avoid creating a uint -30000 value

    positions = iota(tensor.dtype, tensor.sizes, dims=dim)
    keep_mask = less(positions, k)
    value = masked_select(keep_mask, tensor, -30000)
    index = masked_select(keep_mask, indices, -30000)

    indices = cast(indices, indices_dtype)

    return value, index, indices


def topp(tensor, top_p=1.0, top_p_min_tokens=1, tp_degree=1, indices=None, dim=0):
    """
    Get the smallest set of tensor values that add up to top_p or higher
    along a specified dimension.

    Arguments:
        tensor: The values to select the top-p from.
        top_p: The cumulative sum of values.
        top_p_min_tokens: The minimum number of tokens that cannot be filtered.
        tp_degree: The number of ranks to collect across.
        indices: Optional indices of the pre-sorted tensor in descending order.
            If `indices` is provided it is assumed `tensor` is also sorted
            in descending order.
        dim: The dimension along which to select the values from.

    Returns:
        probs: The top-p values in the tensor.
        indices: Indices of the pre-sorted tensor in descending order.
    """

    p_is_hlo_scalar = _is_hlo_scalar(top_p)
    top_p_min_tokens_is_hlo_scalar = _is_hlo_scalar(top_p_min_tokens)
    converter = compiler.DataTypeConverter()

    if not p_is_hlo_scalar:
        if top_p < 0 or top_p > 1.0:
            raise ValueError(f"`top_p` has to be a float > 0 and <= 1, but is {top_p}")
        top_p = tensor.dtype.Constant(constant_value=top_p)
    else:
        top_p_torch_dtype = converter.hlo2torch(top_p.dtype.shape_proto.element_type)
        assert top_p_torch_dtype == torch.float32, (
            f"Expected `top_p` to be a float, but `top_p` is {top_p_torch_dtype}."
        )
        top_p = cast(top_p, tensor.dtype)

    if not top_p_min_tokens_is_hlo_scalar:
        if not isinstance(top_p_min_tokens, int) or (top_p_min_tokens < 1):
            raise ValueError(
                f"`top_p_min_tokens` has to be a positive integer, but is {top_p_min_tokens}"
            )
        top_p_min_tokens = tensor.dtype.Constant(constant_value=top_p_min_tokens)
    else:
        top_p_min_tokens_torch_dtype = converter.hlo2torch(
            top_p_min_tokens.dtype.shape_proto.element_type
        )
        assert top_p_min_tokens_torch_dtype in [torch.int32, torch.int64], (
            "Expected `top_p_min_tokens` to be an integer,"
            f"but `top_p_min_tokens` is {top_p_min_tokens_torch_dtype}"
        )
        top_p_min_tokens = cast(top_p_min_tokens, tensor.dtype)

    scribe = tensor.scribe
    s32 = scribe.s32

    if indices is None:
        if tp_degree > 1:
            tensor = all_gather(tensor, dim=dim, tp_degree=tp_degree)
        tensor, indices = sort_with_indices(tensor, dim=dim, descending=True)

    # Probability mask - Keep only the positions whose cumulative
    #   probability is less than the user-provided threshold. This removes
    #   the long tail of small probababilitiess.
    probs = softmax(tensor, dim=dim)
    cumulative_prob = cumsum(probs, dim=dim)

    # This gives us one position before the total set of top_p tokens we want to keep
    keep_probs = less(cumulative_prob, top_p)

    # Get the "cutoff" position
    keep_probs = cast(keep_probs, s32)
    keep_positions = reduce_sum(keep_probs, dim=dim)
    positions = iota(probs.dtype, probs.sizes, dims=dim)

    # Positional mask - Support minimum number of positions.
    # Reference: https://github.com/huggingface/transformers/blob/v4.39.0/src/transformers/generation/logits_process.py#L409-L410
    criteria = subtract(top_p_min_tokens, 1)
    limit = maximum(criteria, keep_positions)
    dims = list(range(len(positions.sizes)))
    dims.pop(dim)
    limit = broadcast(limit, positions.sizes, dims)
    limit = cast(limit, positions.dtype)

    # Keep all required values and mask out the rest with -30000
    keep_mask = less_equal(positions, limit)
    probs = masked_select(keep_mask, tensor, -30000)

    return probs, indices


def numel(tensor):
    elements = 1
    for dim in tensor.sizes:
        elements *= dim
    return elements


def sort(tensor, dim=-1, descending=False):
    if dim < 0:
        dim = dim % len(tensor.sizes)

    dtype = tensor.dtype
    sizes = tensor.sizes

    def comparator(scribe):
        p0 = dtype.Parameter(parameter_number=0)
        p1 = dtype.Parameter(parameter_number=1)
        if descending:
            return greater(p0, p1)
        else:
            return less(p0, p1)

    return dtype[sizes].Sort(tensor, to_apply=comparator, dimensions=[dim])


def sort_with_indices(tensor, dim=-1, descending=False):
    if dim < 0:
        dim = dim % len(tensor.sizes)

    scribe = tensor.scribe
    dtype = tensor.dtype
    sizes = tensor.sizes
    s32 = scribe.s32

    def comparator(scribe):
        p0 = dtype.Parameter(parameter_number=0)
        p1 = dtype.Parameter(parameter_number=1)
        p2 = s32.Parameter(parameter_number=2)
        p3 = s32.Parameter(parameter_number=3)
        if descending:
            return greater(p0, p1)
        else:
            return less(p0, p1)

    indices = iota(s32, sizes, dims=[dim])
    result = scribe.tuple(dtype[sizes], s32[sizes]).Sort(tensor, indices, to_apply=comparator, dimensions=[dim])
    value = get_tuple_element(result, 0)
    index = get_tuple_element(result, 1)
    return value, index


def argsort(tensor, dim=-1, descending=False):
    _, index = sort_with_indices(tensor, dim, descending)
    return index


def multinomial(probabilities, dim, deterministic=False):
    """
    Single sample multinomial selection along a dimension.

    Input probabilities must be sorted in descending order along the
    dimension. This operator cannot validate that this prerequisite is true.

    Args:
        probabilities: The probabilities to sample. This must be sorted from
            in descending order.
        dim: The dimension to sample across.
        deterministic: Flag which enables using a constant 0.5 as token
            acceptance threshold instead of a random uniform. This is used for
            debug/testing.

    Returns:
        indices: The selected index across the given dimension.
    """
    scribe = probabilities.scribe
    dtype = probabilities.dtype
    u32 = scribe.u32

    cumprob = cumsum(probabilities, dim)

    sizes = list(probabilities.sizes)
    vocab_size = sizes.pop(dim)
    # Fix to ensure cumprob add up to 1.0 to prevent chance of
    # generating sample with value of vocab_size when cumprob is < 1.0.
    # As an example, cumprob can be 0.996 or 1.007 due to floating point accumulation error.
    max_prob = slice_along(cumprob, dim, vocab_size, start=vocab_size-1, stride=1)
    max_prob = broadcast(max_prob, cumprob.sizes, list(range(len(cumprob.sizes))))
    cumprob = divide(cumprob, max_prob)

    if deterministic:
        uniform = full(0.5, dtype, sizes)
    else:
        uniform = random_uniform(dtype, sizes)

    dims = list(range(len(probabilities.sizes)))
    dims.pop(dim)
    uniform = broadcast(uniform, probabilities.sizes, dims)

    result = greater(uniform, cumprob)
    result = cast(result, u32)
    result = reduce_sum(result, dim, keepdim=True)
    return result


def full(value, dtype, sizes):
    result = dtype.Constant(constant_value=value)
    result = dtype[sizes].Broadcast(result, dimensions=[])
    return result


# https://www.tensorflow.org/xla/operation_semantics#broadcastindim
def broadcast(tensor, out_dim_size, broadcast_dimensions):
    dtype = tensor.dtype
    sizes = list(tensor.sizes)

    assert len(broadcast_dimensions) == len(tensor.sizes), (
        f"Input operand rank ({len(tensor.sizes)}) does not match broadcast dimensions ({broadcast_dimensions})"
    )

    br_dims_to_keep = []
    reshape_sizes = []
    for i, (dim, size) in enumerate(zip(broadcast_dimensions, sizes)):

        # Broadcast dimension must be within the output shape
        assert dim < len(out_dim_size), (
            f"Broadcasting dimension {dim} is out of range of destination size {out_dim_size} (src={tensor.sizes} dst={out_dim_size})"
        )

        # Sizes of 1 may always be broadcast to any other size
        if size == 1:
            continue

        br_dims_to_keep.append(dim)
        reshape_sizes.append(size)

        # Broadcast dimension sizes must match when non-1
        dst = out_dim_size[dim]
        assert dst == size, (
            f"Non-1 broadcast source dimension ({i}) of size {size} "
            f"must match destination dimension ({dim}) of size {dst} "
            f"(src={tensor.sizes} dst={out_dim_size})"
        )

    # Legalize the Broadcast op input so that it complies with https://openxla.org/xla/operation_semantics#broadcast syntax
    tensor = reshape(tensor, reshape_sizes)
    output = dtype[out_dim_size].Broadcast(tensor, dimensions=br_dims_to_keep)
    # tensor = reshape(tensor, [])
    # output = dtype[out_dim_size].Broadcast(tensor, dimensions=[])
    return output


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

    # Convert boolean tensors to int8 to avoid an error in Python 3.11:
    #   `TypeError: True has type <class 'numpy.bool_'>, but expected one of: (<class 'bool'>, <class 'numbers.Integral'>)`
    original_dtype = dtype
    dtype_is_bool = dtype == dtype.scribe.pred
    if dtype_is_bool:
        dtype = dtype.scribe.s8

    # Convert tensor data to expected HLO data type
    torch_dtype = converter.hlo2torch(dtype.shape_proto.element_type)
    if tensor.dtype != torch_dtype:
        tensor = tensor.to(torch_dtype)

    data = tensor.data.numpy().ravel()
    if tensor.dtype in [torch.float16, torch.bfloat16, torch.int16, torch.int8]:
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
    result = cast(result, original_dtype)
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


def add(lhs, rhs):
    lhs, rhs = _binary_primitive_broadcast(lhs, rhs)
    _check_binary_arguments(lhs, rhs)
    return lhs.dtype[lhs.sizes].Add(lhs, rhs)


def subtract(lhs, rhs):
    lhs, rhs = _binary_primitive_broadcast(lhs, rhs)
    _check_binary_arguments(lhs, rhs)
    return lhs.dtype[lhs.sizes].Subtract(lhs, rhs)


def divide(lhs, rhs):
    lhs, rhs = _binary_primitive_broadcast(lhs, rhs)
    _check_binary_arguments(lhs, rhs)
    return lhs.dtype[lhs.sizes].Divide(lhs, rhs)


def multiply(lhs, rhs):
    lhs, rhs = _binary_primitive_broadcast(lhs, rhs)
    _check_binary_arguments(lhs, rhs)
    return lhs.dtype[lhs.sizes].Multiply(lhs, rhs)

def minimum(lhs, rhs):
    lhs, rhs = _binary_primitive_broadcast(lhs, rhs)
    _check_binary_arguments(lhs, rhs)
    return lhs.dtype[lhs.sizes].Minimum(lhs, rhs)

def maximum(lhs, rhs):
    lhs, rhs = _binary_primitive_broadcast(lhs, rhs)
    _check_binary_arguments(lhs, rhs)
    return lhs.dtype[lhs.sizes].Maximum(lhs, rhs)

def remainder(lhs, rhs):
    lhs, rhs = _binary_primitive_broadcast(lhs, rhs)
    _check_binary_arguments(lhs, rhs)
    return lhs.dtype[lhs.sizes].Remainder(lhs, rhs)


def iota(dtype, shape, dims):
    if isinstance(dims, int):
        dims = [dims]
    for dim in dims:
        assert dim < len(shape), f"Dimension {dim} is larger than tensor rank {len(shape)}"
    return dtype[shape].Iota(dimensions=dims)


def clamp(tensor, minimum=None, maximum=None):
    if minimum is not None:
        min = full_like(tensor, minimum)
        condition = greater_equal(tensor, min)
        tensor = masked_select(condition, tensor, min)

    if maximum is not None:
        maximum = full_like(tensor, maximum)
        condition = less_equal(tensor, maximum)
        tensor = masked_select(condition, tensor, maximum)

    return tensor


def random_uniform(dtype, shape, minimum=0, maximum=1):
    minimum = dtype.Constant(constant_value=minimum)
    maximum = dtype.Constant(constant_value=maximum)
    return dtype[shape].Rng(minimum, maximum, distribution=1) # Uniform distribution


def reshape(tensor, shape):
    # Return a scalar reshape directly
    if shape == []:
        return tensor.dtype.Reshape(tensor)
    if isinstance(shape, int):
        shape = [shape]
    if shape == tensor.sizes:
        return tensor
    if not tensor.sizes: # Handle scalar input special case
        return tensor.dtype[shape].Reshape(tensor)
    dst_numel = functools.reduce(operator.mul, shape)
    src_numel = functools.reduce(operator.mul, tensor.sizes)
    assert dst_numel == src_numel, (
        f"Shape {tensor.sizes} with {src_numel} elements cannot be reshaped to "
        f"{shape} with {dst_numel} elements"
    )
    return tensor.dtype[shape].Reshape(tensor)

def get_hlo_scalar_by_index(tensor, index):
    assert index < tensor.sizes[0]
    return reshape(slice_along(tensor, 0, start=index, limit=index+1), [])

def scatter(operands, scatter_indices, updates, scatter_dims, to_apply):
    operand_rank = len(operands.sizes)
    index_vector_dim = scatter_dims.get("index_vector_dim", [])
    update_window_dims = scatter_dims.get("update_window_dims", [])
    inserted_window_dims = scatter_dims.get("inserted_window_dims", [])
    assert operand_rank == (len(update_window_dims) + len(inserted_window_dims)), \
        "operand.rank must equal the sum of update_window_dims.size and inserted_window_dims.size"
    assert operands.dtype == updates.dtype, "inconsistent dtype between operands and updates"
    scatter_dims_to_operand_dims = scatter_dims.get("scatter_dims_to_operand_dims", [])
    if index_vector_dim == len(scatter_indices.sizes):
        # If index_vector_dim is equal to scatter_indices.rank
        # we implicitly consider scatter_indices to have a trailing 1 dimension.
        scatter_indices_sizes = list(scatter_indices.sizes) + [1]
    else:
        scatter_indices_sizes = list(scatter_indices.sizes)
    assert len(scatter_dims_to_operand_dims) == scatter_indices_sizes[index_vector_dim], \
        "scatter_dims_to_operand_dims.size must be equal to scatter_indices.shape.dims[index_vector_dim]"
    assert len(updates.sizes) == (len(update_window_dims) + len(scatter_indices_sizes) - 1), \
        "Each updates array must be of rank (update_window_dims.size + scatter_indices.rank - 1)"
    dtype = updates.dtype
    updated_sizes = operands.sizes
    assert scatter_indices.sizes[0] == updates.sizes[0], \
        "update window size must match betwen scatter_indices and updates."
    updated = dtype[updated_sizes].Scatter(
        operands, scatter_indices, updates, scatter_dimension_numbers=scatter_dims, to_apply=to_apply)
    return updated


def reduce_scatter(tensor, dim, replica_groups, to_apply, dtype=None):
    size = list(tensor.sizes)
    tensor_dtype = tensor.dtype
    scribe = tensor.scribe

    if dtype is None:
        all_reduce_dtype = tensor_dtype
    elif isinstance(dtype, str):
        all_reduce_dtype = dtypes.to_pyhlo_type(scribe, dtype)
    else:
        all_reduce_dtype = dtype
    tensor = cast(tensor, all_reduce_dtype)
    size[dim] = size[dim]//len(replica_groups[0])
    output = all_reduce_dtype[size].ReduceScatter(tensor,  dimensions = [dim],
                                        replica_groups = replica_groups, to_apply=to_apply)
    return output


def reduce_scatter_sum(tensor, tp_degree, dim, dtype=None, replica_groups=None):

    if tp_degree == 1:
        return tensor

    to_apply = gen_add_func(tensor.dtype)

    if replica_groups is None:
        replica_groups = [list(range(tp_degree))]

    return reduce_scatter(
        tensor,
        dim=dim,
        replica_groups=replica_groups,
        to_apply=to_apply,
        dtype=dtype,
    )


def sin(tensor):
    sizes = tensor.sizes
    dtype = tensor.dtype
    return dtype[sizes].Sin(tensor)


def cos(tensor):
    sizes = tensor.sizes
    dtype = tensor.dtype
    return dtype[sizes].Cos(tensor)


def floor(tensor):
    return tensor.dtype[tensor.sizes].Floor(tensor)


def transpose210(tensor):
    dtype = tensor.dtype
    size0, size1, size2 = tensor.sizes
    return dtype[size2,size1,size0].Transpose(tensor, dimensions=[2, 1, 0])

def transpose102(tensor):
    dtype = tensor.dtype
    size0, size1, size2 = tensor.sizes
    return dtype[size1,size0,size2].Transpose(tensor, dimensions=[1, 0, 2])


# credit: https://github.com/facebookresearch/llama/blob/8992dea3b2c98e82e335efef004534413f4f2d2e/llama/model.py#L164-L173
def repeat_kv(tensor, n_repeats, repeat_dim):
    if n_repeats == 1:
        return tensor
    if repeat_dim == 2:
        n_positions, n_seqs, n_kv_heads, d_head = tensor.sizes
        tensor_br_sizes = n_positions, n_seqs, n_kv_heads, n_repeats, d_head
        tensor_br = broadcast(tensor, out_dim_size=tensor_br_sizes, broadcast_dimensions=[0, 1, 2, 4])
        output = reshape(tensor_br, [n_positions, n_seqs, n_kv_heads * n_repeats, d_head])
    else:
        raise RuntimeError(f"invalid repeat_dim ({repeat_dim})")
    return output


def _is_hlo_scalar(value):
    return hasattr(value, 'sizes') and value.sizes == ()


def is_floating_point(value):
    scribe = value.scribe
    return value.dtype in [
        scribe.f64,
        scribe.f32,
        scribe.f16,
        scribe.bf16,
    ]


def _binary_primitive_broadcast(lhs, rhs):
    lhs_primitive = isinstance(lhs, (int, float, bool))
    rhs_primitive = isinstance(rhs, (int, float, bool))
    lhs_scalar = _is_hlo_scalar(lhs)
    rhs_scalar = _is_hlo_scalar(rhs)

    assert not (lhs_primitive and rhs_primitive), (
        "HLO Operation cannot be performed on two primitives"
    )
    if rhs_primitive:
        rhs = full(rhs, dtype=lhs.dtype, sizes=lhs.sizes)
    if lhs_primitive:
        lhs = full(lhs, dtype=rhs.dtype, sizes=rhs.sizes)
    if rhs_scalar:
        rhs = broadcast(rhs, lhs.sizes, [])
        rhs = cast(rhs, lhs.dtype)
    if lhs_scalar:
        lhs = broadcast(lhs, rhs.sizes, [])
        lhs = cast(lhs, rhs.dtype)

    return lhs, rhs


def _check_binary_arguments(lhs, rhs, dtype=None):
    assert lhs.sizes == rhs.sizes, (
        "Tensor Size Mismatch. "
        f"LHS shape={lhs.sizes} "
        f"RHS shape={rhs.sizes}"
    )
    assert lhs.dtype == rhs.dtype
    if dtype is not None:
        assert lhs.dtype == dtype
        assert rhs.dtype == dtype


def compare(lhs, rhs, direction):
    lhs, rhs = _binary_primitive_broadcast(lhs, rhs)
    _check_binary_arguments(lhs, rhs)
    pred = lhs.scribe.pred
    return pred[lhs.sizes].Compare(lhs, rhs, comparison_direction=direction)


def equal(lhs, rhs):
    return compare(lhs, rhs, 'EQ')


def less(lhs, rhs):
    return compare(lhs, rhs, 'LT')


def less_equal(lhs, rhs):
    return compare(lhs, rhs, 'LE')


def greater(lhs, rhs):
    return compare(lhs, rhs, 'GT')


def greater_equal(lhs, rhs):
    return compare(lhs, rhs, 'GE')


def logical_and(lhs, rhs):
    lhs, rhs = _binary_primitive_broadcast(lhs, rhs)
    pred = lhs.scribe.pred
    _check_binary_arguments(lhs, rhs, dtype=pred)
    return pred[lhs.sizes].And(lhs, rhs)


def logical_or(lhs, rhs):
    lhs, rhs = _binary_primitive_broadcast(lhs, rhs)
    pred = lhs.scribe.pred
    _check_binary_arguments(lhs, rhs, dtype=pred)
    return pred[lhs.sizes].Or(lhs, rhs)


def logical_not(lhs):
    pred = lhs.scribe.pred
    return pred[lhs.sizes].Not(lhs)


def dtype_maximum(dtype):
    scribe = dtype.scribe
    maximums = {
         scribe.s64: 2 ** 63 - 1,
         scribe.s32: 2 ** 31 - 1,
         scribe.s16: 2 ** 15 - 1,
         scribe.s8:  2 ** 7 - 1,
         scribe.u64: 2 ** 64,
         scribe.u32: 2 ** 32,
         scribe.u16: 2 ** 16,
         scribe.u8:  2 ** 8,
         scribe.pred: True,
    }
    return maximums.get(dtype, float('inf'))


def reduce_min(tensor, dim, keepdim=False):

    dtype = tensor.dtype
    reduce_shape = list(tensor.sizes)
    reduce_shape.pop(dim)

    def reducer(scribe):
        p0 = dtype.Parameter(parameter_number=0)
        p1 = dtype.Parameter(parameter_number=1)
        return dtype.Minimum(p0, p1)

    minimum = dtype.Constant(constant_value=dtype_maximum(dtype))
    value = dtype[reduce_shape].Reduce(tensor, minimum, dimensions=[dim], to_apply=reducer)

    if keepdim:
        keepdim_shape = list(tensor.sizes)
        keepdim_shape[dim] = 1
        value = dtype[keepdim_shape].Reshape(value)

    return value


def triangle_mask(dtype, sizes, comparison='GE'):
    assert len(sizes) == 2, (
        f"Expected rank 2 triangle mask size but found {sizes}"
    )
    pred = dtype.scribe.pred
    s32 = dtype.scribe.s32
    a = s32[sizes].Iota(dimensions=[0])
    b = s32[sizes].Iota(dimensions=[1])
    result = pred[sizes].Compare(a, b, comparison_direction=comparison)
    result = cast(result, dtype)
    return result


def triu_mask(dtype, sizes):
    return triangle_mask(dtype, sizes, 'LE')


def tril_mask(dtype, sizes):
    return triangle_mask(dtype, sizes, 'GE')


def decoder_attention_mask_window(cache_ids, start_ids, n_positions):
    """
    Creates decomposed prior/active masks for windowed & speculative attention.

    This mask should only be used when computing a number of context tokens
    in parallel (>1) which may also require looking at the previously computed
    KV-cache tokens.

    Example:

        # In this example consider a batch of size 1 that is left-padded
        input_ids = [
            [-, a, b, c, d, e],
        ]
        start_ids = [1]

    Window 1:

        # Consider a window computing the left-most tokens in the sequence
        cache_ids = [0, 1, 2]

        # The prior tokens can be ignored since the KV-cache should be empty.
        prior_mask = [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
        # The active token mask is empty at position 0 due to left padding.
        active_mask = [
            [0, 0, 0],
            [0, 1, 0],
            [0, 1, 1],
        ]

    Window 2:

        # In the next window consider computing an arbitrary window within
        # the sequence
        cache_ids = [3, 4, 5]

        # Next the prior mask at is empty at position 0 due to left padding.
        # Unlike the prior iteration, we begin attending to KV-cache
        # projections since we expect they have been populated.
        prior_mask = [
            [0, 1, 1, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 1, 1, 0, 0],
        ]

        # For the current window of tokens, we generate a full triangular mask
        # since we no longer have any padding tokens to consider.
        active_mask = [
            [1, 0, 0],
            [1, 1, 0],
            [1, 1, 1],
        ]

    Arguments:
        cache_ids: The positions to update in the cache.
        start_ids: The padding offset from the left side.
        n_positions: The total size of the KV cache to consider. This is
            equal to the size of the bucket.

    Returns:
        prior_mask: The attention mask to apply to the KV cache
        active_mask: The attention mask to apply to the active tokens.
    """
    scribe = cache_ids.scribe
    s32 = scribe.s32
    pred = scribe.pred
    batch_size, = start_ids.sizes
    n_active_tokens, = cache_ids.sizes

    cache_ids = cast(cache_ids, s32)
    start_ids = cast(start_ids, s32)

    # Compute decomposed mask for the prior tokens
    positions = s32[n_positions].Iota(dimensions=[0])
    size = batch_size, n_positions
    positions = broadcast(positions, size, [1])
    starts = broadcast(start_ids, size, [0])
    start_mask = greater_equal(positions, starts)
    minimum_id = reduce_min(cache_ids, 0)
    minimum_id = broadcast(minimum_id, size, [])
    end_mask = less(positions, minimum_id)
    prior_mask = logical_and(start_mask, end_mask)
    # Final broadcast ensures we use correct path in layers/attention.py:mask
    size = (batch_size, n_active_tokens, n_positions)
    prior_mask = broadcast(prior_mask, size, [0, 2])

    # Compute a mask for the active tokens
    size = batch_size, n_active_tokens, n_active_tokens
    cache_br = broadcast(cache_ids, size, [2])
    start_br = broadcast(start_ids, size, [0])
    start_mask = greater_equal(cache_br, start_br)
    causal_mask = tril_mask(pred, (n_active_tokens, n_active_tokens))
    causal_mask = broadcast(causal_mask, size, [1, 2])
    active_mask = logical_and(causal_mask, start_mask)

    return prior_mask, active_mask


def decoder_attention_mask_lhs_aligned(cache_ids, n_positions, last_token_id=None, num_active_blocks=None, neuron_config=None):
    """
    Create attention masks for LHS-aligned sequences.

    Args:
        cache_ids: The 2d positions to update in the cache.
        n_positions: The total size of the KV cache to consider. This is
            equal to the current bucket size.

    Returns:
        prior_mask: The attention mask to apply to the KV cache.
        active_mask: The attention mask to apply to the active tokens.
    """
    batch_size, n_active_tokens = cache_ids.sizes
    if n_active_tokens == n_positions:
        # Context Encoding
        if neuron_config and neuron_config.use_1d_query:
            # For concatenated prompt encoding (1D query), last_token_id is used as prompt_lens
            return decoder_attention_block_diagonal_causal_mask(last_token_id, n_positions)
        else:
            return decoder_attention_mask_lhs_aligned_context(cache_ids, n_positions)
    else:
        # Token generation
        return decoder_attention_mask_lhs_aligned_token(
            cache_ids, n_positions, num_active_blocks=num_active_blocks, neuron_config=neuron_config)


def decoder_attention_mask_lhs_aligned_context(cache_ids, n_positions):
    """
    Creates a lower triangular mask for LHS-aligned context encoding.

    This mask is static and does not depend on the inputs. During LHS-aligned
    context encoding, there is a guarantee that each token in a sequence must
    attend to all prior positions. This is unlike RHS-aligned sequences where
    batch padding may require that an earlier position must not be attended to.

    Example:

        # A context where the size is equal to the bucket width
        n_positions = 3

        prior_mask = [
            [1, 0, 0], # At position 0 attend to self only
            [1, 1, 0], # At position 1 attend to self and prior
            [1, 1, 1], # At position 2 attend to self and all prior
        ]
        active_mask = None

    Args:
        cache_ids: The 2d positions to update in the cache.
        n_positions: The total size of the KV cache to consider. This is
            equal to the current bucket size.

    Returns:
        prior_mask: The attention mask to apply to the KV cache.
        active_mask: The attention mask to apply to the active tokens (None).
    """
    dtype = cache_ids.scribe.pred
    batch_size, _ = cache_ids.sizes
    prior_mask = tril_mask(dtype, (n_positions, n_positions))
    # Final broadcast ensures we use correct path in layers/attention.py:mask
    size = batch_size, n_positions, n_positions
    prior_mask = broadcast(prior_mask, size, [1, 2])

    active_mask = None
    return prior_mask, active_mask


def decoder_attention_block_diagonal_causal_mask(prompt_lens, n_positions):
    """
    Creates block diagonal causal masks for multiple prompts.

    This mask is dynamic and depends on the input prompt lengths.

    Example:
        prompt_lens = [2, 3, 2, 0], n_positions = 7

        prior_mask = [
            [1, 0, 0, 0, 0, 0, 0], # At position 0 attend to 1st prompt
            [1, 1, 0, 0, 0, 0, 0], # At position 1 attend to 1st prompt
            [0, 0, 1, 0, 0, 0, 0], # At position 0 attend to 2nd prompt
            [0, 0, 1, 1, 0, 0, 0], # At position 1 attend to 2nd prompt
            [0, 0, 1, 1, 1, 0, 0], # At position 2 attend to 2nd prompt
            [0, 0, 0, 0, 0, 1, 0], # At position 0 attend to 3rd prompt
            [0, 0, 0, 0, 0, 1, 1], # At position 1 attend to 3rd prompt
        ]
        active_mask = None

    Args:
        prompt_lens: The list of prompt lengths.
        n_positions: The total size of the KV cache to consider. This is
            equal to the current bucket size.

    Returns:
        prior_mask: The attention mask to apply to the KV cache.
        active_mask: The attention mask to apply to the active tokens (None).
    """
    s32 = prompt_lens.scribe.s32
    sizes = n_positions, n_positions
    num_prompts, = prompt_lens.sizes

    a = iota(s32, sizes, [0])
    b = iota(s32, sizes, [1])
    prior_mask = compare(a, b, "GE")

    anchors = cumsum(prompt_lens, dim=0)

    for idx in range(num_prompts):
        zero = s32.Constant(constant_value=idx)
        value = dynamic_slice_along(anchors, dim=0, start=zero, size=1)

        # mask = jnp.logical_and(b<c, a>=c)
        c = broadcast(value, sizes, [0])
        c = cast(c, s32)
        bc = compare(b, c, "LT")
        ac = compare(a, c, "GE")
        mask = logical_and(bc, ac)

        # prior_mask = jnp.logical_and(prior_mask, jnp.logical_not(mask))
        prior_mask = logical_and(prior_mask, logical_not(mask))

    # [n_positions, n_positions] -> [1, n_positions, n_positions]
    prior_mask = unsqueeze(prior_mask, dim=0)

    active_mask = None
    return prior_mask, active_mask


def decoder_attention_mask_lhs_aligned_token(cache_ids, n_positions, num_active_blocks=None, neuron_config=None):
    if neuron_config and neuron_config.optimized_paged_attention:
        block_size = neuron_config.continuous_batching.block_size
        assert isinstance(num_active_blocks, int) and (num_active_blocks is not None), \
            f"num_active_blocks is expected to be an int, but got {num_active_blocks}"
        return decoder_attention_mask_lhs_aligned_token_blockwise(cache_ids, block_size=block_size, num_blocks=num_active_blocks)
    else:
        return decoder_attention_mask_lhs_aligned_token_padded(cache_ids, n_positions)


def decoder_attention_mask_lhs_aligned_token_blockwise(context_lens, block_size, num_blocks):
    """
    Creates the block-wise attention mask for decoding with multiple KV cache blocks.

    Example:

    # INPUTS:
    context_lens: tensor([ 3, 10, 5, 0])
    block_size: 4
    num_blocks: 8

    # EXPECTED OUTPUT:
    # attention mask with 8 blocks total (w/ block_size=4, total elements=32)
    # 24 active blocks are padded to 32 blocks
    attention_mask
    [
    # seq0 - 3 elements (pad to 4)
    1,1,1,0,

    # seq1 - 10 elements (pad to 12)
    1,1,1,1,
    1,1,1,1,
    1,1,0,0,

    # seq2 - 5 elements (pad to 8)
    1,1,1,1,
    1,0,0,0,

    # pad 8 elements to 32 total elements
    0,0,0,0,
    0,0,0,0
    ]

    Algorithm for implementing block-wise attention mask:

    >>> # plan the output space for the active blocks
    >>> blocks_cumsum = lax.cumsum((context_lens+block_size-1)//block_size, axis=0)
    >>>
    >>> prior_mask = jnp.zeros(num_tokens, dtype=context_lens.dtype)
    >>> mask_iota = jnp.arange(start=0, stop=num_tokens, dtype=context_lens.dtype)
    >>> for seq_id in range(max_num_seqs):
    >>>     start_idx = 0 if seq_id == 0 else blocks_cumsum[seq_id-1] * block_size
    >>>     end_idx = start_idx + context_lens[seq_id]
    >>>     mask = jnp.int32(jnp.logical_and(mask_iota >= start_idx, mask_iota < end_idx))
    >>>     prior_mask = prior_mask + mask
    """
    assert len(context_lens.sizes) == 2, "context_lens is expected to be a 2D vector."

    s32 = context_lens.scribe.s32
    f32 = context_lens.scribe.f32
    pred = context_lens.scribe.pred
    max_num_seqs = context_lens.sizes[0]
    num_tokens = block_size * num_blocks

    # reshape from 2D to 1D vector
    context_lens = reshape(context_lens, (max_num_seqs,))

    # blocks_cumsum = cumsum((context_lens+block_size-1)//block_size, axis=0)
    blocks_add = add(context_lens, block_size-1)
    blocks_div = cast(floor(divide(cast(blocks_add, f32), block_size)), s32)
    blocks_cumsum = cumsum(blocks_div, dim=0)

    blocks_mul = multiply(blocks_cumsum, block_size)
    prior_mask = full(0, dtype=pred, sizes=(num_tokens,))
    mask_iota = iota(s32, (num_tokens,), [0])
    for seq_id in range(max_num_seqs):
        zero, prev_seq_id, curr_seq_id = s32.Constant(constant_value=0), s32.Constant(constant_value=seq_id-1), s32.Constant(constant_value=seq_id)
        start_idx = zero if seq_id == 0 else dynamic_slice_along(blocks_mul, dim=0, start=prev_seq_id, size=1)
        br_dims = [] if seq_id == 0 else [0]
        start_idx_br = broadcast(start_idx, out_dim_size=(num_tokens,), broadcast_dimensions=br_dims)
        end_idx = add(dynamic_slice_along(context_lens, dim=0, start=curr_seq_id, size=1), start_idx)
        end_idx_br = broadcast(end_idx, out_dim_size=(num_tokens,), broadcast_dimensions=[0])
        mask = logical_and(greater_equal(mask_iota, start_idx_br), less(mask_iota, end_idx_br))
        # start_idx is loaded from cumsum of block placement location, therefore the mask update location should never overlap.
        prior_mask = add(prior_mask, cast(mask, pred))
    prior_mask = reshape(prior_mask, (num_blocks, 1, block_size))
    active_mask = full(1, dtype=pred, sizes=(max_num_seqs,1,1))
    return prior_mask, active_mask


def decoder_attention_mask_lhs_aligned_token_padded(cache_ids, n_positions):
    """
    Creates decomposed prior/active masks for LHS-aligned token generation.

    Unlike LHS-aligned context encoding, this mask cannot be a completely
    static because each batch line may need to attend to a different number
    of prior tokens depending on the current token(s) being computed.

    This function assumes that `cache_ids` are linearly increasing per batch
    line when `n_active_tokens > 1` (speculative & windowed attention).

    Example: Single Token Generation

        n_positions = 4
        cache_ids = [
            [2]
        ]

        # Attend to all prior positions from the current token (2)
        prior_mask = [
            [1, 1, 0, 0]
        ]
        # Always attend to the current token
        active_mask = [
            [1]
        ]

    Example: Batched Execution

        n_positions = 4
        cache_ids = [
            [3] # Batch 0
            [1] # Batch 1
        ]

        # Attend to all prior positions on each batch line
        prior_mask = [
            [1, 1, 1, 0] # Batch 0
            [1, 0, 0, 0] # Batch 1
        ]
        # Always attend to the current token on each batch line
        active_mask = [
            [1], # Batch 0
            [1], # Batch 1
        ]

    Example: Speculative Sampling

        n_positions = 4
        cache_ids = [
            [2, 3] # Batch 0
            [1, 2] # Batch 1
        ]

        # Attend to all prior positions on each batch line
        prior_mask = [
            [1, 1, 0, 0] # Batch 0
            [1, 0, 0, 0] # Batch 1
        ]
        # Use lower triangular mask for each active set of tokens
        active_mask = [
            [1, 0],
            [1, 1]
        ]

    Args:
        cache_ids: The 2d positions to update in the cache.
        n_positions: The total size of the KV cache to consider. This is
            equal to the current bucket size.

    Returns:
        prior_mask: The attention mask to apply to the KV cache
        active_mask: The attention mask to apply to the active tokens.
    """
    pred = cache_ids.scribe.pred
    dtype = cache_ids.dtype
    batch_size, n_active_tokens = cache_ids.sizes

    # Prior mask
    if n_active_tokens > 1:
        # Multi-token speculative sampling & windowed attention
        cache_ids = reduce_min(cache_ids, dim=1, keepdim=True)
    size = (batch_size, n_active_tokens, n_positions)
    positions = dtype[size].Iota(dimensions=[2])
    cache_ids = broadcast(cache_ids, size, [0, 1])
    prior_mask = greater(cache_ids, positions)

    # Active mask
    if n_active_tokens == 1:
        # Single token (Always pay attention to self)
        active_mask = full(1, pred, (batch_size, n_active_tokens))
    else:
        # Multi-token speculative sampling & windowed attention
        causal_mask = tril_mask(pred, (n_active_tokens, n_active_tokens))
        size = (batch_size, n_active_tokens, n_active_tokens)
        active_mask = broadcast(causal_mask, size, [1, 2])

    return prior_mask, active_mask


def exp(tensor):
    dtype = tensor.dtype
    return dtype[tensor.sizes].Exp(tensor)


def sqrt(tensor):
    dtype = tensor.dtype
    return dtype[tensor.sizes].Sqrt(tensor)


def rsqrt(tensor):
    dtype = tensor.dtype
    return dtype[tensor.sizes].Rsqrt(tensor)


def get_tuple_element(tup, tuple_index):
    element = tup.get_tuple_element(tuple_index)
    size = element.sizes
    dtype = element.dtype
    return dtype[size].GetTupleElement(tup, tuple_index=tuple_index)


def log_softmax(scores, tp_degree=1, dim=None):
    rank = len(scores.sizes)
    if dim is None:
        dim = rank - 1
    dtype = scores.dtype
    br_dims = [di for di in range(rank) if di != dim]
    reduce_sizes = [scores.sizes[di] for di in br_dims]
    const_min = dtype.Constant(constant_value=float('-inf'))
    max_func = gen_max_func(dtype)
    reductions = dtype[reduce_sizes].Reduce(scores, const_min, dimensions=[dim], to_apply=max_func)
    global_reductions = all_gather(reductions, dim, tp_degree)
    global_max = reduce_max(global_reductions, dim, keepdim=True)
    br_reduce_max = broadcast(global_max, scores.sizes, broadcast_dimensions=br_dims)
    sub = subtract(scores, br_reduce_max)
    exp_score = exp(sub)
    exp_score = all_gather(exp_score, dim, tp_degree=tp_degree)
    exp_score_max = reduce_sum(exp_score, dim)
    log = dtype[exp_score_max.sizes].Log(exp_score_max)
    br_log = broadcast(log, scores.sizes, broadcast_dimensions=br_dims)
    out = subtract(scores, br_log)
    return out


# https://www.tensorflow.org/xla/operation_semantics#select
def masked_select(mask, true_tensor, false_tensor):
    true_tensor, false_tensor = _binary_primitive_broadcast(true_tensor, false_tensor)
    dtype = true_tensor.dtype
    assert mask.dtype == mask.scribe.pred, "Mask must be a boolean tensor."
    assert dtype == false_tensor.dtype
    assert mask.sizes == true_tensor.sizes == false_tensor.sizes, (
        "Tensor size mismatch."
        f"mask shape={len(mask.sizes)}, true_tensor shape={len(true_tensor.sizes)}, false_tensor shape={len(false_tensor.sizes)}"
    )
    return dtype[mask.sizes].Select(mask, true_tensor, false_tensor)


def reshape_and_cache(key, value, key_cache, value_cache, slot_mapping):
    """
    Fused K/V cache placement with slot_mapping.

    Example:
        prompt_lens = [3, 6, 2], num_blocks = 4, block_size = 128,

        #              |<-- seq 0 -->|<--------   seq 1 ---------->|  seq 2  |
        slot_mapping = [128, 129, 130, 256, 257, 258, 259, 260, 261, 384, 385]

        updated_keys = [
            # |<---- block_size (128) ------>|
            [................................] # empty slot
            [128, 129, 130, .................] # 3 tokens taking 2nd block
            [256, 257, 258, 259, 260, 261 ...] # 6 tokens taking 3rd block
            [384, 385 .......................] # 2 tokens taking 4th block
        ]
    """
    dtype = key.dtype
    assign_func = gen_assign_func(dtype)
    _, n_active_tokens, n_head, d_head = key.sizes
    n_blocks, block_size, _, _ = key_cache.sizes
    hidden_size = n_head * d_head

    key = reshape(key, [n_active_tokens, hidden_size])
    value = reshape(value, [n_active_tokens, hidden_size])
    key_cache = reshape(key_cache, [n_blocks*block_size, hidden_size])
    value_cache = reshape(value_cache, [n_blocks*block_size, hidden_size])

    scatter_dims = dict(update_window_dims=[1],
                        inserted_window_dims=[0],
                        scatter_dims_to_operand_dims=[0],
                        index_vector_dim=1)
    updated_keys = scatter(key_cache, slot_mapping, key, scatter_dims=scatter_dims, to_apply=assign_func)
    updated_values = scatter(value_cache, slot_mapping, value, scatter_dims=scatter_dims, to_apply=assign_func)

    updated_keys = reshape(updated_keys, [n_blocks, block_size, n_head, d_head])
    updated_values = reshape(updated_values, [n_blocks, block_size, n_head, d_head])

    return updated_keys, updated_values


def speculative_mask(
    draft_ids,           # shape: (k, batch_size)
    draft_scores,        # shape: (vocab_size_tp, k, batch_size)
    target_scores,       # shape: (vocab_size_tp, k + 1, batch_size)
    tp_degree=1,
    deterministic=False,
):
    s32 = draft_ids.scribe.s32

    # Convert score into probabilities
    draft_probabilities = softmax(draft_scores, dim=0, tp_degree=tp_degree)
    target_probabilities = softmax(target_scores, dim=0, tp_degree=tp_degree)

    # For simplicity all-gather the scores
    if tp_degree > 1:
        draft_probabilities = all_gather(draft_probabilities, dim=0, tp_degree=tp_degree)
        target_probabilities = all_gather(target_probabilities, dim=0, tp_degree=tp_degree)

    vocab_size, k, batch_size = draft_scores.sizes

    # Gather the target/draft probabilities corresponding to draft sampling predictions
    index = reshape(draft_ids, (1, k, batch_size))
    target_probs = slice_along(target_probabilities, 1, limit=k)
    target_probs = gather(target_probs, 0, index)
    draft_probs = gather(draft_probabilities, 0, index)
    target_probs = squeeze(target_probs, 0)              # shape: (k, batch_size)
    draft_probs = squeeze(draft_probs, 0)                # shape: (k, batch_size)

    # Compare ratio of probabilities at locations to a random sample
    ratio = divide(target_probs, draft_probs) # shape: (k, batch_size)
    ratio = clamp(ratio, maximum=1.0)
    if deterministic:
        random = full_like(ratio, 0.5)
    else:
        random = random_uniform(ratio.dtype, ratio.sizes)
    accepted_mask = less(random, ratio) # shape: (k, batch_size)

    # Mask out all tokens past the accepted token
    accepted_mask = cast(accepted_mask, s32)
    accepted_cumsum = cumsum(accepted_mask, dim=0)
    positions = iota(s32, (k, batch_size), 0)
    positions = add(positions, 1)
    accepted_mask = equal(accepted_cumsum, positions) # shape: (k, batch_size)

    # Compute index of *final* accepted token per batch line
    accepted_mask = pad(accepted_mask, dim=0, size=1, value=False)

    return accepted_mask # (k + 1, batch_size)


def speculative_token_selection(
        draft_ids,           # shape: (k, batch_size)
        target_ids,          # shape: (k + 1, batch_size)
        draft_scores,        # shape: (vocab_size_tp, k, batch_size)
        target_scores,       # shape: (vocab_size_tp, k + 1, batch_size)
        tp_degree=1,
        pad_token_id=0,
        deterministic=False,
        output_mask=False,
    ):
    """
    A speculative token acceptor based on original DeepMind paper.

    Reference: https://arxiv.org/pdf/2302.01318.pdf

    Note that this function does not perform initial sampling and assumes that
    both target/draft token generation has been performed prior to executing
    this function.

    Args:
        draft_ids: The sampled draft model tokens.
        target_ids: The sampled target model tokens.
        draft_scores: The raw draft model logit output.
        target_scores: The raw target model logits output.
        tp_degree: The number of ranks to collect across.
        deterministic: Flag which enables using a constant 0.5 as token
            acceptance threshold instead of a random uniform. This is used for
            debug/testing.
        output_mask: Flag that enables returning the token selection mask. This
            is useful for downstream tasks such as retrieving the accepted
            scores in a speculative loop.

    Returns:
        token: The accepted tokens (with 0-padding)
        index: The last index accepted for each batch line.
        mask: (optional) The token selection mask.
    """
    s32 = draft_ids.scribe.s32

    accepted_mask = speculative_mask(
        draft_ids,           # shape: (k, batch_size)
        draft_scores,        # shape: (vocab_size_tp, k, batch_size)
        target_scores,       # shape: (vocab_size_tp, k + 1, batch_size)
        tp_degree=tp_degree,
        deterministic=deterministic,
    )

    # Select the final tokens
    # Note:
    # - When all rejected: Mask will select target in all positions (position 0 required)
    # - When all accepted: Mask will select draft in all positons but the last
    draft_ids = pad(draft_ids, dim=0, size=1, value=0)
    tokens = masked_select(accepted_mask, draft_ids, target_ids)

    # Zero-out tokens beyond the used tokens
    positions = iota(s32, tokens.sizes, 0)
    index = reduce_sum(cast(accepted_mask, s32), dim=0)
    mask = greater_equal(broadcast(index, tokens.sizes, [1]), positions)
    tokens = masked_select(mask, tokens, pad_token_id)

    # Transpose back to SB -> BS layout
    tokens = transpose(tokens, 0, 1)

    if output_mask:
        # Transpose back to SB -> BS layout
        mask = transpose(mask, 0, 1)
        return (
            tokens, # shape: (batch_size, k + 1)
            index,  # shape: (batch_size,)
            mask,   # shape: (batch_size, k + 1)
        )
    return (
        tokens,  # shape: (batch_size, k + 1)
        index,   # shape: (batch_size,)
    )


def speculative_combine(
    draft_ids,           # shape: (k, batch_size)
    target_ids,          # shape: (1, batch_size)
    accepted_mask,       # shape: (k + 1, batch_size)
    pad_token_id=0,
):
    scribe = draft_ids.scribe
    s32 = scribe.s32

    # Pad the draft ids
    draft_ids = pad(draft_ids, dim=0, size=1, value=0)

    # Pad the target ids
    target_ids = broadcast(target_ids, draft_ids.sizes, [0, 1])

    # Select the correct ids according to the mask
    tokens = masked_select(accepted_mask, draft_ids, target_ids)

    # Zero-out tokens beyond the used tokens
    index = reduce_sum(cast(accepted_mask, s32), dim=0)
    positions = iota(s32, tokens.sizes, 0)
    mask = greater_equal(broadcast(index, tokens.sizes, [1]), positions)
    tokens = masked_select(mask, tokens, pad_token_id)

    # Transpose back to SB -> BS layout
    tokens = transpose(tokens, 0, 1)
    return tokens, index


def flip(tensor, dims):
    if isinstance(dims, int):
        dims = [dims]
    return tensor.dtype[tensor.sizes].Reverse(tensor, dimensions=dims)


def diff(tensor, dim):
    """
    Computes the n-th forward difference along the given dimension, where n=1.

    Example:
    >>> c = tensor([[1, 2, 3], [3, 4, 5]])
    >>> diff(c, dim=0)
    tensor([[2, 2, 2]])
    >>> diff(c, dim=1)
    tensor([[1, 1],
            [1, 1]])
    """
    o_sizes = tensor.sizes
    a = slice_along(tensor, dim, limit=o_sizes[dim]-1, start=0)
    b = slice_along(tensor, dim, limit=o_sizes[dim], start=1)
    output = subtract(b, a)
    return output


def get_window_indices_from_cache_ids(cache_ids, window_size, n_positions):
    """
    Get the sliding-window indices from cache_ids
    This funciton is used in token generation in sliding-window attention for mistral-7b-v0.1
    Given a cache_id, the indices would be [cache_id-(w-1), cache_id-(w-2), ..., cache_id-1], w is the window size
    When cache_id < w-1, the indices would be [0, 1, ..., w-2], we rely on the attention mask to remove the cache with indices of [cache_id, ..., w-1]
    Arguments:
        cache_ids: shape (b, s), b is batch size, s is the number of active tokens, s = 1 in token generation
        window_size: window size in sliding-window attention
        n_positions: sequence length of k/v cache
    Return:
        indices: indices for element in k/v cache corresponding to the sliding window, shape (b, w-1)
    Example:
        Denote n as n_positions and w as window_size
        cache_ids = [[5], [15], [35]], batch size is 3
        window_size = 8
        indices = [[0, 1, ..., 6],
                    [8+n, 9+n, ..., 14+n],
                    [28+2n, 29+2n, ..., 34+2n]]
    """
    dtype = cache_ids.dtype
    b, s = cache_ids.sizes # s = 1 for token generation
    sizes = (b, window_size-1)

    # generate matrix of shape (b, w-1)
    #  [[0, 0, ..., 0],
    #   [n, n, ..., n],
    #   [2n, 2n, ..., 2n]]
    offset = iota(dtype, sizes, [0])
    offset = multiply(offset, n_positions)

    # cacluate win_start from cache_ids and then broadcast to shape (b, w)
    # cache_ids = [[5], [15], [35]]
    # window_size = 8
    # win_start = [[0], [8], [28]]
    # after broadcast
    #  [[0, 0, ..., 0],
    #   [8, 8, ..., 8],
    #   [28, 28, ..., 28]]
    win_start = subtract(cache_ids, window_size-1)
    win_start = maximum(win_start, 0)
    win_start = broadcast(win_start, out_dim_size=sizes, broadcast_dimensions=[0, 1])

    # calculate the indices of shape (b, w)
    # win_indices = win_start + offset
    #  [[0, 1, ..., 6],
    #   [8+n, 9+n, ..., 14+n],
    #   [28+2n, 28+2n, ..., 34+2n]]
    win_indices = iota(dtype, sizes, [1])
    win_indices = add(win_start, win_indices)
    win_indices = add(win_indices, offset)

    return win_indices


def window_slice_kv(cache, indices):
    """
    Apply sliding-window slicing over k/v cache with provided indices
    This funciton is used in token generation in sliding-window attention for mistral-7b-v0.1
    k/v cache layout shall be (b, s, k, h), b: batch size, s: sequence length, k: num of kv heads per tp, h: hidden size
    For k/v cache of (s, b, k, h) layout, cache needs to be transposed to (b, s, k, h) before calling this function
    and transposed back to (s, b, k, h) afterwards
    Arguments:
        cache: key/value cache, shape (b, s, k, h)
        indices: (b, w)
    Return:
        result: sliced k/v cache, shape (b, w-1, k, h)
    """
    sizes = cache.sizes # (b, s, k, h)
    ind_sizes = indices.sizes # (b, w-1)

    # (b, s, k, h) -> (b*s, k, h)
    cache = reshape(cache, (sizes[0]*sizes[1], sizes[2], sizes[3]))

    # (b, w-1) -> (b*(w-1),)
    indices = reshape(indices, (ind_sizes[0]*ind_sizes[1]))

    # (b*s, k, h) -> (b*(w-1), k, h)
    result = index_select(cache, 0, indices)

    # (b*(w-1), k, h) -> (b, w-1, k, h)
    result = reshape(result, (ind_sizes[0], ind_sizes[1], sizes[2], sizes[3]))

    return result


def window_slice_mask(mask, indices):
    """
    Apply sliding-window slice over mask with provided indices
    This funciton is used in token generation in sliding-window attention for mistral-7b-v0.1
    mask layout shall be (b, n_active_tokens, s), n_active_tokens=1 for token generation
    Arguments:
        mask: (b, 1, s)
        indices: (b, w)
    Return:
        result: sliced mask, shape (b, 1, w-1)
    """
    sizes = mask.sizes # (b, 1, s)
    ind_sizes = indices.sizes # (b, w-1)

    assert sizes[1] == 1, f"window_slice_mask works only for mask dim (b, 1, s), but mask dim[1]: {sizes[1]} != 1"

    # (b, 1, s) -> (b*s,)
    mask = reshape(mask, (sizes[0]*sizes[2]))

    # (b, w-1) -> (b*(w-1),)
    indices = reshape(indices, (ind_sizes[0]*ind_sizes[1]))

    # (b*s,) -> (b*(w-1),)
    result = index_select(mask, 0, indices)

    # (b*(w-1),) -> (b, 1, w-1)
    result = reshape(result, (ind_sizes[0], 1, ind_sizes[1]))

    return result


def sliding_window_slice(cached_keys, cached_values, mask, cache_ids, window_size, n_positions, cache_layout, lhs_aligned):
        # slice the portion of kv cache and mask inside the sliding window
        # this function is used in token generation only
        # for continuous batcing, cache_ids is 2d, shape (b, s), s=1 for token gen
        # for static batching, cache_ids is 1d, shape (s,), s=1 for token gen
        use_2d_cache_ids = len(cache_ids.sizes) > 1

        # continuous batching
        if use_2d_cache_ids:
            if cache_layout == constants.LAYOUT_BSH:
                batch_size = list(cached_keys.sizes)[0]
                seq_len = list(cached_keys.sizes)[1]
                assert list(cached_keys.sizes)[1] == list(cached_values.sizes)[1] == list(mask.sizes)[2], \
                    f"Sequence length not equal for K/V cache and mask"
            elif cache_layout == constants.LAYOUT_SBH:
                batch_size = list(cached_keys.sizes)[1]
                seq_len = list(cached_keys.sizes)[0]
                assert list(cached_keys.sizes)[0] == list(cached_values.sizes)[0] == list(mask.sizes)[2], \
                    f"Sequence length not equal for K/V cache and mask"
            else:
                raise RuntimeError(f"Unsupported cache layout: {cache_layout}")

            if seq_len > window_size:
                indices = get_window_indices_from_cache_ids(cache_ids, window_size, n_positions)
                useful_cached_keys = window_slice_kv(cached_keys, indices)
                useful_cached_values = window_slice_kv(cached_values, indices)
                useful_mask = window_slice_mask(mask, indices)
            else:
                useful_cached_keys, useful_cached_values, useful_mask = cached_keys, cached_values, mask
        # static batching
        else:
            # slice when seq len > window size
            # cache layout is (s, b, k, h), k is the num of kv heads per tp
            curr_window_start = subtract(cache_ids, window_size - 1)
            curr_window_start = maximum(curr_window_start, 0)
            assert list(cached_keys.sizes)[0] == list(cached_values.sizes)[0] == list(mask.sizes)[2], \
                f"Sequence length not equal for K/V cache and mask"
            seq_len = list(cached_keys.sizes)[0]
            if seq_len > window_size:
                batch_size = list(cached_keys.sizes)[1]
                assert (lhs_aligned and batch_size == 1) or (not lhs_aligned), \
                    f"Sliding-window attention under static batching with lhs alignment only works for batch size 1, but get {batch_size}"

                curr_window_start_scalar = reshape(curr_window_start, [])
                useful_cached_keys = dynamic_slice_along(cached_keys, dim=0, start=curr_window_start_scalar, size=window_size-1)
                useful_cached_values = dynamic_slice_along(cached_values, dim=0, start=curr_window_start_scalar, size=window_size-1)
                useful_mask = dynamic_slice_along(mask, dim=2, start=curr_window_start_scalar, size=window_size-1)
            else:
                useful_cached_keys, useful_cached_values, useful_mask = cached_keys, cached_values, mask

        return useful_cached_keys, useful_cached_values, useful_mask
