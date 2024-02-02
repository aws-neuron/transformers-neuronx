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
from typing import Optional, List
import torch
from transformers_neuronx.config import QuantizationConfig


def maybe_quantize_weights(
    tensor: torch.Tensor,
    quantize_config: QuantizationConfig,
    out_feature_dim: Optional[int] = 1,
    contract_dims: Optional[List] = None,
):
    """
    Quantize tensors using the dtype and method specified in quantize_config.

    Arguments:
        tensor: The PyTorch tensor that will be quantized.
        quantize_config: Config that specifies the quantization dtype and method.
        out_feature_dim: Output feature dimension for the matrix when it's multiplied.
        contract_dims: Contraction dimension(s) for the tensor when it's multiplied.

    Returns:
        quantized_weights: The quantized tensor.
        scales: Scales to "rescale" the quantized tensor after it's multiplied,
            where W_f = W_q * scales.
    """

    if tensor is None:
        return None, None

    tensor_rank = len(list(tensor.shape))

    if contract_dims is None:
        assert (
            tensor_rank == 2
        ), f"Contract dimensions must be specified for {tensor_rank}-dimensional tensors."
        # Preserve original quantization API behavior
        contract_dims = [0] if out_feature_dim == 1 else [1]
    else:
        assert tensor_rank - len(contract_dims) == 1, (
            "Quantization is only supported when the number of contract "
            "dimensions is 1 less than the number of tensor dimensions. Number "
            f"of tensor dimensions: {tensor_rank}. Number of provided contract "
            f"dimensions: {len(contract_dims)}."
        )

    assert out_feature_dim not in contract_dims, (
        f"out_feature_dim ({out_feature_dim}) should not be included in "
        f"contract_dims ({contract_dims})"
    )

    tensor = tensor.to(torch.float32)

    if quantize_config.quantize_method == "vector_dynamic":
        if quantize_config.quant_dtype == "s8":
            # Use W_f = W_q * scales
            int8_min = torch.iinfo(torch.int8).min
            int8_max = torch.iinfo(torch.int8).max
            max_values = torch.amax(torch.abs(tensor), dim=contract_dims, keepdim=True)
            scales = max_values / int8_max
            if scales.count_nonzero() == 0:
                # If the scales is all zeros, the weight tensor are zeros
                quantized_weights = tensor.to(torch.int8)
            else:
                quantized_weights = tensor / scales
                quantized_weights = torch.round(quantized_weights)
                quantized_weights = torch.clamp(quantized_weights, int8_min, int8_max)
                quantized_weights = quantized_weights.to(torch.int8)
                scales = scales.flatten()
        else:
            raise NotImplementedError(
                f"{quantize_config.quant_dtype} for {quantize_config.quantize_method} is not yet implemented."
            )
    else:
        raise NotImplementedError(
            f"{quantize_config.quantize_method} is not yet implemented."
        )
    return quantized_weights, scales
