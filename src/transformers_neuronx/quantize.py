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
import torch
from transformers_neuronx.config import QuantizationConfig

def quantize_weights(tensor, quantize_config: QuantizationConfig, weight_transposed=True):
    tensor = tensor.to(torch.float32)
    if quantize_config.quantize_method == 'vector_dynamic':
        reduce_dim = 0 if weight_transposed else 1
        if quantize_config.quant_dtype == "s8":
             # Use W_f = W_q * scales
            int8_min = torch.iinfo(torch.int8).min
            int8_max = torch.iinfo(torch.int8).max
            max_values, _ = torch.max(torch.abs(tensor), dim=reduce_dim, keepdim=not weight_transposed)
            scales = max_values / int8_max
            if scales.count_nonzero() == 0:
                # If the scales is all zeros, the weight tensor are zeros
                quantized_weights = tensor.to(torch.int8)
            else:
                quantized_weights = tensor / scales
                quantized_weights = torch.round(quantized_weights)
                quantized_weights = torch.clamp(quantized_weights, int8_min, int8_max)
                quantized_weights = quantized_weights.to(torch.int8)
        else:
            raise NotImplementedError(f"{quantize_config.quant_dtype} for {quantize_config.quantize_method}")
    else:
        raise NotImplementedError(f"{quantize_config.quantize_method} not implemented")
    
    return quantized_weights, scales
    