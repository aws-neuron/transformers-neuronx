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
from typing import Literal

class QuantizationConfig:
    """ The config class that contains all quantization related settings """

    def __init__(self, quant_dtype='s8', dequant_dtype='f16', quantize_method='vector_dynamic',
                 quantize_attn=True):
        QUANT_DTYPE_LIST = ['s8',]
        QUANTIZE_METHOD_LIST = ['vector_dynamic',]

        # The data type that the parameter is quantized into
        self.quant_dtype = quant_dtype
        if self.quant_dtype not in QUANT_DTYPE_LIST:
            raise NotImplementedError(f"{self.quant_dtype} is not implemented. \
                                      Available options are {','.join(QUANT_DTYPE_LIST)}")

        # The data type that is dequantized to 
        self.dequant_dtype = dequant_dtype

        # Which quantization algorithm to use
        self.quantize_method = quantize_method
        if self.quantize_method not in QUANTIZE_METHOD_LIST:
            raise NotImplementedError(f"{self.quantize_method} is not implemented. \
                                      Available options are {','.join(QUANTIZE_METHOD_LIST)}")        

        # Decide whether the attention layer needs be quantized
        self.quantize_attn = quantize_attn

class NeuronConfig():
    """ The class contains all Neuron related configs """
    def __init__(self, **kargs):
        # Quantization related configurations
        self.quant = kargs.pop('quant', None)
