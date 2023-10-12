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
from transformers_neuronx import constants


import os

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


class ContinuousBatchingConfig:
    """ The config class that contains all continuous batching related settings """

    def __init__(self, batch_size_for_shared_caches):
        self.batch_size_for_shared_caches = batch_size_for_shared_caches


class NeuronConfig():
    """
    Configuration class to store all Neuron related configs.

    Arguments:
        all_reduce_dtype (str, optional): Data type that's used for AllReduce
            CC ops, to be selected from `["float32", "float16", "bfloat16"]`.
            Default: `None`.
        sparse_attn (`sparse_attn_utils.SparseAttnConfig`, optional): Sparse
            attention related configurations. Default: `None`.
        quant (`QuantizationConfig`, optional): Quantization related
            configurations. Default: `None`.
        cast_logits_dtype (`str`, optional): Cast logits to this dtype at the end
            of every forward pass. Must be selected from `["float32", "float16", "bfloat16"]`.
            Default: Upcasts all logits to `float32`.
        continuous_batching (`ContinuousBatchingConfig`, optional): Continuous
            batching related configurations. Default: `None`.
        use_2d_cache_ids (bool, optional): Whether to use 2D layout for cache_ids (aka position_ids).
            Default: `False`.
        attention_layout (`str`, optional): Layout to be used for attention computation.
            To be selected from `["HSB", "BSH"]`.
             Default: `"HSB"`.
    """
    def __init__(self, **kargs):
        self.all_reduce_dtype = kargs.pop('all_reduce_dtype', None)
        self.sparse_attn = kargs.pop('sparse_attn', None)
        self.quant = kargs.pop('quant', None)
        self.cast_logits_dtype = kargs.pop('cast_logits_dtype', 'float32')
        self.fuse_qkv = kargs.pop('fuse_qkv', False)
        self.continuous_batching = kargs.pop('continuous_batching', None)
        self.use_2d_cache_ids = kargs.pop('use_2d_cache_ids', False)
        if self.continuous_batching:
            # Force using 2D cache_ids layout for continuous batching.
            self.use_2d_cache_ids = True
        self.attention_layout = kargs.pop('attention_layout', constants.LAYOUT_HSB)
        self.group_query_attention = kargs.pop('group_query_attention', None)
        if self.group_query_attention is not None:
            self.group_query_attention = constants.GQA(self.group_query_attention)

        self.rank_id = int(os.getenv("NEURON_RANK_ID", "0"))

        self.local_tp = os.getenv("NEURON_LOCAL_TP", None)

        if self.local_tp is not None:
            self.local_tp = int(self.local_tp)

    def get_local_tp(self, tp):
        if self.local_tp is None:
            return tp
        return self.local_tp

    def get_g_start_device_id(self, tp):
        return self.rank_id*self.get_local_tp(tp)


class GenerationConfig:

    def __init__(self, *,
        max_length = None,      # Default: Infer max sequence length from model
        do_sample = False,      # Default: Greedy
        top_k = 50,             # Default: Top 50 (when sampling)
        eos_token_id = None,    # Default: Ignore EOS token, otherwise enable early stop.
        temperature = None,     # Default: No temperature application
    ):
        self.max_length = max_length
        self.do_sample = do_sample
        self.top_k = top_k
        self.eos_token_id = eos_token_id
        self.temperature = temperature
