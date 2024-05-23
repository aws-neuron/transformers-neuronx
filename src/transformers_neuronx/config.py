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
import math
import logging
import warnings
from typing import Optional, List

from transformers_neuronx import GQA, Layout, SparseAttnConfig


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
    """
    The config class that contains all continuous batching related settings
    """

    def __init__(self, batch_size_for_shared_caches):
        self.batch_size_for_shared_caches = batch_size_for_shared_caches


class GenerationConfig:

    def __init__(self, *,
        max_length = None,      # Default: Infer max sequence length from model
        do_sample = False,      # Default: Greedy
        top_k = 50,             # Default: Top 50 (when sampling)
        top_p = 1.0,            # Default: Use all tokens
        top_p_min_tokens = 1,   # Default: A minimum of 1 for Top-P sampling
        global_top_k = None,    # Default: Do not use a global top-k value
        eos_token_id = None,    # Default: Ignore EOS token, otherwise enable early stop.
        temperature = 1.0,      # Default: No temperature application
        dynamic = False,        # Default: Do not support changing generation config at runtime
        deterministic = False,  # Default: Do not use a constant 0.5 as token acceptance threshold during sampling
    ):
        self.max_length = max_length
        self.do_sample = do_sample
        self.top_k = top_k
        self.top_p = float(top_p)
        self.top_p_min_tokens = top_p_min_tokens
        self.global_top_k = global_top_k
        self.eos_token_id = eos_token_id
        self.temperature = float(temperature)
        self.dynamic = dynamic
        self.deterministic = deterministic

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, GenerationConfig):
            return False
        return self.__dict__ == value.__dict__


valid_dtypes = [
    "float32",
    "float16",
    "bfloat16",
]


class NeuronConfig():
    """
    Neuron configurations for extra features and performance optimizations.

    Arguments:
        sparse_attn: Enables attention sparsity with the given
            configurations.
        quant: Enables quantization with the given configurations.
        continuous_batching: Enables the model to be used with continuous
            batching using the given configurations.
        attention_layout: Layout to be used for attention computation.
            To be selected from `["HSB", "BSH"]`.
        collectives_layout: Layout to be used for collectives within attention.
            To be selected from `["HSB", "BSH"]`.
        cache_layout: Layout to be used for storing the KV cache.
            To be selected from `["SBH", "BSH"]`.
        padding_side: The expected tokenizer batch padding side. See:
            https://huggingface.co/docs/transformers/v4.39.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizer.padding_side
            The default padding side is "left", however using "right"
            padding enables variable length sequences to be used. This is
            enabled when using features such as continuous batching or batched
            speculation.
        group_query_attention: The sharding configuration to use when the number
            of query attention heads is not equal to the number of key/value
            heads. Neuron attempts to select the best configuration by default.
        sequence_parallel_norm: Enables sharding input sequences for rms_norm parallel
	        execution. Supported for llama models.
        sequence_parallel_norm_threshold: Sets the minimum threshold to shard rms_norm
            sequences. Use with sequence_parallel_norm.
        on_device_embedding: Enables the input embedding to be performed on
            Neuron. By default, the embedding is computed on CPU.
        on_device_generation: Enables token generation to be performed on Neuron
            hardware with the given configuration. By default token generation
            is computed on CPU. By configuring this at compilation time,
            generation configurations cannot be dynamically configured during
            inference.
        all_reduce_dtype: The data type that is used for AllReduce collectives.
            To be selected from `["float32", "float16", "bfloat16"]`.
        cast_logits_dtype: The data type to cast logits to in the forward
            pass. To be selected from `["float32", "float16", "bfloat16"]`.
        fuse_qkv: Fuses the QKV projection into a single matrix multiplication.
        qkv_tiling: Splits attention QKV to introduce "free" 128 dimensions.
        weight_tiling: Splits model MLP to introduce "free" 128 dimensions.
        mlp_in_weight_tiling_permute_order: permute order to permute the mlp input weight tiling split [K/128, 128, N/128, 128]. default=[1,2,0,3].
        mlp_out_weight_tiling_permute_order: permute order to permute the mlp output weight tiling split [K/128, 128, N/128, 128]. default=[1,2,0,3].
        log_softmax_scores: Return log-softmax scores along with logits.
        shard_over_sequence: Enables flash decoding / sequence parallel attention for token gen models, `default=False`
        output_all_logits: Return all logits from each model invocation.
        attn_output_transposed: Transposes the attention output projection weight tensor.
    """
    def __init__(self, *,
        sparse_attn: Optional[SparseAttnConfig] = None,
        quant: Optional[QuantizationConfig] = None,
        continuous_batching: Optional[ContinuousBatchingConfig] = None,
        attention_layout: Layout = Layout.HSB,
        collectives_layout: Layout = Layout.HSB,
        cache_layout: Layout = Layout.SBH,
        padding_side: str = 'left',
        group_query_attention: Optional[GQA] = None,
        sequence_parallel_norm: bool = False,
        sequence_parallel_norm_threshold: int = 2048,
        on_device_embedding: bool = False,
        on_device_generation: Optional[GenerationConfig] = None,
        all_reduce_dtype: Optional[str] = None,
        cast_logits_dtype: str = 'float32',
        fuse_qkv: bool = False,
        qkv_tiling: bool = False,
        weight_tiling: bool = False,
        mlp_in_weight_tiling_permute_order: List[int] = [1,2,0,3],
        mlp_out_weight_tiling_permute_order: List[int] = [1,2,0,3],
        log_softmax_scores: bool = False,
        shard_over_sequence: bool = False,
        output_all_logits: bool = False,
        attn_output_transposed: bool = False,
        **kwargs,
    ):
        self.all_reduce_dtype = all_reduce_dtype
        self.sparse_attn = sparse_attn
        self.quant = quant
        self.cast_logits_dtype = cast_logits_dtype
        assert cast_logits_dtype in valid_dtypes, (
            f"The `cast_logits_dtype={cast_logits_dtype}` argument must be one of {valid_dtypes}"
        )
        self.fuse_qkv = fuse_qkv
        self.continuous_batching = continuous_batching
        self.padding_side = padding_side
        assert padding_side in ['left', 'right'], (
            f"The `padding_side={padding_side}` argument must be either 'left' or 'right'"
        )

        self.lhs_aligned = padding_side == 'right'
        if 'use_2d_cache_ids' in kwargs:
            warnings.warn(
                "NeuronConfig `use_2d_cache_ids` argument is deprecated. "
                "Please specify `padding_side = 'right'`."
            )
            self.lhs_aligned = kwargs.pop('use_2d_cache_ids', False)
        if 'lhs_aligned' in kwargs:
            warnings.warn(
                "NeuronConfig `lhs_aligned` argument is deprecated. "
                "Please specify `padding_side = 'right'`."
            )
            self.lhs_aligned = kwargs.pop('lhs_aligned', False)
        if self.continuous_batching:
            # Force left alignment for continuous batching.
            self.lhs_aligned = True

        self.attention_layout = attention_layout
        self.collectives_layout = collectives_layout
        self.cache_layout = cache_layout
        self.log_softmax_scores = log_softmax_scores
        self.group_query_attention = group_query_attention
        if self.group_query_attention is not None:
            self.group_query_attention = GQA(self.group_query_attention)
        self.sequence_parallel_norm = sequence_parallel_norm
        self.sequence_parallel_norm_threshold = sequence_parallel_norm_threshold
        assert sequence_parallel_norm_threshold > 0, (
            f"sequence_parallel_norm_threshold={sequence_parallel_norm_threshold} must be greater than zero"
        )
        self.on_device_embedding = on_device_embedding
        self.on_device_generation = on_device_generation
        self.qkv_tiling = qkv_tiling
        if self.qkv_tiling is True:
            assert self.fuse_qkv is True, (
                "QKV weight tiling is currently only supported when QKV fusion is enabled."
            )

        self.weight_tiling = weight_tiling
        self.mlp_in_weight_tiling_permute_order = mlp_in_weight_tiling_permute_order
        self.mlp_out_weight_tiling_permute_order = mlp_out_weight_tiling_permute_order

        assert self.mlp_in_weight_tiling_permute_order.index(2) < self.mlp_in_weight_tiling_permute_order.index(3), \
            "original dim 2 has to be front of dim 3 after applying `mlp_in_weight_tiling_permute_order`"

        assert self.mlp_out_weight_tiling_permute_order.index(2) < self.mlp_out_weight_tiling_permute_order.index(3), \
            "original dim 2 has to be front of dim 3 after applying `mlp_out_weight_tiling_permute_order`"

        if os.environ.get("NEURON_INTERNAL_TRANSFORM_WEIGHT_LAYOUT", False):
            warnings.warn(
                "NEURON_INTERNAL_TRANSFORM_WEIGHT_LAYOUT is deprecated. "
                "To enable weight tiling, please use `NeuronConfig(weight_tiling=True)` instead.",
            )
            self.weight_tiling = True
        self.output_all_logits = output_all_logits

        assert len(kwargs) == 0, (
            f"Unexpected NeuronConfig keyword arguments: {kwargs}"
        )

        self.rank_id = int(os.getenv("NEURON_RANK_ID", "0"))

        self.local_tp = os.getenv("NEURON_LOCAL_TP", None)

        self.pp_stages = int(os.getenv("NEURON_PP_STAGES", 1))

        if self.local_tp is not None:
            self.local_tp = int(self.local_tp)

        self.dist = None

        self.layer_partition = {}

        self.shard_over_sequence = shard_over_sequence

        self.is_sequence_parallel = False

        self.attn_output_transposed = attn_output_transposed

    @property
    def use_2d_cache_ids(self):
        return self.lhs_aligned

    @property
    def vectorize_last_token_id(self):
        return self.lhs_aligned

    def is_valid_layer(self, layer_id):
        if not self.is_pp():
            return True
        return layer_id in self.layer_partition[self.rank_id]

    def is_pp(self):
        return self.pp_stages > 1

    def auto_layer_partition(self, num_layers):
        self.num_layers = num_layers
        if not self.is_pp():
            return list(range(self.num_layers))

        num_layers_per_stage = math.ceil(num_layers / self.pp_stages)

        for i in range(self.pp_stages):
            self.layer_partition[i] = [l for l in range(i*num_layers_per_stage, min((i+1)*num_layers_per_stage, num_layers))]

        logging.debug(f"auto_layer_partition: {self.layer_partition}")

        return self.layer_partition[self.rank_id]

    def valid_layers(self):
        if self.is_pp():
            return len(self.layer_partition[self.rank_id])
        else:
            return self.num_layers

    def is_valid_lm_head(self):
        if self.is_pp():
            return self.last_rank()
        return True

    def first_rank(self):
        return self.rank_id == 0

    def last_rank(self):
        return self.rank_id == self.pp_stages - 1

    def get_g_device_count(self, tp_degree):
        return self.pp_stages * tp_degree

    def get_replica_groups(self, tp_degree):
        if self.is_pp():
            return [list(range(self.rank_id*tp_degree, (self.rank_id+1)*tp_degree))]

        return [list(range(tp_degree))]

    def get_local_tp(self, tp):
        if self.local_tp is None:
            return tp
        return self.local_tp

    def get_g_start_device_id(self, tp):
        return self.rank_id*self.get_local_tp(tp)

