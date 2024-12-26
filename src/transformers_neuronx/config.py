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
import json
import enum
import math
import logging
import warnings
import contextlib
from typing import Optional, List

from transformers_neuronx import GQA, Layout, SparseAttnConfig

import torch


class QuantizationConfig:
    """ The config class that contains all quantization related settings """

    def __init__(self, quant_dtype='s8', dequant_dtype='f16', quantize_method='vector_dynamic',
                 quantize_attn=True, no_quantize_list=[]):
        QUANT_DTYPE_LIST = ['s8', 'f8e4m3fn']
        QUANTIZE_METHOD_LIST = ['vector_dynamic', 'direct_cast']
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
        self.no_quantize_list = no_quantize_list

    def is_unit_scale(self, layer_num):
        return f"model.layers.{layer_num}" in self.no_quantize_list

class KVCacheQuantizationConfig:
    """ The config class that contains all KV cache quantization related settings """
    def __init__(self, quant_dtype='s8', dequant_dtype='bf16', quantize_method='direct_cast'):
        QUANT_DTYPE_LIST = ['s8', 'f16', 'bf16', 'f8e4m3fn']
        # TODO: add vector_dynamic/absmax quantization support
        QUANTIZE_METHOD_LIST = ['direct_cast',]
        self.quant_dtype = quant_dtype
        if self.quant_dtype not in QUANT_DTYPE_LIST:
            raise NotImplementedError(f"{self.quant_dtype} is not implemented. "
                                    f"Available options are {','.join(QUANT_DTYPE_LIST)}")
        self.dequant_dtype = dequant_dtype
        self.quantize_method = quantize_method
        if self.quantize_method not in QUANTIZE_METHOD_LIST:
            raise NotImplementedError(f"{self.quantize_method} is not implemented. "
                                    f" Available options are {','.join(QUANTIZE_METHOD_LIST)}")

class ContinuousBatchingConfig:
    """
    The config class that contains all continuous batching related settings
    """

    def __init__(self, **kwargs):
        self.max_num_seqs = kwargs.pop("max_num_seqs") if kwargs.get("max_num_seqs", None) is not None else kwargs.pop("batch_size_for_shared_caches")
        self.max_model_len = kwargs.pop("max_model_len") if kwargs.get("max_model_len", None) is not None else None
        self.optimized_paged_attention = kwargs.pop("optimized_paged_attention") if kwargs.get("optimized_paged_attention", None) is not None else False
        self.enable_chunked_prefill = kwargs.pop("enable_chunked_prefill") if kwargs.get("enable_chunked_prefill", None) is not None else False
        if self.enable_chunked_prefill:
            assert self.optimized_paged_attention, "chunked prefill is only supported with optimized paged attention"
        self.block_size = None
        self.num_blocks = None
        assert len(kwargs) == 0, f"unexpected key word arguments: {kwargs.keys()}"

    @property
    def batch_size_for_shared_caches(self) -> int:
        return self.max_num_seqs

    @property
    def _paged_attention(self) -> bool:
        if self.block_size and self.max_model_len:
            if self.block_size < self.max_model_len:
                return True
        return False

    def init_cache_engine(self, block_size, num_blocks):
        self.block_size = block_size
        self.num_blocks = num_blocks



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
        per_batch_line = False, # Default: Do not use different sampling paramerters for each batch line
    ):
        self.max_length = max_length
        self.do_sample = do_sample
        self.per_batch_line = per_batch_line
        self.top_k = top_k
        self.top_p = float(top_p) if not isinstance(top_p, list) else self._convert_list_to_float(top_p)
        self.temperature = float(temperature) if not isinstance(temperature, list) else self._convert_list_to_float(temperature)
        self.top_p_min_tokens = top_p_min_tokens
        self.global_top_k = global_top_k
        self.eos_token_id = eos_token_id
        self.dynamic = dynamic
        self.deterministic = deterministic

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, GenerationConfig):
            return False
        return self.__dict__ == value.__dict__

    def _convert_list_to_float(self, params):
        return [float(p) for p in params]

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
        kv_cache_quant: Enables KV cache quantization with the given configurations.
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
        bf16_rms_norm: Uses BF16 weights and hidden states input for RMS norm operations.
            By default, the RMS norm operates on FP32 dtype of inputs.
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
        mlp_out_weight_transpose: transpose the mlp output weight layout from [H, F] into [F, H]. `default=False`
        log_softmax_scores: Return log-softmax scores along with logits.
        shard_over_sequence: Enables flash decoding / sequence parallel attention for token gen models, `default=False`
        duplicate_q_weight_sos: Duplicate q weights to skip allgather in shard_over_sequence
        output_all_logits: Return all logits from each model invocation.
        fused_rmsnorm_qkv: Use the fused RMS norm and QKV input projection kernel.
        fused_rmsnorm_mlp: Use the fused RMSNorm and MLP BIR kernel for llama3.
        attn_output_transposed: Transposes the attention output projection weight tensor.
        compilation_worker_count: Count of concurrent compilation workers.
    """
    def __init__(self, *,
        sparse_attn: Optional[SparseAttnConfig] = None,
        quant: Optional[QuantizationConfig] = None,
        kv_cache_quant: Optional[KVCacheQuantizationConfig] = None,
        continuous_batching: Optional[ContinuousBatchingConfig] = None,
        attention_layout: Layout = Layout.HSB,
        collectives_layout: Layout = Layout.HSB,
        cache_layout: Layout = Layout.SBH,
        padding_side: str = 'left',
        group_query_attention: Optional[GQA] = None,
        sequence_parallel_norm: bool = False,
        sequence_parallel_norm_threshold: int = 2048,
        bf16_rms_norm: bool = False,
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
        fused_rmsnorm_qkv: bool = False,
        fused_rmsnorm_mlp: bool = False,
        mlp_out_weight_transpose: bool = False,
        fuse_mlp: bool = False,
        is_eagle_target: bool = False,
        is_eagle_draft: bool = False,
        has_pre_attention_norm: bool = True,
        compilation_worker_count: Optional[int] = None,
        duplicate_q_weight_sos: bool = False,
        **kwargs,
    ):
        self.all_reduce_dtype = all_reduce_dtype
        self.sparse_attn = sparse_attn
        self.quant = quant
        self.kv_cache_quant = kv_cache_quant
        self.cast_logits_dtype = cast_logits_dtype
        assert cast_logits_dtype in valid_dtypes, (
            f"The `cast_logits_dtype={cast_logits_dtype}` argument must be one of {valid_dtypes}"
        )
        self.fuse_qkv = fuse_qkv
        self.fuse_mlp = fuse_mlp
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
            self.padding_side = "right"
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
        self.bf16_rms_norm = bf16_rms_norm
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

        self.fused_rmsnorm_qkv = fused_rmsnorm_qkv
        self.fused_rmsnorm_mlp = fused_rmsnorm_mlp

        if any([not self.fuse_qkv,
                self.attention_layout != Layout.BSH,
                self.group_query_attention != GQA.REPLICATED_HEADS,
                self.sequence_parallel_norm]):
            self.fused_rmsnorm_qkv = False

        if any([self.attention_layout != Layout.BSH,
                self.group_query_attention != GQA.REPLICATED_HEADS]):
            self.fused_rmsnorm_mlp = False

        self.mlp_out_weight_transpose = mlp_out_weight_transpose

        if self.shard_over_sequence:
            assert self.sparse_attn is None, f"sparse attn is not supported with flash decoding"
            if not (self.continuous_batching and self.continuous_batching.optimized_paged_attention):
                assert self.cache_layout == Layout.SBH, f"flash decoding only support SBH layout , got {self.cache_layout}"

        self.duplicate_q_weight_sos = duplicate_q_weight_sos

        self.is_eagle_target = is_eagle_target
        self.is_eagle_draft = is_eagle_draft
        self.has_pre_attention_norm = has_pre_attention_norm
        self.compilation_worker_count = compilation_worker_count

    @property
    def use_2d_cache_ids(self):
        return self.lhs_aligned

    @property
    def use_1d_query(self):
        return self.cache_layout == Layout.BSH and self.padding_side == 'right'

    @property
    def paged_attention(self):
        if self.continuous_batching:
            return self.continuous_batching._paged_attention and self.bsh_cache_layout
        return False

    @property
    def enable_chunked_prefill(self):
        return self.paged_attention and self.continuous_batching.enable_chunked_prefill

    @property
    def optimized_paged_attention(self):
        return self.paged_attention and self.continuous_batching.optimized_paged_attention

    @property
    def bsh_cache_layout(self):
        from transformers_neuronx import constants
        return self.cache_layout == constants.Layout.BSH

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

    def to_json(self):
        json_serializable_types = (str, int, float, bool)
        def _to_json(obj):
            if obj is None or isinstance(obj, json_serializable_types):
                return obj
            elif isinstance(obj, enum.Enum):
                return obj.value
            elif isinstance(obj, torch.Tensor):
                return obj.tolist()
            elif isinstance(obj, list):
                return [ _to_json(e) for e in obj ]
            elif isinstance(obj, dict):
                return { _to_json(k): _to_json(v) for k, v in obj.items() }
            elif isinstance(obj, tuple):
                return str(tuple(_to_json(e) for e in obj))
            else:
                as_dict = obj.__dict__
                return _to_json(as_dict)
        return _to_json(self)


@contextlib.contextmanager
def maybe_dump_config(config, neuron_config):
    if "NEURONX_DUMP_TO" in os.environ and (neuron_config or config):
        dump_to = os.environ.get('NEURONX_DUMP_TO', '/tmp')
        os.makedirs(dump_to, exist_ok=True)
        config_to_dump = {}
        if neuron_config:
            config_to_dump['neuron_config'] = neuron_config.to_json()
        if config:
            key_aliases = {
                'attention_dropout': ['attn_pdrop'],
                'hidden_act': ['activation_function'],
                'max_position_embeddings': ['n_positions'],
                'num_hidden_layers': ['n_layer'],
                'num_attention_heads': ['n_head'],
                'intermediate_size': ['ffn_dim', 'n_inner'],
                'hidden_size': ['n_embd'],
                'initializer_range': ['init_std']
            }
            key_mapping = {}  # inverted and flattened key_aliases
            for key, aliases in key_aliases.items():
                for alias in aliases: key_mapping[alias] = key
            model_config = { key_mapping.get(k, k): v for k, v in config.__dict__.items() }
            config_to_dump['model_config'] = model_config
        config_dump_path = os.path.join(dump_to, 'neuron_model_config.json')
        with open(config_dump_path, 'w') as fp:
            json.dump(config_to_dump, fp)
        yield
        # by now, the config has been copied into the sub-directories, so we can clean this one up
        try:
            os.remove(config_dump_path)
        except FileNotFoundError:
            pass
    else:
        yield
