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
import itertools
import warnings
import logging

import torch
from transformers_neuronx import base
from transformers_neuronx import bucket
from transformers_neuronx import compiler
from transformers_neuronx import dtypes
from transformers_neuronx import hlo
from transformers_neuronx import ops
from transformers_neuronx import parallel
from transformers_neuronx import utils
from transformers_neuronx import config
from transformers_neuronx import quantize
from transformers_neuronx import constants
from transformers_neuronx import global_debugger
from transformers_neuronx.layers import generation
from transformers_neuronx.config import NeuronConfig, GenerationConfig
from transformers_neuronx.utils import interleave_qkv
from transformers_neuronx.util.token_tree import generate_attention_mask
from transformers_neuronx.llama.hlo import LlamaForSamplingNoEmbeddingHlo

from safetensors.torch import save_file
from safetensors import safe_open


class DecoderLmHeadForSamplingNoEmbedding(torch.nn.Module, base.NeuronBaseSerializer):

    def __init__(self, tp_degree, n_positions_list, n_active_tokens, batch_size,
                 attention_head_size, amp, num_layers, n_head=None, n_kv_head=0,
                 unroll=None, neuron_config=None, allow_pad=True, prefixed_length=0,
                 return_all_outputs=True, builder=None, tag=None, prompt_batch_size=None, token_tree=None):
        super().__init__()
        if unroll is None:
            unroll = num_layers
        self.tp_degree = tp_degree
        self.n_positions_list = n_positions_list
        self.n_active_tokens = n_active_tokens
        self.batch_size = bucket.batch_sizes(batch_size)
        self.prompt_batch_size = prompt_batch_size
        self.attention_head_size = attention_head_size  # TODO: rename to size_per_head
        self.n_head = n_head
        self.n_kv_head = n_kv_head if (n_kv_head > 0) else n_head
        self.return_all_outputs=return_all_outputs
        self.amp = amp
        self.num_layers = num_layers
        self.unroll = unroll
        self.neuron_config = NeuronConfig() if neuron_config is None else neuron_config
        self.prefixed_length = prefixed_length
        self.layers = torch.nn.ModuleList()
        self.ln_f_weight = None
        self.ln_f_bias = None
        self.lm_head_weight = None
        self.lm_head_bias = None
        self.logits_indices = None
        self.generation_inputs = []
        self.top_k = None
        self.top_p = None
        self.temperature = None
        self.top_p_min_tokens = None
        self.inputs_sdim = None
        self.inputs_builder = None
        self.embedding_builder = None
        self.layer_builder = None
        self.ln_lm_head_params = []
        self.ln_lm_head_builder = None
        self.program = None
        self.pre_layer_parameters = []
        self.pre_layer_builder = None
        self.allow_pad = allow_pad
        self.use_executor = False
        self.return_ranks = -1 if not self.neuron_config.on_device_generation else 1
        self.need_reorder_cache = False
        self.builder = builder
        self.check_gqa_fallback()
        self.token_tree = token_tree
        self.tag = tag
        self._cpu_compile = False

    @property
    def bsh_cache_layout(self):
        return self.neuron_config.cache_layout == constants.LAYOUT_BSH

    def check_gqa_fallback(self):
        """
        Check if a fallback mechanism is needed for a user-provided GQA config.

        The initial fallback mechanism will be that no special GQA configuration
        is used. This will attempt to evenly distribute Q and KV heads to all
        NeuronCores.

        The second (safest) fallback mechanism will replicate the KV heads to be
        equal to the number of Q heads. This makes the GQA model look identical
        to an MHA model.
        """
        gqa = self.neuron_config.group_query_attention

        if gqa is None:
            # MHA Early exit - This avoids emitting irrelevant GQA warnings
            if self.n_head == self.n_kv_head:
                if self.neuron_config.shard_over_sequence:
                    warnings.warn(
                            f'Cannot enable shard_over_sequence when a n_heads ({self.n_head}) == n_kv_heads ({self.n_kv_head})'
                            f'disabling shard over sequence'
                        )
                    self.neuron_config.shard_over_sequence = False
                return
            self.neuron_config.group_query_attention = constants.GQA.SHARD_OVER_HEADS

        if gqa == constants.GQA.REPLICATED_HEADS:
            return

        if gqa == constants.GQA.SHARD_OVER_BATCH:
            success = True
            for batch_size in self.batch_size:
                if batch_size % self.tp_degree != 0:
                    warnings.warn(
                        f'Cannot enable "{gqa}" when a batch size '
                        f'({batch_size} in {self.batch_size}) is not evenly '
                        f'divisible by the tensor parallel degree '
                        f'({self.tp_degree})'
                    )
                    success = False
                    self.neuron_config.group_query_attention = constants.GQA.SHARD_OVER_HEADS
            if success:
                return

        if gqa == constants.GQA.ALL_GATHER_HEADS:
            if (self.n_kv_head * self.attention_head_size) % self.tp_degree != 0:
                warnings.warn(
                    f'Cannot enable "{gqa}" when the hidden size of KV '
                    f'({self.n_kv_head} x {self.attention_head_size}) is not evenly divisible '
                    f'by the tensor parallel degree ({self.tp_degree})'
                )
                self.neuron_config.group_query_attention = constants.GQA.SHARD_OVER_HEADS

            if self.n_head % self.tp_degree != 0:
                # try pad on n_head, if pad_size could be evenly disible by n_kv_head,
                # then we can evenly distribute same number of padding q_head to each k/v head
                pad_size = utils.get_pad_size(self.n_head, self.tp_degree)

                if pad_size % self.n_kv_head == 0:
                    return
                else:
                    warnings.warn(
                        f'Cannot enable "{gqa}" when the number of padding {pad_size} need for query '
                        f'attention heads ({self.n_head}) with the tensor parallel degree ({self.tp_degree}) '
                        f'is not divisible by KV heads ({self.n_kv_head})'
                    )
                    self.neuron_config.group_query_attention = constants.GQA.SHARD_OVER_HEADS
            else:
                return

        if self.n_kv_head % self.tp_degree != 0:
            warnings.warn(
                f'KV head replication will be enabled since the number of KV '
                f'heads ({self.n_kv_head}) is not evenly divisible by the '
                f'tensor parallel degree ({self.tp_degree})'
            )
            self.neuron_config.group_query_attention = constants.GQA.REPLICATED_HEADS

    def init_context_decoder(self, unroll, buckets, model_obj, context_batch_sizes=None):
        cls = type(self)
        decoder_lm_head = {}
        if context_batch_sizes:
            # if context_batch_sizes is passed we directly use it. This is useful for cases
            # like chunked prefill where batch size bucket is used for the number of active KV cache 
            # blocks.
            self.context_batch_sizes = context_batch_sizes
        elif self.prompt_batch_size:
            self.context_batch_sizes = [self.prompt_batch_size]
        elif self.neuron_config and self.neuron_config.continuous_batching:
            self.context_batch_sizes = [1]
        else:
            self.context_batch_sizes = self.batch_size
        return_all_outputs = False
        if self.neuron_config and self.neuron_config.output_all_logits:
            return_all_outputs = True
        for context_length_estimate in buckets:
            for batch_size in self.context_batch_sizes:
                decoder_lm_head[context_length_estimate, batch_size] = cls(
                    tp_degree=self.tp_degree,
                    n_positions_list=[context_length_estimate],
                    n_active_tokens=context_length_estimate,
                    batch_size=batch_size,
                    attention_head_size=self.attention_head_size,
                    amp=self.amp,
                    num_layers=self.num_layers,
                    n_head=self.n_head,
                    n_kv_head=self.n_kv_head,
                    unroll=unroll,
                    neuron_config=self.neuron_config,
                    allow_pad=self.allow_pad,
                    return_all_outputs=return_all_outputs,
                    builder=self.builder,
                    tag="context"
                )
                base.NeuronModelBase.register_for_serialization(model_obj,decoder_lm_head[context_length_estimate, batch_size])
        return decoder_lm_head


    def init_token_decoder(self,unroll, buckets, model_obj):
        cls = type(self)
        decoder_lm_head = cls(
            tp_degree=self.tp_degree,
            n_positions_list=buckets,
            n_active_tokens=1,
            batch_size=self.batch_size,
            attention_head_size=self.attention_head_size,
            amp=self.amp,
            num_layers=self.num_layers,
            n_head=self.n_head,
            n_kv_head=self.n_kv_head,
            unroll=unroll,
            neuron_config=self.neuron_config,
            allow_pad=True,
            return_all_outputs=True,
            builder=self.builder,
            tag="token",
        )
        if not self.neuron_config.enable_chunked_prefill: # skip token model for chunked prefill due to not needed and also compiler error
            base.NeuronModelBase.register_for_serialization(model_obj,decoder_lm_head)
        decoder_lm_head.add_inputs_builder(self.builder.inputs)
        if hasattr(self.builder, 'pre_layer'):
            decoder_lm_head.add_pre_layer_builder(self.builder.pre_layer)
        decoder_lm_head.add_layer_builder(self.builder.layer)
        decoder_lm_head.add_ln_lm_head_builder(self.builder.ln_lm_head)
        if hasattr(self.builder, 'embedding'):
            decoder_lm_head.add_embedding_builder(self.builder.embedding)
        return decoder_lm_head

    def init_speculative_decoder(self, unroll, buckets, model_obj, n_active_tokens, batch_size=None, token_tree=None):
        cls = type(self)
        decoder_lm_head = cls(
            tp_degree=self.tp_degree,
            n_positions_list=buckets,
            n_active_tokens=n_active_tokens,
            batch_size=self.batch_size if batch_size is None else batch_size,
            attention_head_size=self.attention_head_size,
            amp=self.amp,
            num_layers=self.num_layers,
            n_head=self.n_head,
            n_kv_head=self.n_kv_head,
            unroll=unroll,
            neuron_config=self.neuron_config,
            allow_pad=True,
            return_all_outputs=True,
            builder=self.builder,
            token_tree=token_tree,
            tag=f"speculation-k{n_active_tokens}",
        )
        base.NeuronModelBase.register_for_serialization(model_obj,decoder_lm_head)
        return decoder_lm_head

    def init_window_context_decoder(self, unroll, buckets, model_obj, n_active_tokens):
        cls = type(self)
        return_all_outputs = False
        if self.neuron_config and self.neuron_config.output_all_logits:
            return_all_outputs = True
        decoder_lm_head = cls(
            tp_degree=self.tp_degree,
            n_positions_list=buckets,
            n_active_tokens=n_active_tokens,
            batch_size=self.batch_size,
            attention_head_size=self.attention_head_size,
            amp=self.amp,
            num_layers=self.num_layers,
            n_head=self.n_head,
            n_kv_head=self.n_kv_head,
            unroll=unroll,
            neuron_config=self.neuron_config,
            allow_pad=True,
            return_all_outputs=return_all_outputs,
            builder=self.builder,
            tag=f"window-width{n_active_tokens}",
        )
        assert self.neuron_config.shard_over_sequence is False, "flash decoding not supported with windowed context encoder"
        base.NeuronModelBase.register_for_serialization(model_obj,decoder_lm_head)
        return decoder_lm_head

    def setup_reorder_cache(self):
        self.need_reorder_cache = True

    def enable_executor(self, return_ranks=-1):
        self.return_ranks = return_ranks if not self.neuron_config.on_device_generation else 1
        self.program.enable_executor()

    def add_inputs_builder(self, inputs_builder):
        self.inputs_builder = inputs_builder

    def add_embedding_builder(self, embedding_builder):
        self.embedding_builder = embedding_builder

    def add_pre_layer_parameter(self, param, sharding=None, allow_pad=False):
        self.pre_layer_parameters.append((param, sharding, allow_pad))

    def add_pre_layer_builder(self, builder):
        self.pre_layer_builder = builder

    def add_layer_builder(self, layer_builder):
        self.layer_builder = layer_builder

    def add_ln_lm_head_builder(self, ln_lm_head_builder):
        self.ln_lm_head_builder = ln_lm_head_builder

    def add_post_layer_builder(self, builder):
        self.post_layer_builder = builder

    def new_layer(self, is_unit_scale=False):
        # handle cases where some layers are not quantized with unit scale
        *_, n_positions = self.n_positions_list
        layer = DecoderLayer(self.tp_degree, n_positions, self.batch_size, self.attention_head_size,
                             amp=self.amp, neuron_config=self.neuron_config, allow_pad=self.allow_pad, n_active_tokens=self.n_active_tokens,
                             n_head=self.n_head, n_kv_head=self.n_kv_head, layer_num=len(self.layers),
                             is_unit_scale=is_unit_scale)
        layer._cpu_compile = self._cpu_compile
        self.layers.append(layer)
        return layer

    def add_final_layer_norm(self, weight, bias):
        self.ln_f_weight = weight
        self.ln_f_bias = bias

    def add_lm_head(self, weight, bias=None):
        self.lm_head_weight = weight
        self.lm_head_bias = bias

    def to_neuron(self):
        # validate on device generation params
        if self.neuron_config.on_device_generation:
            self.validate_generation_configs(self.neuron_config.on_device_generation)

        manipulator = MaybeParallelTensorManipulator(self.tp_degree, on_cpu=self._cpu_compile, rank_id=self.neuron_config.rank_id, local_tp_degree=self.neuron_config.get_local_tp(self.tp_degree))
        self.pre_layer_parameters = self._prepare_pre_layer_params(manipulator, self.pre_layer_parameters)

        self.ln_f_weight = manipulator.duplicate(self.ln_f_weight)
        self.ln_f_bias = manipulator.duplicate(self.ln_f_bias)
        _, vocab_size = self.lm_head_weight.shape
        # Pad vocab size such that it can be divided by the following factor
        divisor = int(os.environ.get('NEURON_VOCAB_PAD_DIVISOR', str(self.tp_degree)))
        vocab_pad = utils.get_pad_size(vocab_size, divisor)
        lm_head_weight = torch.nn.functional.pad(self.lm_head_weight, (0, vocab_pad, 0, 0))
        self.lm_head_weight = manipulator.shard_along(lm_head_weight, dim=1)
        ln_lm_head_params = [self.ln_f_weight, self.ln_f_bias, self.lm_head_weight]
        ln_lm_head_params = [param for param in ln_lm_head_params if param is not None]
        if self.lm_head_bias is not None:
            self.lm_head_bias = manipulator.shard_along(self.lm_head_bias, dim=0)
            ln_lm_head_params.append(self.lm_head_bias)
        if self.neuron_config.on_device_generation:
            logits_indices = torch.arange(lm_head_weight.shape[-1], dtype=torch.int32)
            self.logits_indices = manipulator.shard_along(logits_indices, dim=0)
            ln_lm_head_params.append(self.logits_indices)
            if self.neuron_config.on_device_generation.dynamic:
                config = self.neuron_config.on_device_generation
                self.top_k = manipulator.duplicate(torch.tensor(config.top_k))
                self.generation_inputs.append(self.top_k)
                self.top_p = manipulator.duplicate(torch.tensor(config.top_p))
                self.generation_inputs.append(self.top_p)
                self.temperature = manipulator.duplicate(torch.tensor(config.temperature))
                self.generation_inputs.append(self.temperature)
                self.top_p_min_tokens = manipulator.duplicate(torch.tensor(config.top_p_min_tokens))
                self.generation_inputs.append(self.top_p_min_tokens)
                # FIXME: Use a better mechanism to pass extra params into the model
                ln_lm_head_params += self.generation_inputs
        self.ln_lm_head_params = ln_lm_head_params
        self.finish_program_setup()


    def finish_program_setup(self):
        self.program = self._build_program()
        # setup_reorder_cache needs to be able to be called before compilation
        # and after compilation for backwards compatability.
        # If called before compilation this logic will be reached and it will
        # create the HLO for reorder_cache now which can be deserialized or compiled normally.
        # If called after, setup_reorder_cache will be called from the NeuronModelBase without
        # serialization logic and will be compiled normally.
        if self.need_reorder_cache:
            self.program.setup_reorder_cache(also_compile_now=False)

        if self.neuron_config.shard_over_sequence:
            assert self.tp_degree%self.n_kv_head == 0, f"tp_degree {self.tp_degree} not divisble by n_kv_heads {self.n_kv_head} shard_over_sequence is not supported"
            self.kv_replication = self.tp_degree//self.n_kv_head

    def build_weight_shared(self, n_positions_list=None, n_active_tokens=None, batch_size=None,
                            unroll=None, share_caches=False, new=None, embed_weight=None):
        if new == None:
            cls = type(self)
            new = cls(
                self.tp_degree, self.n_positions_list, self.n_active_tokens, self.batch_size, self.attention_head_size,
                amp=self.amp, num_layers=self.num_layers, n_head=self.n_head, n_kv_head=self.n_kv_head,
                unroll=self.unroll, neuron_config=self.neuron_config, allow_pad=self.allow_pad,
                prefixed_length=self.prefixed_length, return_all_outputs=self.return_all_outputs
            )
        if new.token_tree is not None:
            new.add_inputs_builder(self.builder.token_tree_inputs)
            new.add_embedding_builder(self.builder.token_tree_embedding)
            new.add_pre_layer_builder(self.builder.token_tree_pre_layer)
            new.add_layer_builder(self.builder.token_tree_layer)
        else:
            new.add_inputs_builder(self.inputs_builder)
            new.add_embedding_builder(self.embedding_builder)
            new.add_pre_layer_builder(self.pre_layer_builder)
            new.add_layer_builder(self.layer_builder)
        new.add_ln_lm_head_builder(self.ln_lm_head_builder)
        new._cpu_compile = self._cpu_compile
        if self.neuron_config.log_softmax_scores:
            new.add_post_layer_builder(self.post_layer_builder)
        for layer in self.layers:
            new_layer = new.new_layer()
            new_layer.assign_parameters(layer)
            if self.neuron_config.shard_over_sequence:
                new_layer.kv_replication = layer.kv_replication
            if share_caches:
                buckets_from_src = self.neuron_config and self.neuron_config.continuous_batching
                new_layer.assign_caches(layer, buckets_from_src=buckets_from_src)
            else:
                new_layer.init_caches()
            new_layer.extra_parameters = layer.extra_parameters
        if new.token_tree is not None:
            if self.neuron_config.on_device_embedding:
                new.add_pre_layer_parameter(embed_weight, sharding=1, allow_pad=True)
            new.add_pre_layer_parameter(generate_attention_mask(new.token_tree))
            manipulator = MaybeParallelTensorManipulator(self.tp_degree, on_cpu=self._cpu_compile, rank_id=self.neuron_config.rank_id, local_tp_degree=self.neuron_config.get_local_tp(new.tp_degree))
            new.pre_layer_parameters = self._prepare_pre_layer_params(manipulator, new.pre_layer_parameters)
        else:
            new.pre_layer_parameters = self.pre_layer_parameters
        new.add_final_layer_norm(self.ln_f_weight, self.ln_f_bias)
        new.add_lm_head(self.lm_head_weight, self.lm_head_bias)
        ln_lm_head_params = [new.ln_f_weight, new.ln_f_bias, new.lm_head_weight]
        ln_lm_head_params = [param for param in ln_lm_head_params if param is not None]
        if new.lm_head_bias is not None:
            ln_lm_head_params.append(new.lm_head_bias)
        if self.neuron_config.on_device_generation:
            new.logits_indices = self.logits_indices
            ln_lm_head_params.append(self.logits_indices)
            if self.neuron_config.on_device_generation.dynamic:
                new.top_k = self.top_k
                new.generation_inputs.append(self.top_k)
                new.top_p = self.top_p
                new.generation_inputs.append(self.top_p)
                new.temperature = self.temperature
                new.generation_inputs.append(self.temperature)
                new.top_p_min_tokens = self.top_p_min_tokens
                new.generation_inputs.append(self.top_p_min_tokens)
                ln_lm_head_params += new.generation_inputs
        new.ln_lm_head_params = ln_lm_head_params
        new.program = new._build_program()
        return new

    def setup(self):
        self.program.setup(self.layers, self.pre_layer_parameters, self.ln_lm_head_params)
        if self.need_reorder_cache:
            self.program.setup_reorder_cache_kernels()
        if self.use_executor:
            self.enable_executor()

    def reset(self):
        for layer in self.layers:
            layer.reset()

    def forward_single(self, *inputs):
        """
        Fast-path forward function which avoids as much overhead as possible.

        This path makes the assumption that inputs are correctly sized for a
        sequence length of 1. This allows us to avoid checking buckets, slicing,
        etc.
        """
        _, cache_ids, start_ids, *_ = inputs
        batch_size = start_ids.shape[0]
        # With 2D cache_ids, take largest cache_id and use the power-of-two policy to find the appropriate bucket.
        if self.neuron_config and self.neuron_config.use_2d_cache_ids:
            # Enabling Output bucketing for continuous batching with SBH cache layout out of available cache layouts[SBH, BSH].
            if self.neuron_config.cache_layout == constants.LAYOUT_SBH:
                bucket_id = self.program.find_bucket_id(cache_ids.max().item())
                batch_size, _ = cache_ids.shape
            # Enabling Output bucketing for Paged attention with BSH cache layout.
            elif self.bsh_cache_layout and self.neuron_config.optimized_paged_attention:
                bucket_id = 0
                batch_size = self.program.find_block_bucket_size(context_length=inputs[1],block_size=self.neuron_config.continuous_batching.block_size)
            else:
                bucket_id = 0
                batch_size, _ = cache_ids.shape
        elif self.neuron_config and self.neuron_config.shard_over_sequence:
             bucket_id = self.program.find_bucket_id(cache_ids.item() + self.kv_replication - 1)
        else:
            bucket_id = self.program.find_bucket_id(cache_ids.item())
        if self.use_executor:
            output = self.program.execute(bucket_id, batch_size, *inputs, return_ranks=self.return_ranks)
            self.program.debug_tensors_to_host(bucket_id, batch_size)
            return output
        else:
            self.program.inputs_host_to_device(inputs, batch_size)
            self.program.run(bucket_id, batch_size)
            self.program.debug_tensors_to_host(bucket_id, batch_size)
            return self.program.maybe_logits_device_to_host(batch_size,  return_ranks=self.return_ranks)

    def forward(self, *inputs):
        hidden, cache_ids, start_ids, *_ = inputs
        batch_size = 1 if self.neuron_config.use_1d_query else start_ids.shape[0]
        if self.neuron_config.enable_chunked_prefill:
            batch_size = self.batch_size[0] # in this case batch size is single element list
            assert len(self.batch_size) == 1
        sequence_dim, *_ = self.inputs_sdim
        sequence_length = hidden.shape[sequence_dim]
        if sequence_length == 1:
            return self.forward_single(*inputs)

        outputs = None
        slice_loop_var = range(0, sequence_length, self.n_active_tokens)
        if self.n_active_tokens > 1 and self.return_all_outputs:
            slice_loop_var = [0]

        for start in slice_loop_var:
            slicing = slice(start, start + self.n_active_tokens)
            input_tensors = []
            for sdim, tensor in zip(self.inputs_sdim, inputs):
                if sdim is not None:
                    slices = [slice(None) for _ in tensor.shape]
                    slices[sdim] = slicing
                    tensor = tensor[tuple(slices)].contiguous()
                input_tensors.append(tensor)
            max_id = cache_ids.max().item()
            min_id = cache_ids.min().item()
            # When context_length == m * n_active_tokens, bucket-size of n_active_tokens should be chosen.
            # This is useful for Fusion-In-Decoder case, where 2nd n_active_tokens don't need to attend to
            # 1st n_active_tokens.
            if self.neuron_config.enable_chunked_prefill:
                bucket_id = self.program.find_bucket_id(self.n_active_tokens - 1) # can't use max_id since it can be high
            else:
                bucket_id = self.program.find_bucket_id(max_id)
            if self.use_executor:
                if self.neuron_config and self.neuron_config.sequence_parallel_norm:
                    self.program.inputs_host_to_device(input_tensors, batch_size)
                    input_tensors = []
                outputs = self.program.execute(bucket_id, batch_size, *input_tensors, return_ranks=self.return_ranks)
            else:
                self.program.inputs_host_to_device(input_tensors, batch_size)
                self.program.run(bucket_id, batch_size)
            self.program.debug_tensors_to_host(bucket_id, batch_size)

        if not self.use_executor:
            outputs = self.program.maybe_logits_device_to_host(batch_size, return_ranks=self.return_ranks)

        return outputs

    def embed_positions_ids(self, position_ids, start_ids=None, batch_size=None):
        if batch_size is None:
            assert len(self.batch_size) == 1,"batch_size should be specified if model compiled with multiple batch sizes"
            batch_size = self.batch_size[0]
        if start_ids is None:
            return position_ids, torch.zeros([batch_size], dtype=torch.int32)
        if not self.neuron_config.use_2d_cache_ids:
            position_ids = position_ids.unsqueeze(0).repeat(batch_size, 1)
            position_ids -= start_ids.unsqueeze(1)
        position_ids.masked_fill_(position_ids < 0, 0)
        return position_ids, start_ids

    def _prepare_pre_layer_params(self, manipulator, pre_layer_parameters):
        extras = []
        for param, dim, allow_pad in pre_layer_parameters:
            if allow_pad and dim is not None:
                if param.shape[dim] % self.tp_degree != 0:
                    size = utils.round_up_to_divisor(param.shape[dim], self.tp_degree)
                    param = utils.pad(param, dim, size)
            extras.append(manipulator.duplicate_or_shard_along(param, dim))
        return extras

    def _build_program(self):
        hlo_modules = dict()
        debug_tensors = dict()
        # for pipeline parallel: only do FullyUnrolled when there is valid ln_lm_head
        if self.unroll == self.num_layers and self.neuron_config.is_valid_lm_head() and not self.neuron_config.is_pp():
            # For page attention we are replacing the batch size with the active blocks,
            # And the n_positions used for the various block sizes corresponded to the max n_position.
            if self.neuron_config.optimized_paged_attention and self.n_active_tokens == 1:
                # do not compile token decoder for chunked prefill
                if self.neuron_config.enable_chunked_prefill:
                    return None
                # Taking block sizes based on n_positions,block_size for bucketing.
                block_sizes = [n_pos * self.neuron_config.continuous_batching.max_num_seqs // self.neuron_config.continuous_batching.block_size for n_pos in self.n_positions_list]
                assert max(block_sizes) <= self.neuron_config.continuous_batching.num_blocks, "Too few blocks allocated, consider increasing gpu_memory_utilization or override"
                # Taking max of n_positions_list for building bucketing for paged attention token gen.
                pa_npos = [max(self.n_positions_list)]
                for npos,block_size in itertools.product(pa_npos, block_sizes):
                    hlo_modules[npos,block_size], debug_tensors[npos,block_size] = self._hlo_fully_unrolled(npos, block_size)
                num_inputs = len(self.inputs_sdim)
                batch_size_for_shared_caches = self.neuron_config.continuous_batching.batch_size_for_shared_caches \
                    if self.neuron_config.continuous_batching else None
                program = DecoderProgramFullyUnrolled(self.neuron_config, self.layers, hlo_modules, debug_tensors, num_inputs, self.tp_degree, pa_npos, block_sizes, self.prefixed_length,
                                                    batch_size_for_shared_caches=batch_size_for_shared_caches, tag=self.tag, on_cpu=self._cpu_compile)
            else:
                for npos,batch_size in itertools.product(self.n_positions_list, self.batch_size):
                    hlo_modules[npos,batch_size], debug_tensors[npos,batch_size] = self._hlo_fully_unrolled(npos, batch_size)
                num_inputs = len(self.inputs_sdim)
                batch_size_for_shared_caches = self.neuron_config.continuous_batching.batch_size_for_shared_caches \
                    if self.neuron_config.continuous_batching else None
                program = DecoderProgramFullyUnrolled(self.neuron_config, self.layers, hlo_modules, debug_tensors, num_inputs, self.tp_degree, self.n_positions_list, self.batch_size, self.prefixed_length,
                                                    batch_size_for_shared_caches=batch_size_for_shared_caches, tag=self.tag, on_cpu=self._cpu_compile)
        else:

            if utils.amp_is_u8(self.amp):
                raise NotImplementedError(f'amp={self.amp} only supports fully unrolled decoder')
            for npos,batch_size in itertools.product(self.n_positions_list, self.batch_size):
                hlo_modules[npos,batch_size], debug_tensors[npos,batch_size] = self._hlo_multi_layer(npos,batch_size)
            ln_lm_head_hlo_modules = [self._hlo_ln_lm_head(batch_size) for batch_size in self.batch_size]
            num_inputs = len(self.inputs_sdim)
            ode_hlo_modules = None
            ode_num_inputs = None
            if self.neuron_config.is_pp():
                program = PipelineParallelProgram(self.neuron_config, self.layers, ode_hlo_modules, ode_num_inputs, hlo_modules, debug_tensors, ln_lm_head_hlo_modules, num_inputs,
                                                    self.num_layers, self.unroll, self.tp_degree,
                                                    self.n_positions_list, self.batch_size, self.prefixed_length, tag=self.tag)
                program.init_pp_sync_programs()
            else:
                if self.neuron_config.on_device_embedding:
                    ode_hlo_modules = [self._hlo_embedding_layer(batch_size) for batch_size in self.batch_size]
                    ode_num_inputs = len(self.ode_sdim)
                batch_size_for_shared_caches = self.neuron_config.continuous_batching.batch_size_for_shared_caches \
                if self.neuron_config.continuous_batching else None
                program = DecoderProgramMultiLayer(self.neuron_config, self.layers, ode_hlo_modules, ode_num_inputs, hlo_modules, debug_tensors, ln_lm_head_hlo_modules, num_inputs,
                                                    self.num_layers, self.unroll, self.tp_degree,
                                                    self.n_positions_list, self.batch_size, self.prefixed_length, batch_size_for_shared_caches=batch_size_for_shared_caches, tag=self.tag, on_cpu=self._cpu_compile)
        return program

    def _hlo_embedding_layer(self, batch_size):

        *_, n_positions = self.n_positions_list
        self.builder.n_positions = n_positions

        def _embedding(scribe):
            amp, quantized, dequantized = utils.parse_amp(self.amp)
            dtype = getattr(scribe, amp)
            (hidden, *tensors), self.ode_sdim = self.inputs_builder(
                    scribe, dtype, self.n_active_tokens, batch_size)
            param_builder = DecoderParameterBuilder(scribe, len(self.ode_sdim))
            pre_layer_params = self._hlo_pre_layer_params(param_builder)
            hidden = self._hlo_embedding(hidden, tensors, pre_layer_params)
            return hidden

        return compiler.compile_py_func(_embedding)

    def _hlo_unroll(self, hidden, tensors, layers_caches, layers_weights, pre_layer_params, lm_head_params):
        last_token_id = tensors[2]
        hidden = self._hlo_embedding(hidden, tensors, pre_layer_params)
        hidden, tensors = self._hlo_pre_layer(hidden, tensors, pre_layer_params)
        hidden, out_caches = self._hlo_layers(hidden, tensors, self.layers, layers_caches, layers_weights, alias_caches=False)
        logits = self.ln_lm_head_builder(hidden, last_token_id, *lm_head_params, return_all_outputs=self.return_all_outputs)
        return logits, out_caches

    def _hlo_eagle_target_unroll(self, hidden, tensors, layers_caches, layers_weights, pre_layer_params, lm_head_params, tree_mask=None, position_ids=None):
        """
        This is the special unroll function that returns the output hidden for EAGLE target model
        """
        last_token_id = tensors[2]
        hidden = self._hlo_embedding(hidden, tensors, pre_layer_params)
        hidden, tensors = self._hlo_pre_layer(hidden, tensors, pre_layer_params, position_ids=position_ids)
        # tensors = last_token_id, pos_embed, cache_ids, start_ids, block_to_seq, mask, active_mask, core_id
        if tree_mask is not None:
            active_mask = tensors[6]
            tensors[6] = hlo.token_tree_attention_mask(tree_mask, active_mask)
        hidden, out_caches = self._hlo_layers(hidden, tensors, self.layers, layers_caches, layers_weights, alias_caches=False)
        logits, hidden = self.ln_lm_head_builder(hidden, last_token_id, *lm_head_params, return_all_outputs=self.return_all_outputs)
        return logits, hidden, out_caches

    def _hlo_eagle_draft_unroll(self, hidden, tensors, layers_caches, layers_weights, pre_layer_params, lm_head_params, tree_mask=None, position_ids=None):
        """
        This is the special unroll function that returns the output hidden for EAGLE draft model.
        """
        last_token_id = tensors[2]
        prev_hidden = tensors[5]
        tensors = tensors[0:5]
        hidden = self._hlo_embedding(hidden, tensors, pre_layer_params)
        hidden = hlo.concatenate([hidden, prev_hidden], 2)
        hidden, *tensors = self.builder.eagle_draft_pre_layer(hidden, *tensors, *pre_layer_params, position_ids=position_ids)
        # tensors = last_token_id, pos_embed, cache_ids, start_ids, block_to_seq, mask, active_mask, core_id
        if tree_mask is not None:
            active_mask = tensors[6]
            tensors[6] = hlo.token_tree_attention_mask(tree_mask, active_mask)
        hidden, out_caches = self._hlo_layers(hidden, tensors, self.layers, layers_caches, layers_weights, alias_caches=False)
        logits, hidden = self.ln_lm_head_builder(hidden, last_token_id, *lm_head_params, return_all_outputs=self.return_all_outputs)
        return logits, hidden, out_caches

    def _hlo_fully_unrolled(self, n_positions, batch_size):

        self.builder.n_positions = n_positions
        if self.neuron_config.optimized_paged_attention and self.n_active_tokens == 1:
            self.builder.num_active_blocks = batch_size
        if self.neuron_config.enable_chunked_prefill and self.n_active_tokens == n_positions:
            self.builder.num_active_blocks = batch_size
            batch_size = 1
        def fully_unrolled(scribe):
            amp, quantized, dequantized = utils.parse_amp(self.amp)
            dtype = getattr(scribe, amp)

            # Page attention parameters
            if self.neuron_config.optimized_paged_attention and self.n_active_tokens == 1:
                (hidden, *tensors), self.inputs_sdim = self.inputs_builder(
                    scribe, dtype, self.n_active_tokens, self.neuron_config.continuous_batching.max_num_seqs)
            elif self.neuron_config.is_eagle_draft:
                (hidden, *tensors), self.inputs_sdim = self.builder.eagle_draft_inputs(
                    scribe, dtype, self.n_active_tokens, batch_size)
            # Create user parameters
            else:
                (hidden, *tensors), self.inputs_sdim = self.inputs_builder(
                    scribe, dtype, self.n_active_tokens, batch_size)
            param_builder = DecoderParameterBuilder(scribe, len(self.inputs_sdim))

            # Create inputs for all weights & caches
            in_caches, layers_weights, pre_layer_params, lm_head_params, generation_params = self._hlo_parameters(n_positions, batch_size, param_builder)

            # Unroll the graph
            if self.neuron_config.is_eagle_target:
                logits, hidden, out_caches = self._hlo_eagle_target_unroll(hidden, tensors, in_caches, layers_weights, pre_layer_params, lm_head_params)
            elif self.neuron_config.is_eagle_draft:
                logits, hidden, out_caches = self._hlo_eagle_draft_unroll(hidden, tensors, in_caches, layers_weights, pre_layer_params, lm_head_params)
            else:
                logits, out_caches = self._hlo_unroll(hidden, tensors, in_caches, layers_weights, pre_layer_params, lm_head_params)
            self._hlo_cache_aliases(in_caches, out_caches)
            output = self._hlo_generation(logits, generation_params)

            # Set the output
            out_caches = itertools.chain(*out_caches)
            if self.neuron_config.log_softmax_scores:
                logits, scores = self._hlo_post_layer(logits)
                outputs = [logits, scores, *out_caches]
            elif self.neuron_config.is_eagle_target:
                outputs = [output, hidden, *out_caches]
            else:
                outputs = [output, *out_caches]

            # Filter out the None's in outputs
            outputs = [o for o in outputs if o is not None]
            return outputs

        debug_tensors = {}
        patched_func = global_debugger.populate_debug_tensors(debug_tensors)(fully_unrolled)
        return compiler.compile_py_func(patched_func), debug_tensors

    def _hlo_multi_layer(self, n_positions, batch_size):

        self.builder.n_positions = n_positions

        def multi_layer(scribe):
            # TODO: Add support for dynamic generation to multi layer
            dtype = getattr(scribe, self.amp)
            (hidden, *tensors), self.inputs_sdim = self.inputs_builder(
                scribe, dtype, self.n_active_tokens, batch_size)
            param_builder = DecoderParameterBuilder(scribe, len(self.inputs_sdim))
            # use the first `unroll` layers to build the HLO -- assuming all layers are same
            layers = self.layers[:self.unroll]
            layers_caches, layers_weights = self._hlo_layers_params(param_builder, layers, n_positions, batch_size)
            pre_layer_params = self._hlo_pre_layer_params(param_builder)
            hidden, tensors = self._hlo_pre_layer(hidden, tensors, pre_layer_params)
            out_hidden, out_caches = self._hlo_layers(hidden, tensors, layers, layers_caches, layers_weights)
            out_hidden.set_alias_to(hidden)
            out_caches = itertools.chain(*out_caches)
            outputs = [out_hidden, *out_caches]
            # Filter out the None's in outputs
            outputs = [o for o in outputs if o is not None]
            return outputs

        # NOTE: Forcefully disable on device embedding when setting up multilayer
        #       layers to ensure that hidden size is uniform on input/output
        prior = self.neuron_config.on_device_embedding
        self.neuron_config.on_device_embedding = False
        debug_tensors = {}
        patched_func = global_debugger.populate_debug_tensors(debug_tensors)(multi_layer)
        result = compiler.compile_py_func(patched_func)
        self.neuron_config.on_device_embedding = prior
        return result, debug_tensors

    def _hlo_parameters(self, n_positions, batch_size, param_builder):
        layers_caches, layers_weights = self._hlo_layers_params(param_builder, self.layers, n_positions, batch_size)
        pre_layer_params = self._hlo_pre_layer_params(param_builder)
        lm_head_params = self._hlo_lm_head_params(param_builder)
        generation_params = self._hlo_generation_params(param_builder)
        return layers_caches, layers_weights, pre_layer_params, lm_head_params, generation_params

    def all_parameters(self, n_positions, batch_size):
        """
        Get all the parameters for the current model.

        NOTE: It is extremely important that these tensors are returned in the
              same order as the parameters returned in the _hlo_parameters
              function. If this is not done correctly, the HLO parameter and
              the corresponding weight tensor cannot be assocated.
        """
        parameters = list()

        # Layer caches
        for layer in self.layers:
            for cache in layer.attn_k_cache[batch_size], layer.attn_v_cache[batch_size]:
                parameters.append(cache)

        # Layer weights
        for layer in self.layers:
            parameters.extend(layer.all_parameters())

        # Prelayer parameters
        parameters.extend(self.pre_layer_parameters)

        # LM head parameters
        parameters.append(self.ln_f_weight)
        parameters.append(self.ln_f_bias)
        parameters.append(self.lm_head_weight)
        parameters.append(self.lm_head_bias)

        # Generation parameters
        parameters.append(self.logits_indices)
        parameters.append(self.top_k)
        parameters.append(self.top_p)
        parameters.append(self.temperature)
        parameters.append(self.top_p_min_tokens)

        return parameters

    def save_presharded_weights(self, directory):
        save_dict = {}
        for key, value in self.__dict__.items():
            if isinstance(value, torch.Tensor) and value.device.type == 'xla':
                cpu_tensor = ops.parallel_cpu(value)
                for i in range(len(cpu_tensor)):
                    save_dict[f'{key}*{i}'] = cpu_tensor[i]

        for i, param in enumerate(self.pre_layer_parameters):
            cpu_tensor = ops.parallel_cpu(param)
            for j in range(len(cpu_tensor)):
                save_dict[f'pre_layer_parameter{i}*{j}'] = cpu_tensor[j]

        save_file(save_dict, os.path.join(directory, 'DecoderLMHead.safetensors'))

    def load_presharded_weights(self, ps_dir):
        with safe_open(os.path.join(ps_dir, f"DecoderLMHead.safetensors"), framework='pt') as f:
            lm_head_attr_names = get_attribute_names(f)
            presharded_weights_to_neuron(f, self, lm_head_attr_names)

        self.format_pre_layer_parameters()
        self.format_ln_lm_head_params_and_generation_inputs()
        self.finish_program_setup()

    def format_pre_layer_parameters(self):
        # deal with the pre_layer_parameters (decoder_lm_head only)
        i = 0
        self.pre_layer_parameters = []
        while True:
            pre_layer_parameter = getattr(self, f"pre_layer_parameter{i}", None)
            if pre_layer_parameter is None:
                break
            self.pre_layer_parameters.append(pre_layer_parameter)
            i+=1

    def format_ln_lm_head_params_and_generation_inputs(self):
        ln_lm_head_params = [self.ln_f_weight, self.ln_f_bias, self.lm_head_weight, self.lm_head_bias, self.logits_indices]
        generation_inputs = [self.top_k, self.top_p, self.temperature, self.top_p_min_tokens]
        ln_lm_head_params = [param for param in ln_lm_head_params if param is not None]
        generation_inputs = [param for param in generation_inputs if param is not None]
        self.generation_inputs = generation_inputs
        self.ln_lm_head_params = ln_lm_head_params + generation_inputs

    def valid_parameters(self, n_positions, batch_size):
        parameters = self.all_parameters(n_positions, batch_size)
        return [par for par in parameters if par is not None]

    def _hlo_pre_layer_params(self, param_builder):
        params = []
        for param in self.pre_layer_parameters:
            param = param_builder.from_tensor(param)
            param = hlo.transfer_with_static_ring(param)
            params.append(param)
        return params

    def _hlo_pre_layer(self, hidden, tensors, params, position_ids=None):
        if self.pre_layer_builder is not None:
            if self.neuron_config.is_eagle_draft or self.neuron_config.is_eagle_target:
                (hidden, *tensors) = self.pre_layer_builder(hidden, *tensors, *params, position_ids=position_ids)
            else:
                (hidden, *tensors) = self.pre_layer_builder(hidden, *tensors, *params)
        return hidden, tensors

    def _hlo_embedding(self, hidden, tensors, params):
        # Only insert embedding operation when on-device embedding is being used
        if not self.neuron_config.on_device_embedding:
            return hidden

        assert self.embedding_builder is not None, (
            f"On-device embedding may only be used on models which provide this functionality"
        )
        hidden = self.embedding_builder(hidden, *tensors, *params)
        return hidden

    def _hlo_layers_params(self, param_builder, layers, n_positions, batch_size):
        layers_caches = []
        dim_size = {1: n_positions} if self.bsh_cache_layout else {0: n_positions}
        if self.neuron_config.continuous_batching:
            batch_size = self.neuron_config.continuous_batching.max_num_seqs
            if self.neuron_config.paged_attention:
                block_size = self.neuron_config.continuous_batching.block_size
                dim_size = {1: block_size} if self.bsh_cache_layout else {0: block_size}
        for layer in layers:
            layer_caches = []
            if self.neuron_config.shard_over_sequence:
                if self.neuron_config.enable_chunked_prefill:
                    dim_size = {k:block_size//layer.kv_replication for k in dim_size.keys()}
                else:
                    dim_size = {k:n_positions//layer.kv_replication for k in dim_size.keys()}
            for cache in layer.attn_k_cache[batch_size], layer.attn_v_cache[batch_size]:
                par = param_builder.from_tensor(cache, dim_size=dim_size)
                layer_caches.append(par)
            layers_caches.append(layer_caches)
        layers_weights = []
        for layer in layers:
            layer_weights = [param_builder.from_tensor(weight) for weight in layer.all_parameters()]
            layers_weights.append(layer_weights)
        return layers_caches, layers_weights

    def _hlo_layers(self, hidden, tensors, layers, layers_caches, layers_weights, alias_caches=True):
        output_caches = []
        for idx, (layer, caches, weights) in enumerate(zip(layers, layers_caches, layers_weights)):
            in_caches = [maybe_transfer_with_static_ring(cache) for cache in caches]
            weights = [maybe_transfer_with_static_ring(weight) for weight in weights]
            weights = layer.hlo_maybe_dequantize_weights(weights)
            is_first_last_layer = True if idx == 0 or idx == len(layers) - 1 else False
            if isinstance(self.layer_builder.__self__, LlamaForSamplingNoEmbeddingHlo):
                # Positional information is needed for fused residual adds in kernels in Llama 3
                hidden, *out_caches = self.layer_builder(hidden, *tensors, *in_caches, *weights, is_first_last_layer=is_first_last_layer)
            else:
                hidden, *out_caches = self.layer_builder(hidden, *tensors, *in_caches, *weights)
            output_caches.append(out_caches)

        if alias_caches:
            self._hlo_cache_aliases(layers_caches, output_caches)

        return hidden, output_caches

    def _hlo_cache_aliases(self, in_caches, out_caches):
        assert len(in_caches) == len(out_caches)
        for src, dst in zip(itertools.chain(*in_caches), itertools.chain(*out_caches)):
            if dst is not None:
                assert src is not None, "out_cache must alias with a valid cache!"
                dst.set_alias_to(src, must=True)

    def _hlo_lm_head_params(self, param_builder):
        ln_f_weight = param_builder.from_tensor(self.ln_f_weight)
        ln_f_bias = param_builder.from_tensor(self.ln_f_bias)
        head_weight = param_builder.from_tensor(self.lm_head_weight)
        head_bias = param_builder.from_tensor(self.lm_head_bias)
        ln_f_weight = maybe_transfer_with_static_ring(ln_f_weight)
        ln_f_bias = maybe_transfer_with_static_ring(ln_f_bias)
        head_weight = maybe_transfer_with_static_ring(head_weight)
        head_bias = maybe_transfer_with_static_ring(head_bias)
        return ln_f_weight, ln_f_bias, head_weight, head_bias

    def _hlo_ln_lm_head(self, batch_size):
        hidden_sizes = []
        *_, n_positions = self.n_positions_list
        self.builder.n_positions = n_positions

        def capture_hidden_sizes(scribe):
            dtype = getattr(scribe, self.amp)
            (hidden, *_), _ = self.inputs_builder(
                scribe, dtype, self.n_active_tokens, batch_size)
            hidden_sizes.clear()
            hidden_sizes.extend(hidden.sizes)
            return hidden

        # NOTE: Forcefully disable on device embedding when setting up multilayer
        #       layers to ensure that hidden size is uniform on input/output
        prior = self.neuron_config.on_device_embedding
        self.neuron_config.on_device_embedding = False
        compiler.compile_py_func(capture_hidden_sizes)
        self.neuron_config.on_device_embedding = prior

        def ln_lm_head(scribe):
            dtype = getattr(scribe, self.amp)
            hidden = dtype[tuple(hidden_sizes)].Parameter(parameter_number=0)
            if self.neuron_config and self.neuron_config.lhs_aligned:
                next_tok_id = scribe.s32[batch_size].Parameter(parameter_number=1)
            else:
                next_tok_id = scribe.s32[1].Parameter(parameter_number=1)
            param_builder = DecoderParameterBuilder(scribe, 2)
            ln_f_weight, ln_f_bias, head_weight, head_bias = self._hlo_lm_head_params(param_builder)
            gneration_params = self._hlo_generation_params(param_builder)
            logits = self.ln_lm_head_builder(hidden, next_tok_id, ln_f_weight, ln_f_bias, head_weight, head_bias, return_all_outputs=self.return_all_outputs)
            output = self._hlo_generation(logits, gneration_params)
            if self.neuron_config.log_softmax_scores:
                logits, scores = self._hlo_post_layer(logits)
                outputs = [logits, scores]
                root_shapes = [shape.dtype[shape.sizes] for shape in outputs]
                return scribe.tuple(*root_shapes).Tuple(*outputs)
            return output

        return compiler.compile_py_func(ln_lm_head)

    def _hlo_post_layer(self, logits):
        return self.post_layer_builder(logits)

    def _hlo_generation_params(self, param_builder):
        logits_indices = param_builder.from_tensor(self.logits_indices)
        logits_indices = maybe_transfer_with_static_ring(logits_indices)
        params = [logits_indices]
        if self.neuron_config.on_device_generation is not None and self.neuron_config.on_device_generation.dynamic:
            for param in self.generation_inputs:
                param = param_builder.from_tensor(param)
                param = maybe_transfer_with_static_ring(param)
                params.append(param)
        return params

    def _hlo_generation(self, logits, params, early_return=False, return_probs=False):
        generation_config = self.neuron_config.on_device_generation
        if generation_config is None:
            return logits
        logits_indices, *dynamic_generation_params = params
        if generation_config.dynamic:
            top_k, top_p, temperature, top_p_min_tokens  = dynamic_generation_params
            self.neuron_config.on_device_generation.top_k = top_k
            self.neuron_config.on_device_generation.top_p = top_p
            self.neuron_config.on_device_generation.temperature = temperature
            self.neuron_config.on_device_generation.top_p_min_tokens = top_p_min_tokens
        return generation.generate(
            logits,
            logits_indices,
            config=generation_config,
            tp_degree=self.tp_degree,
            early_return=early_return,
            return_probs=return_probs,
        )

    # Mainly used for serialization purposes.
    # Defines how to access all the kernels.
    def get_all_kernels(self):
        return self.program.get_kernels()

    def validate_generation_configs(self, generation_config: GenerationConfig):
        current_generation_config = self.neuron_config.on_device_generation
        if  current_generation_config:
            assert generation_config.per_batch_line == current_generation_config.per_batch_line, f"Invalid new generation config. \n \
            Recieved new generation config with per_batch_line = {generation_config.per_batch_line},\n \
            while current generation config has per_batch_line = {current_generation_config.per_batch_line}"

        batch_size = self.batch_size if isinstance(self.batch_size,int) else self.batch_size[0]

        if generation_config.per_batch_line:
            if not isinstance(generation_config.top_k, list):
                generation_config.top_k = [generation_config.top_k] * batch_size

            if not isinstance(generation_config.top_p, list):
                generation_config.top_p = [generation_config.top_p] * batch_size

            if not isinstance(generation_config.temperature, list):
                generation_config.temperature = [generation_config.temperature] * batch_size

            if not isinstance(generation_config.top_p_min_tokens, list):
                generation_config.top_p_min_tokens = [generation_config.top_p_min_tokens] * batch_size

            # check all sampling parameters lists are of same size

            assert len(generation_config.top_k) \
                   == len(generation_config.top_p) \
                   == len(generation_config.temperature)  \
                   == len(generation_config.top_p_min_tokens)  \
                   == batch_size, f"For per batch-line sampling, sampling parameters top_k, top_p, \
                   top_p_min_tokens, temperature must be of same size as bach_size. \n \
                   Recieved len(top_k):  {len(generation_config.top_k)} \n \
                   len(top_p): {len(generation_config.top_p)} \n \
                   len(top_p_min_tokens): {len(generation_config.top_p_min_tokens)} \n \
                   len(temperature): {len(generation_config.temperature)} \n \
                   batch_size: {batch_size}"
        else:
            assert not isinstance(generation_config.top_k, list) \
               and not isinstance(generation_config.top_p, list) \
               and not isinstance(generation_config.top_p_min_tokens, list) \
               and not isinstance(generation_config.temperature, list) \
               , f"Sampling parameters cannot be of type list when per_batch_line = False"

    def update_generation_config(self, generation_config: config.GenerationConfig):
        self.validate_generation_configs(generation_config)
        num_cores = self.neuron_config.get_local_tp(self.tp_degree)
        duplicate = lambda tensor: [torch.tensor(tensor) for _ in range(num_cores)]
        ops.parallel_write(self.top_k, duplicate(generation_config.top_k))
        ops.parallel_write(self.top_p, duplicate(generation_config.top_p))
        ops.parallel_write(self.temperature, duplicate(generation_config.temperature))
        ops.parallel_write(self.top_p_min_tokens, duplicate(generation_config.top_p_min_tokens))


def read_n_position(hlo_module, num_inputs):
    return hlo_module.host_program_shape.parameters[num_inputs].dimensions[0]


def read_n_active_tokens(hlo_module):
    return hlo_module.host_program_shape.parameters[0].dimensions[1]


def read_batch_size(hlo_module):
    return hlo_module.host_program_shape.parameters[0].dimensions[0]


def maybe_transfer_with_static_ring(shape):
    if shape is None:
        return None
    return hlo.transfer_with_static_ring(shape)

### This is a place-holder to indicate what we want this to look like
### This is not currently utilized anywhere
### TO-DO: Modify/integrate these to have decoder-specific forward functionality
class SpeculativeDecoder(torch.nn.Module):
    def forward(self, hidden, *args):
        hidden = hidden.transpose(0, -1).contiguous()
        logits = self.decoder_lm_head(hidden, *args)
        logits = logits.to(torch.float32)
        logits = logits[:self.config.vocab_size, -self.n_active_tokens:, :]
        logits = logits.transpose(0, 1)
        logits=logits.transpose(1, 2)
        return logits

### This is a place-holder to indicate what we want this to look like
### This is not currently utilized anywhere
### TO-DO: Modify/integrate these to have decoder-specific forward functionality
class ContextDecoder(torch.nn.Module):

    def context(self, hidden, cache_ids, start_ids, last_token_id):
        """A helper to process context (prompt)
        1) if there is available context encoding model (infered from self.context_buckets)
            - when context_length >= estimate, slice the context up to estimate,
                and call context encoding model
            - when context_length < estimate, skip and fall back to serial token generation model

            and mark `current` accordingly

        2) process the left over tokens accroding to `current`
            - if there is no context encoding model, simply do serial token generation for context
        """
        context_length = hidden.shape[1]
        # batch_size is in dim 2 because of the transpose taken in _forward function
        batch_size = hidden.shape[2]

        if self.is_fid:
            # Fusion-In-Decoder context encoding
            fused_context_length = hidden.shape[1]
            context_length = fused_context_length // self.batch_size

        current = 0

        estimate = bucket.find(self.context_buckets, context_length)


        if estimate is not None:
            hidden_context = hidden
            cache_context = cache_ids

            # Slice context that when it is too large
            if context_length > estimate:
                current = estimate
                hidden_context = hidden[:, :estimate]
                cache_context = cache_ids[:estimate]

            # Cannot use context encoding for a context that is too small. This
            # is because the caller must be aware of the cache-ids/start-ids
            # used.
            elif context_length < estimate:
                raise ValueError(f"context_length ({context_length}) shouldn't be smaller than estimate ({estimate})")

            # Directly pass input to the context network when exactly sized
            else:
                current = estimate

            if current == estimate:
                model = self.decoder_lm_head_for_context[estimate, batch_size]
                logits = model(hidden_context, cache_context, start_ids, last_token_id)

        for i in range(current, context_length):
            cache_ids = torch.as_tensor([i], dtype=torch.int32)
            hidden_slice = hidden[:, i:i+1].contiguous()
            logits = self.decoder_lm_head(hidden_slice, cache_ids, start_ids, last_token_id)

        if self.is_fid:
            logits[:] = float('-inf')
            logits[self.bos_token_id] = 1.0

        return logits

    def forward(self, hidden, cache_ids=None, start_ids=None, last_token_id=None):
        hidden = hidden.transpose(0, -1).contiguous()
        logits = self.context(hidden, cache_ids, start_ids, last_token_id)
        logits = logits.to(torch.float32)
        logits = logits[:self.config.vocab_size, -1, :]
        logits = logits.transpose(0, 1)
        return logits

### This is a place-holder to indicate what we want this to look like
### This is not currently utilized anywhere
### TO-DO: Modify/integrate these to have decoder-specific forward functionality
class TokenDecoder(torch.nn.Module):
    def forward(self, hidden, *args):
        hidden = hidden.transpose(0, -1).contiguous()
        logits = TokenDecoder.forward(hidden, *args)
        logits = logits.to(torch.float32)
        logits = logits[:self.config.vocab_size, -1, :]
        logits = logits.transpose(0, 1)
        return logits


class MaybePadder:

    def __init__(self, size, padding="end", split_size=None, interleaved_factor=None) -> None:
        self.split_size = split_size
        self.size = size
        self.padding = padding
        self.interleaved_factor = interleaved_factor

    def __call__(self, weight, dim):
        if self.padding == "end":
            return utils.pad(weight, dim, self.size, left=False)
        else:
            if weight is None:
                return weight
            assert self.padding == "interleaved", f"Invalid padding mode {self.padding}"
            assert self.interleaved_factor, f"interleaved_factor is not provided"
            # when split_size is set, we first split the target weight at dim
            # into (split_size x ?), for example, to do interleaved padding on of KV weight
            # we first need to reshape it into (hidden, num_kv_head, d_head)
            # and then apply interleaved padding on num_kv_head
            weight_shapes = list(weight.shape)

            padded_shape = weight_shapes.copy()
            padded_shape[dim] = self.size

            new_size = self.size
            if self.split_size:
                assert weight_shapes[dim] % self.split_size == 0, f"shape on dim_{dim} {weight_shapes[dim]} cannot be evenly divisible by provided split_size {self.split_size}"
                new_shape = weight_shapes[:dim] + [self.split_size] + [weight_shapes[dim] // self.split_size] + weight_shapes[dim+1:]
                weight = weight.view(new_shape)
                new_size = self.size // (weight_shapes[dim] // self.split_size)
            res = utils.pad_interleaved(weight, dim, new_size,
                weight.shape[dim]//self.interleaved_factor, (new_size-weight.shape[dim])//self.interleaved_factor)
            return res.view(padded_shape)

class DecoderLayer(torch.nn.Module):

    def __init__(self, tp_degree, n_positions, batch_size, attention_head_size, n_head, amp,
                 n_kv_head=0, neuron_config=None, allow_pad=False, n_active_tokens=None, layer_num=None,
                 is_unit_scale=False):
        super().__init__()
        self.pre_attn_ln_weight = None
        self.pre_attn_ln_bias = None
        self.fused_pre_attn_ln_qkv_weight = None
        self.attn_q_weight = None
        self.attn_q_scales = None
        self.attn_q_bias = None
        self.attn_k_weight = None
        self.attn_k_scales = None
        self.attn_k_bias = None
        self.attn_v_weight = None
        self.attn_v_scales = None
        self.attn_v_bias = None
        self.attn_out_weight = None
        self.attn_out_scales = None
        self.attn_out_bias = None
        self.post_attn_ln_weight = None
        self.post_attn_ln_bias = None
        self.pre_mlp_ln_weight = None
        self.pre_mlp_ln_bias = None
        self.mlp_in_weight = None
        self.mlp_in_scales = None
        self.mlp_in_bias = None
        self.mlp_out_weight = None
        self.mlp_out_scales = None
        self.mlp_out_bias = None
        self.post_mlp_ln_weight = None
        self.post_mlp_ln_bias = None
        self.attn_q_min = None
        self.attn_q_max = None
        self.attn_k_min = None
        self.attn_k_max = None
        self.attn_v_min = None
        self.attn_v_max = None
        self.attn_out_min = None
        self.attn_out_max = None
        self.mlp_in_min = None
        self.mlp_in_max = None
        self.mlp_out_min = None
        self.mlp_out_max = None
        # Create KV caches for each batch_size
        self.attn_k_cache = dict()
        self.attn_v_cache = dict()
        self.cache_shape = dict()
        self.tp_degree = tp_degree
        self.n_positions = n_positions
        self.n_head = n_head
        self.n_head_padded = None
        self.n_kv_head = n_kv_head
        self.batch_sizes = batch_size
        self.attention_head_size = attention_head_size  # TODO: rename this to size_per_head
        self.tp_degree = tp_degree
        self.amp = amp
        dtype, _, _ = utils.parse_amp(amp)
        if neuron_config and neuron_config.kv_cache_quant:
            self.cache_dtype = dtypes.to_torch_dtype(neuron_config.kv_cache_quant.quant_dtype)
        else:
            self.cache_dtype = dtypes.to_torch_dtype(dtype)
        self.neuron_config = NeuronConfig() if neuron_config is None else neuron_config
        self.extra_parameters = []
        self.allow_pad = allow_pad
        self.attn_out_sharding = 0
        self.attn_out_transposed = True
        self.mlp_out_sharding = 0
        self.mlp_out_transposed = True
        self.kv_replication  = 1 # default value to denote weight replication factor
        self.layer_num = layer_num
        self.is_unit_scale = is_unit_scale
        self._cpu_compile = False

    def add_parameter(self, param, sharding=None, allow_pad=False, allow_quantize=False,
                      out_feature_dim=1, allow_transform=False):
        self.extra_parameters.append((param, sharding, allow_pad, allow_quantize, out_feature_dim, allow_transform))

    def add_pre_attention_layer_norm(self, weight, bias):
        self.pre_attn_ln_weight = weight
        self.pre_attn_ln_bias = bias

    def add_attention_query(self, weight, bias):
        self.attn_q_weight = weight
        self.attn_q_bias = bias

    def add_attention_key(self, weight, bias):
        self.attn_k_weight = weight
        self.attn_k_bias = bias

    def add_attention_value(self, weight, bias):
        self.attn_v_weight = weight
        self.attn_v_bias = bias

    def add_attention_output(self, weight, bias, sharding=0, transposed=True, out_feature_dim=None, contract_dims=None, pad=True):
        self.attn_out_weight = weight
        self.attn_out_bias = bias
        self.attn_out_sharding = sharding
        self.attn_out_transposed = transposed
        self.attn_out_feature_dim = out_feature_dim
        self.attn_out_contract_dims = contract_dims
        self.attn_out_pad = pad

    def add_post_attention_layer_norm(self, weight, bias):
        self.post_attn_ln_weight = weight
        self.post_attn_ln_bias = bias

    def add_pre_mlp_layer_norm(self, weight, bias):
        self.pre_mlp_ln_weight = weight
        self.pre_mlp_ln_bias = bias

    def add_mlp_input(self, weight, bias):
        self.mlp_in_weight = weight
        self.mlp_in_bias = bias

    def add_mlp_output(self, weight, bias, sharding=0, transposed=True):
        self.mlp_out_weight = weight
        self.mlp_out_bias = bias
        self.mlp_out_sharding = sharding
        self.mlp_out_transposed = transposed

    def add_post_mlp_layer_norm(self, weight, bias):
        self.post_mlp_ln_weight = weight
        self.post_mlp_ln_bias = bias

    def to_neuron(self):
        # If we allow padding then we need to pad non-sharded QKV weight dimensions
        self.neuron_config.n_head_padded = self.n_head
        if self.allow_pad:
            # Hidden size padding
            _, hidden_size = self.attn_q_weight.shape
            n_heads = hidden_size // self.attention_head_size

            n_head_padded, n_kv_heads_padded = utils.get_qkv_padding(n_heads, self.n_kv_head, self.tp_degree, self.neuron_config)
            self.n_head_padded = n_head_padded
            self.neuron_config.n_head_padded = self.n_head_padded

            hidden_size_padded = hidden_size_padded_qkv = n_head_padded * self.attention_head_size
            if self.neuron_config.group_query_attention == constants.GQA.ALL_GATHER_HEADS:
                qkv_maybe_pad = attn_out_maybe_pad = MaybePadder(hidden_size_padded,
                                        padding="interleaved",
                                        split_size=n_heads, interleaved_factor=self.n_kv_head)
            else:
                if self.neuron_config.qkv_tiling:
                    hidden_size_padded_qkv = \
                        utils.round_up_to_divisor(hidden_size_padded // self.tp_degree,
                                            constants.TILE_SIZE) * self.tp_degree
                qkv_maybe_pad = MaybePadder(hidden_size_padded_qkv)
                attn_out_maybe_pad = MaybePadder(hidden_size_padded)

                # Adjust padding strategy if we can use less K/V replication
                # with interleaved padding.
                extra_heads = n_head_padded - n_heads
                if (
                    self.n_head != self.n_kv_head
                    and self.neuron_config.group_query_attention == constants.GQA.REPLICATED_HEADS
                    and self.tp_degree % self.n_kv_head == 0
                    and extra_heads % self.n_kv_head == 0
                    and extra_heads > 0
                ):
                    qkv_maybe_pad = MaybePadder(
                        hidden_size_padded_qkv,
                        padding="interleaved",
                        split_size=n_heads, interleaved_factor=self.n_kv_head
                    )
                    attn_out_maybe_pad = MaybePadder(
                        hidden_size_padded,
                        padding="interleaved",
                        split_size=n_heads, interleaved_factor=self.n_kv_head
                    )

            self.attn_q_weight = qkv_maybe_pad(self.attn_q_weight, dim=1)
            self.attn_q_bias = qkv_maybe_pad(self.attn_q_bias, dim=0)

            node_interleaving = False
            if self.neuron_config.shard_over_sequence:
                node_interleaving = utils.is_attn_node_interleaved(n_heads, self.n_kv_head, self.tp_degree)
                if node_interleaving:
                    replica_groups = utils.build_replica_groups(group_size=self.kv_replication,
                                                                 num_groups=self.tp_degree//self.kv_replication, interleave=True)
                    warnings.warn(f"[SOS] qkv node_inerleaving enabled with replica {replica_groups}")

            if n_kv_heads_padded != self.n_kv_head:

                if n_kv_heads_padded % self.n_kv_head == 0:
                    ratio = int(n_kv_heads_padded / self.n_kv_head)
                else:
                    ratio = int((n_kv_heads_padded - extra_heads) / self.n_kv_head)

                # Full replication: replicate KV heads to original Q heads and then do padding
                if n_head_padded == n_kv_heads_padded and extra_heads > 0:
                    ratio = int((n_kv_heads_padded - extra_heads) / self.n_kv_head)

                def repeat(weight):
                    if weight is None:
                        return weight
                    shape = weight.shape[:-1] + (self.n_kv_head, weight.shape[-1] // self.n_kv_head)
                    weight = weight.view(shape)
                    weight = torch.repeat_interleave(weight, repeats=ratio, dim=-2)
                    shape = weight.shape[:-2] + (weight.shape[-1] * weight.shape[-2],)
                    return weight.view(shape)

                def pad_kv_no_repeat(weight, pad_size):
                    if weight is None:
                        return weight
                    shape = weight.shape[:-1] + (self.n_kv_head, weight.shape[-1] // self.n_kv_head)
                    weight = weight.view(shape)
                    weight = torch.nn.functional.pad(weight, (0, 0, 0, pad_size))
                    shape = weight.shape[:-2] + (weight.shape[-1] * weight.shape[-2],)
                    return weight.view(shape)

                if ratio == 0:
                    # in case no replication is needed, pad kv based on n_kv_heads_padded calculated above
                    self.attn_k_weight = pad_kv_no_repeat(self.attn_k_weight, n_kv_heads_padded - self.n_kv_head)
                    self.attn_v_weight = pad_kv_no_repeat(self.attn_v_weight, n_kv_heads_padded - self.n_kv_head)
                    self.attn_k_bias = pad_kv_no_repeat(self.attn_k_bias, n_kv_heads_padded - self.n_kv_head)
                    self.attn_v_bias = pad_kv_no_repeat(self.attn_v_bias, n_kv_heads_padded - self.n_kv_head)
                    self.n_kv_head = n_kv_heads_padded
                else:
                    self.attn_k_weight = repeat(self.attn_k_weight)
                    self.attn_v_weight = repeat(self.attn_v_weight)
                    self.attn_k_bias = repeat(self.attn_k_bias)
                    self.attn_v_bias = repeat(self.attn_v_bias)
                    self.n_kv_head *= ratio
                self.kv_replication = ratio
                # FIXME: As a workaround to get kv_replication info (after padding) in HLO construction
                self.neuron_config.kv_replication = self.kv_replication

            if self.n_head == self.n_kv_head:
                self.attn_k_weight = qkv_maybe_pad(self.attn_k_weight, dim=1)
                self.attn_k_bias = qkv_maybe_pad(self.attn_k_bias, dim=0)

                self.attn_v_weight = qkv_maybe_pad(self.attn_v_weight, dim=1)
                self.attn_v_bias = qkv_maybe_pad(self.attn_v_bias, dim=0)

            def interleave_by_node(tensor, dim, n_nodes):
                if tensor is None:
                    return tensor
                shape = tensor.shape
                assert shape[dim] % n_nodes == 0 , f"cannot interleave across node for tensor shape {shape}"\
                                                     f" and n_nodes {n_nodes}"
                stride = constants.TRN1_WORLD_SIZE
                view_shape = (stride, shape[0]//stride,shape[1]) if dim == 0 else (shape[0],stride, shape[1]//stride)
                return (tensor.reshape(view_shape).permute(1, 0, 2).reshape(shape) if dim == 0
                                else tensor.reshape(view_shape).permute(0, 2, 1).reshape(shape))

            if self.neuron_config.shard_over_sequence and self.neuron_config.duplicate_q_weight_sos:
                q_weight = self.attn_q_weight
                n_kv_head = n_kv_heads_padded // self.kv_replication
                q_weight = q_weight.reshape(-1, n_kv_head, self.n_head_padded // n_kv_head, self.attn_q_weight.shape[-1] // self.n_head_padded)
                q_weight = q_weight.repeat(1, 1, self.kv_replication, 1)
                self.attn_q_weight = q_weight.reshape(-1, self.kv_replication*self.attn_q_weight.shape[-1])

                if self.attn_q_bias is not None:
                    q_bias = self.attn_q_bias
                    q_bias = q_bias.reshape(n_kv_heads_padded, self.n_head_padded // n_kv_heads_padded, self.attn_q_bias.shape[-1] // self.n_head_padded)
                    q_bias = q_bias.repeat(1, self.kv_replication, 1)
                    self.attn_q_bias = q_bias.reshape(-1)

            if self.neuron_config.topo_aware_sharding:
                row_ranks = torch.arange(16).reshape(4, 4, 1)
                col_ranks = 16 + torch.transpose(torch.arange(16).reshape(4, 4, 1), 0, 1)
                ranks = torch.concat([row_ranks, col_ranks], dim=2).reshape(-1)

                n_head_per_core = n_heads // 32
                q_shape = (self.attn_k_weight.shape[0], 32, n_head_per_core, self.attention_head_size)
                kv_shape = (self.attn_v_weight.shape[0], 32, self.attention_head_size)
                wo_shape = (self.attn_out_weight.shape[0], 32, n_head_per_core * self.attention_head_size)
                if self.neuron_config.duplicate_q_weight_sos:
                    n_head_per_core = n_head_per_core * 4
                    q_shape = (self.attn_k_weight.shape[0], 32, n_head_per_core, self.attention_head_size)
                self.attn_q_weight = self.attn_q_weight.reshape(q_shape)[:, ranks, :, :].reshape(self.attn_q_weight.shape)
                self.attn_k_weight = self.attn_k_weight.reshape(kv_shape)[:, ranks, :].reshape(self.attn_k_weight.shape)
                self.attn_v_weight = self.attn_v_weight.reshape(kv_shape)[:, ranks, :].reshape(self.attn_v_weight.shape)
                self.attn_out_weight = self.attn_out_weight.reshape(wo_shape)[:, ranks, :].reshape(self.attn_out_weight.shape)

            if node_interleaving:
                n_nodes = self.tp_degree // constants.TRN1_WORLD_SIZE
                self.attn_q_weight = interleave_by_node(self.attn_q_weight, dim=1, n_nodes=n_nodes)
                self.attn_k_weight = interleave_by_node(self.attn_k_weight, dim=1, n_nodes=n_nodes)
                self.attn_v_weight = interleave_by_node(self.attn_v_weight, dim=1, n_nodes=n_nodes)
                self.attn_q_bias = interleave_by_node(self.attn_q_bias, dim=0, n_nodes=n_nodes)
                self.attn_k_bias = interleave_by_node(self.attn_k_bias, dim=0, n_nodes=n_nodes)
                self.attn_v_bias = interleave_by_node(self.attn_v_bias, dim=0, n_nodes=n_nodes)


            if self.neuron_config and self.neuron_config.fuse_qkv:
                fused_qkv_weight = interleave_qkv(self.attn_q_weight, self.attn_k_weight, self.attn_v_weight, self.tp_degree, dim=1)
                if self.attn_q_bias is not None:
                    fused_qkv_bias = interleave_qkv(self.attn_q_bias, self.attn_k_bias, self.attn_v_bias, self.tp_degree, dim=0)
                else:
                    fused_qkv_bias = None
                fused_qkv_scales = None
                self.attn_k_weight = None
                self.attn_k_scales = None
                self.attn_k_bias = None
                self.attn_v_weight = None
                self.attn_v_scales = None
                self.attn_v_bias = None
            if self.attn_out_pad:
                self.attn_out_weight = attn_out_maybe_pad(self.attn_out_weight, dim=self.attn_out_sharding)
            if node_interleaving:
                self.attn_out_weight = interleave_by_node(self.attn_out_weight, dim=self.attn_out_sharding, n_nodes=n_nodes)
            # Intermediate MLP layer padding
            if self.mlp_in_weight is not None:
                _, intermediate_size = self.mlp_in_weight.shape
                intermediate_size_padded = utils.round_up_to_divisor(intermediate_size, self.tp_degree)
                if self.neuron_config.weight_tiling:
                    intermediate_size_padded = \
                        utils.round_up_to_divisor(intermediate_size // self.tp_degree,
                                                  constants.TILE_SIZE) * self.tp_degree
                maybe_pad = MaybePadder(intermediate_size_padded)

                self.mlp_in_weight = maybe_pad(self.mlp_in_weight, dim=1)
                self.mlp_in_bias = maybe_pad(self.mlp_in_bias, dim=0)
                if self.neuron_config.fuse_mlp:
                    intermediate_size = intermediate_size // 2
                    intermediate_size_padded = utils.round_up_to_divisor(intermediate_size, self.tp_degree)
                    maybe_pad = MaybePadder(intermediate_size_padded)
                self.mlp_out_weight = maybe_pad(self.mlp_out_weight, dim=self.mlp_out_sharding)

        if utils.amp_is_u8(self.amp):
            self.attn_q_weight, self.attn_q_min, self.attn_q_max = utils.u8_encode(self.attn_q_weight)
            self.attn_k_weight, self.attn_k_min, self.attn_k_max = utils.u8_encode(self.attn_k_weight)
            self.attn_v_weight, self.attn_v_min, self.attn_v_max = utils.u8_encode(self.attn_v_weight)
            self.attn_out_weight, self.attn_out_min, self.attn_out_max = utils.u8_encode(self.attn_out_weight)
            self.mlp_in_weight, self.mlp_in_min, self.mlp_in_max = utils.u8_encode(self.mlp_in_weight)
            self.mlp_out_weight, self.mlp_out_min, self.mlp_out_max = utils.u8_encode(self.mlp_out_weight)
        if self.neuron_config and self.neuron_config.quant:
            if self.mlp_in_weight is not None:
                self.mlp_in_weight, self.mlp_in_scales = \
                    quantize.maybe_quantize_weights(self.mlp_in_weight, self.neuron_config.quant,
                                                    is_unit_scale=self.is_unit_scale)
                self.mlp_out_weight, self.mlp_out_scales = \
                    quantize.maybe_quantize_weights(self.mlp_out_weight, self.neuron_config.quant,
                                                    out_feature_dim = 1 if self.mlp_out_transposed else 0,
                                                    is_unit_scale=self.is_unit_scale)

            if self.neuron_config.quant.quantize_attn:
                assert "self_attn" not in self.neuron_config.quant.no_quantize_list, "self attn quantization not " \
                                                                                     "allowed for this model"
                if self.neuron_config.fuse_qkv:
                    fused_qkv_weight, fused_qkv_scales = \
                        quantize.maybe_quantize_weights(fused_qkv_weight, self.neuron_config.quant)
                else:
                    self.attn_q_weight, self.attn_q_scales = \
                        quantize.maybe_quantize_weights(self.attn_q_weight, self.neuron_config.quant)
                    self.attn_k_weight, self.attn_k_scales = \
                        quantize.maybe_quantize_weights(self.attn_k_weight, self.neuron_config.quant)
                    self.attn_v_weight, self.attn_v_scales = \
                        quantize.maybe_quantize_weights(self.attn_v_weight, self.neuron_config.quant)

                if self.attn_out_feature_dim is not None:
                    out_feature_dim = self.attn_out_feature_dim
                else:
                    out_feature_dim = 1 if self.attn_out_transposed else 0
                self.attn_out_weight, self.attn_out_scales = quantize.maybe_quantize_weights(
                    tensor=self.attn_out_weight,
                    quantize_config=self.neuron_config.quant,
                    out_feature_dim=out_feature_dim,
                    contract_dims=self.attn_out_contract_dims,
                )

        if self.neuron_config and self.neuron_config.fused_rmsnorm_qkv:
            self.fused_pre_attn_ln_qkv_weight = (fused_qkv_weight.T * self.pre_attn_ln_weight.to(dtype=fused_qkv_weight.dtype)).T
        
        maybe_manipulator = MaybeParallelTensorManipulator(self.tp_degree, on_cpu=self._cpu_compile, rank_id=self.neuron_config.rank_id, local_tp_degree=self.neuron_config.get_local_tp(self.tp_degree))
        maybe_duplicate = maybe_manipulator.duplicate
        maybe_shard_along = maybe_manipulator.shard_along
        maybe_primary_only = maybe_manipulator.primary_only
        maybe_shard_along_and_transform = maybe_manipulator.shard_along_and_transform
        self.pre_attn_ln_weight = maybe_duplicate(self.pre_attn_ln_weight)
        self.pre_attn_ln_bias = maybe_duplicate(self.pre_attn_ln_bias)
        qkv_tiling = self.neuron_config.qkv_tiling
        if qkv_tiling:
            qkv_weight_sharder = maybe_shard_along_and_transform
        else:
            qkv_weight_sharder = maybe_shard_along
        if self.neuron_config and self.neuron_config.fuse_qkv:
            self.attn_q_weight = qkv_weight_sharder(fused_qkv_weight, dim=1, weight_tiling=qkv_tiling)
            self.fused_pre_attn_ln_qkv_weight = maybe_shard_along(self.fused_pre_attn_ln_qkv_weight, dim=1) # do not tile weights here
            self.attn_q_bias = maybe_shard_along(fused_qkv_bias, dim=0)
            self.attn_q_scales = maybe_shard_along(fused_qkv_scales, dim=0)
        else:
            self.attn_q_weight = qkv_weight_sharder(self.attn_q_weight, dim=1, weight_tiling=qkv_tiling)
            self.attn_q_bias = maybe_shard_along(self.attn_q_bias, dim=0)
            self.attn_q_scales = maybe_shard_along(self.attn_q_scales, dim=0)
        self.attn_k_weight = qkv_weight_sharder(self.attn_k_weight, dim=1, weight_tiling=qkv_tiling)
        self.attn_k_scales = maybe_shard_along(self.attn_k_scales, dim=0)
        self.attn_k_bias = maybe_shard_along(self.attn_k_bias, dim=0)
        self.attn_v_weight = qkv_weight_sharder(self.attn_v_weight, dim=1, weight_tiling=qkv_tiling)
        self.attn_v_scales = maybe_shard_along(self.attn_v_scales, dim=0)
        self.attn_v_bias = maybe_shard_along(self.attn_v_bias, dim=0)
        self.attn_out_weight = maybe_shard_along(self.attn_out_weight, dim=self.attn_out_sharding)
        self.attn_out_scales = maybe_duplicate(self.attn_out_scales)
        self.attn_out_bias = maybe_primary_only(self.attn_out_bias)
        self.post_attn_ln_weight = maybe_duplicate(self.post_attn_ln_weight)
        self.post_attn_ln_bias = maybe_duplicate(self.post_attn_ln_bias)
        self.pre_mlp_ln_weight = maybe_duplicate(self.pre_mlp_ln_weight)
        self.pre_mlp_ln_bias = maybe_duplicate(self.pre_mlp_ln_bias)
        if self.mlp_in_weight is not None:
            self.mlp_in_weight = maybe_shard_along_and_transform(self.mlp_in_weight, 1, weight_tiling=self.neuron_config.weight_tiling)
            self.mlp_in_scales = maybe_shard_along(self.mlp_in_scales, dim=0)
            self.mlp_in_bias = maybe_shard_along(self.mlp_in_bias, dim=0)
            self.mlp_out_weight = maybe_shard_along_and_transform(self.mlp_out_weight, dim=self.mlp_out_sharding, weight_tiling=self.neuron_config.weight_tiling)
            self.mlp_out_scales = maybe_duplicate(self.mlp_out_scales)
            self.mlp_out_bias = maybe_primary_only(self.mlp_out_bias)
        self.post_mlp_ln_weight = maybe_duplicate(self.post_mlp_ln_weight)
        self.post_mlp_ln_bias = maybe_duplicate(self.post_mlp_ln_bias)

        extras = []
        for param, dim, allow_pad, allow_quantize, out_feature_dim, allow_transform in self.extra_parameters:
            weight_tiling = self.neuron_config.weight_tiling
            if allow_pad:
                size = utils.round_up_to_divisor(param.shape[dim], self.tp_degree)
                if weight_tiling and allow_transform:
                    size = utils.round_up_to_divisor(size // self.tp_degree,
                                                     constants.TILE_SIZE) * self.tp_degree
                param = utils.pad(param, dim, size)

            if allow_quantize:
                # If the parameter is quantizable and the quantization is enabled, we calculate the
                # scaling factors here, otherwise we still need to add a scale placeholder to match
                # the layer arguments
                if self.neuron_config and self.neuron_config.quant:
                    param, scales = quantize.maybe_quantize_weights(param, self.neuron_config.quant,
                                                                    out_feature_dim=out_feature_dim,
                                                                    is_unit_scale=self.is_unit_scale)
                    scales_dim = 0 if dim == out_feature_dim else None
                    scales = maybe_manipulator.duplicate_or_shard_along(scales, scales_dim)
                else:
                    scales = None

            if allow_transform:
                param = maybe_shard_along_and_transform(param, dim, weight_tiling=weight_tiling)
            else:
                param = maybe_manipulator.duplicate_or_shard_along(param, dim)

            extras.append(param)
            if allow_quantize:
                extras.append(scales)

        self.extra_parameters = extras
        self.init_caches()

    def save_presharded_weights(self, directory):
        save_dict = {}
        for key, value in self.__dict__.items():
            if isinstance(value, torch.Tensor) and value.device.type == 'xla':
                cpu_tensor = ops.parallel_cpu(value)
                for i in range(len(cpu_tensor)):
                    save_dict[f'{key}*{i}'] = cpu_tensor[i]

        for i, param in enumerate(self.extra_parameters):
            if param is None:
                # save an empty tensor
                cpu_tensor = [torch.tensor([])]
            else:
                cpu_tensor = ops.parallel_cpu(param)
            for j in range(len(cpu_tensor)):
                save_dict[f'extra_parameter{i}*{j}'] = cpu_tensor[j]

        save_file(save_dict, os.path.join(directory, f"decoder_layer_{self.layer_num}.safetensors"))

    def load_presharded_weights(self, ps_dir):
        # only need to get names for first layer
        with safe_open(os.path.join(ps_dir, f"decoder_layer_{self.layer_num}.safetensors"), framework='pt') as f:
            layer_attr_names = get_attribute_names(f)
            presharded_weights_to_neuron(f, self, layer_attr_names)
            self.format_extra_parameters()
            for batch_size in self.batch_sizes:
                self.init_caches()

    def format_extra_parameters(self):
        # deal with the extra_parameters (decoder_layer only)
        i = 0
        self.extra_parameters = []
        while True:
            extra_param = getattr(self, f"extra_parameter{i}", None)
            if extra_param is not None:
                if extra_param == 'None':
                    self.extra_parameters.append(None)
                else:
                    self.extra_parameters.append(extra_param)
            else: # Previous parameter was the last one
                break
            i+=1

    @property
    def shard_over_batch(self):
        return self.neuron_config.group_query_attention == constants.GQA.SHARD_OVER_BATCH

    @property
    def bsh_cache_layout(self):
        return self.neuron_config.cache_layout == constants.LAYOUT_BSH

    def init_caches(self):
        n_heads_kv_cache = self.n_kv_head

        # When padding, compute the hidden size based on the padding. We must
        # allow the KV cache to be padded so it can be evenly divisible across
        # NeuronCores.
        if self.allow_pad and not self.shard_over_batch:
            n_heads_kv_cache = utils.round_up_to_divisor(self.n_kv_head, self.tp_degree)
        block_size = self.neuron_config.continuous_batching.block_size if self.neuron_config.paged_attention else self.n_positions
        # Select manipulator based on device
        if self._cpu_compile:
            manipulator = parallel.CPUTensorManipulator(self.tp_degree, rank_id=self.neuron_config.rank_id, local_tp_degree=self.neuron_config.get_local_tp(self.tp_degree))
        else:
            manipulator = parallel.ParallelTensorManipulator(self.tp_degree, rank_id=self.neuron_config.rank_id, local_tp_degree=self.neuron_config.get_local_tp(self.tp_degree))
        # Separate KV cache for each batch size
        for batch_size in self.batch_sizes:
            num_blocks = self.neuron_config.continuous_batching.num_blocks if self.neuron_config.paged_attention else batch_size
            if self.bsh_cache_layout:
                cache_shape = [num_blocks, block_size, n_heads_kv_cache, self.attention_head_size]
                self.cache_shape[batch_size] = [num_blocks, block_size, n_heads_kv_cache // self.tp_degree, self.attention_head_size]
            else:
                cache_shape = [block_size, num_blocks, n_heads_kv_cache, self.attention_head_size]
                self.cache_shape[batch_size] = [block_size, num_blocks, n_heads_kv_cache // self.tp_degree, self.attention_head_size]
            if hasattr(torch, 'float8_e4m3fn') and self.cache_dtype==torch.float8_e4m3fn:
                int8_cpu_cache = torch.zeros(cache_shape, dtype=torch.uint8) #Cannot directly use fp8: *** RuntimeError: "fill_cpu" not implemented for 'Float8_e4m3fn'
                cpu_cache= int8_cpu_cache.to(torch.float8_e4m3fn)
            else:
                cpu_cache = torch.zeros(cache_shape, dtype=self.cache_dtype)
            if self.shard_over_batch:
                assert not self.bsh_cache_layout, "shard-over-batch for GQA with BSH cache layout is not supported."
                self.cache_shape[batch_size] = [block_size, num_blocks // self.tp_degree, n_heads_kv_cache, self.attention_head_size]
                self.attn_k_cache[batch_size] = (manipulator.shard_along(cpu_cache, dim=1))
                self.attn_v_cache[batch_size] = (manipulator.shard_along(cpu_cache, dim=1))
            elif self.neuron_config.shard_over_sequence:
                # note here we use kv_replication since self.n_kv_head is replicated
                # for SOS we need original n_kv_head before replication
                kv_replication = self.kv_replication
                if self.neuron_config.paged_attention:
                    cache_shape = [num_blocks, block_size*self.tp_degree//kv_replication, n_heads_kv_cache//self.tp_degree, self.attention_head_size]
                    cpu_cache = torch.zeros(cache_shape, dtype=self.cache_dtype)
                    self.cache_shape[batch_size] = [num_blocks, block_size//kv_replication, n_heads_kv_cache//self.tp_degree, self.attention_head_size]
                    self.attn_k_cache[batch_size] = (manipulator.shard_along(cpu_cache, dim=1))
                    self.attn_v_cache[batch_size] = (manipulator.shard_along(cpu_cache, dim=1))
                else:
                    cache_shape = [self.n_positions*self.tp_degree//kv_replication, batch_size, n_heads_kv_cache//self.tp_degree, self.attention_head_size]
                    cpu_cache = torch.zeros(cache_shape, dtype=torch.uint8).to(torch.float8_e4m3fn) \
                        if hasattr(torch, 'float8_e4m3fn') and self.cache_dtype==torch.float8_e4m3fn \
                        else torch.zeros(cache_shape, dtype=self.cache_dtype)
                    self.cache_shape[batch_size] = [self.n_positions//kv_replication, batch_size, n_heads_kv_cache//self.tp_degree, self.attention_head_size]
                    self.attn_k_cache[batch_size] = (manipulator.shard_along(cpu_cache, dim=0))
                    self.attn_v_cache[batch_size] = (manipulator.shard_along(cpu_cache, dim=0))
            else:
                assert (n_heads_kv_cache >= self.tp_degree) and (n_heads_kv_cache % self.tp_degree == 0), \
                    f"cannot shard along kv_heads dimension: n_kv_head={n_heads_kv_cache}, tp_degree={self.tp_degree}"
                self.attn_k_cache[batch_size] = (manipulator.shard_along(cpu_cache, dim=2))
                self.attn_v_cache[batch_size] = (manipulator.shard_along(cpu_cache, dim=2))

    def assign_caches(self, layer, buckets_from_src=False):
        batch_sizes = self.batch_sizes
        if buckets_from_src:
            # In continuous batching, we exclusively use batch_size=1 for parallel context encoding.
            # But still use all batch_sizes for decoding.
            batch_sizes = layer.batch_sizes
        for batch_size in batch_sizes:
            self.attn_k_cache[batch_size] = layer.attn_k_cache[batch_size]
            self.attn_v_cache[batch_size] = layer.attn_v_cache[batch_size]
            self.cache_shape[batch_size] = layer.cache_shape[batch_size]

    def all_parameters(self):
        return [
            self.pre_attn_ln_weight,
            self.pre_attn_ln_bias,
            self.fused_pre_attn_ln_qkv_weight,
            self.attn_q_weight,
            self.attn_q_scales,
            self.attn_q_bias,
            self.attn_k_weight,
            self.attn_k_scales,
            self.attn_k_bias,
            self.attn_v_weight,
            self.attn_v_scales,
            self.attn_v_bias,
            self.attn_out_weight,
            self.attn_out_scales,
            self.attn_out_bias,
            self.post_attn_ln_weight,
            self.post_attn_ln_bias,
            self.pre_mlp_ln_weight,
            self.pre_mlp_ln_bias,
            self.mlp_in_weight,
            self.mlp_in_scales,
            self.mlp_in_bias,
            self.mlp_out_weight,
            self.mlp_out_scales,
            self.mlp_out_bias,
            self.post_mlp_ln_weight,
            self.post_mlp_ln_bias,
            *self.extra_parameters,
        ]

    def valid_parameters(self):
        return [par for par in self.all_parameters() if par is not None]

    def u8_bounds(self):
        bounds = (
            self.attn_q_min, self.attn_q_max, self.attn_k_min, self.attn_k_max,
            self.attn_v_min, self.attn_v_max, self.attn_out_min, self.attn_out_max,
            self.mlp_in_min, self.mlp_in_max, self.mlp_out_min, self.mlp_out_max,
        )
        if any(bd is None for bd in bounds):
            return None
        return bounds

    def hlo_maybe_dequantize_weights(self, hlo_weights):
        u8_bounds = self.u8_bounds()
        if u8_bounds is None:
            return hlo_weights
        first_valid_weight, *_ = [weight for weight in hlo_weights if weight is not None]
        scribe = first_valid_weight.scribe
        amp, quantized, dequantized = utils.parse_amp(self.amp)
        dtype = getattr(scribe, amp)
        dequant_dtype = None if dequantized is None else getattr(scribe, dequantized)

        def attn_u8_decode(q_weight, k_weight, v_weight, out_weight, u8_bounds):
            q_min, q_max, k_min, k_max, v_min, v_max, out_min, out_max, *_ = u8_bounds
            q_weight = hlo.u8_decode(dtype, dequant_dtype, q_weight, q_min, q_max)
            k_weight = hlo.u8_decode(dtype, dequant_dtype, k_weight, k_min, k_max)
            v_weight = hlo.u8_decode(dtype, dequant_dtype, v_weight, v_min, v_max)
            out_weight = hlo.u8_decode(dtype, dequant_dtype, out_weight, out_min, out_max)
            return q_weight, k_weight, v_weight, out_weight

        def mlp_u8_decode(in_weight, out_weight, u8_bounds):
            *_, in_min, in_max, out_min, out_max = u8_bounds
            in_weight = hlo.u8_decode(dtype, dequant_dtype, in_weight, in_min, in_max)
            out_weight = hlo.u8_decode(dtype, dequant_dtype, out_weight, out_min, out_max)
            return in_weight, out_weight

        (
            pre_attn_ln_weight,
            pre_attn_ln_bias,
            fused_pre_attn_ln_qkv_weight,
            attn_q_weight,
            attn_q_scales,
            attn_q_bias,
            attn_k_weight,
            attn_k_scales,
            attn_k_bias,
            attn_v_weight,
            attn_v_scales,
            attn_v_bias,
            attn_out_weight,
            attn_out_scales,
            attn_out_bias,
            post_attn_ln_weight,
            post_attn_ln_bias,
            pre_mlp_ln_weight,
            pre_mlp_ln_bias,
            mlp_in_weight,
            mlp_in_scales,
            mlp_in_bias,
            mlp_out_weight,
            mlp_out_scales,
            mlp_out_bias,
            post_mlp_ln_weight,
            post_mlp_ln_bias,
        ) = hlo_weights
        attn_q_weight, attn_k_weight, attn_v_weight, attn_out_weight = attn_u8_decode(
            attn_q_weight, attn_k_weight, attn_v_weight, attn_out_weight, u8_bounds)
        mlp_in_weight, mlp_out_weight = mlp_u8_decode(mlp_in_weight, mlp_out_weight, u8_bounds)
        return [
            pre_attn_ln_weight,
            pre_attn_ln_bias,
            fused_pre_attn_ln_qkv_weight,
            attn_q_weight,
            attn_q_scales,
            attn_q_bias,
            attn_k_weight,
            attn_k_scales,
            attn_k_bias,
            attn_v_weight,
            attn_v_scales,
            attn_v_bias,
            attn_out_weight,
            attn_out_scales,
            attn_out_bias,
            post_attn_ln_weight,
            post_attn_ln_bias,
            pre_mlp_ln_weight,
            pre_mlp_ln_bias,
            mlp_in_weight,
            mlp_in_scales,
            mlp_in_bias,
            mlp_out_weight,
            mlp_out_scales,
            mlp_out_bias,
            post_mlp_ln_weight,
            post_mlp_ln_bias,
        ]

    def reset(self):
        for batch_size in self.batch_sizes:
            if isinstance(self.attn_k_cache[batch_size], list):
                self.attn_k_cache[batch_size] = self.attn_k_cache[batch_size][0]
            zero_cache = torch.zeros(self.attn_k_cache[batch_size].shape, dtype=self.attn_k_cache[batch_size].dtype)
            zero_cache = [zero_cache for _ in range(self.neuron_config.get_local_tp(self.tp_degree))]
            if not self._cpu_compile:
                ops.parallel_write(self.attn_k_cache[batch_size], zero_cache)
                ops.parallel_write(self.attn_v_cache[batch_size], zero_cache)
            else:
                self.attn_k_cache[batch_size] = zero_cache
                self.attn_v_cache[batch_size] = zero_cache

    def assign_parameters(self, layer):
        self.pre_attn_ln_weight = layer.pre_attn_ln_weight
        self.pre_attn_ln_bias = layer.pre_attn_ln_bias
        self.fused_pre_attn_ln_qkv_weight = layer.fused_pre_attn_ln_qkv_weight
        self.attn_q_weight = layer.attn_q_weight
        self.attn_q_scales = layer.attn_q_scales
        self.attn_q_bias = layer.attn_q_bias
        self.attn_k_weight = layer.attn_k_weight
        self.attn_k_scales = layer.attn_k_scales
        self.attn_k_bias = layer.attn_k_bias
        self.attn_v_weight = layer.attn_v_weight
        self.attn_v_scales = layer.attn_v_scales
        self.attn_v_bias = layer.attn_v_bias
        self.attn_out_weight = layer.attn_out_weight
        self.attn_out_scales = layer.attn_out_scales
        self.attn_out_bias = layer.attn_out_bias
        self.post_attn_ln_weight = layer.post_attn_ln_weight
        self.post_attn_ln_bias = layer.post_attn_ln_bias
        self.pre_mlp_ln_weight = layer.pre_mlp_ln_weight
        self.pre_mlp_ln_bias = layer.pre_mlp_ln_bias
        self.mlp_in_weight = layer.mlp_in_weight
        self.mlp_in_scales = layer.mlp_in_scales
        self.mlp_in_bias = layer.mlp_in_bias
        self.mlp_out_weight = layer.mlp_out_weight
        self.mlp_out_scales = layer.mlp_out_scales
        self.mlp_out_bias = layer.mlp_out_bias
        self.post_mlp_ln_weight = layer.post_mlp_ln_weight
        self.post_mlp_ln_bias = layer.post_mlp_ln_bias
        self.attn_q_min = layer.attn_q_min
        self.attn_q_max = layer.attn_q_max
        self.attn_k_min = layer.attn_k_min
        self.attn_k_max = layer.attn_k_max
        self.attn_v_min = layer.attn_v_min
        self.attn_v_max = layer.attn_v_max
        self.attn_out_min = layer.attn_out_min
        self.attn_out_max = layer.attn_out_max
        self.mlp_in_min = layer.mlp_in_min
        self.mlp_in_max = layer.mlp_in_max
        self.mlp_out_min = layer.mlp_out_min
        self.mlp_out_max = layer.mlp_out_max
        self.extra_parameters = layer.extra_parameters

class MaybeParallelTensorManipulator:

    def __init__(self, tp_degree, on_cpu=False, rank_id=0, local_tp_degree=None):
        self.use_cpu = on_cpu
        if on_cpu:
            self.manipulator = parallel.CPUTensorManipulator(tp_degree, rank_id=rank_id, local_tp_degree=local_tp_degree)
        else:
            self.manipulator = parallel.ParallelTensorManipulator(tp_degree, rank_id=rank_id, local_tp_degree=local_tp_degree)

    def duplicate(self, tensor):
        if tensor is None:
            return None
        return self.manipulator.duplicate(tensor)

    def shard_along(self, tensor, dim, weight_tiling=None):
        # weight_tiling is not used
        if tensor is None:
            return None
        return self.manipulator.shard_along(tensor, dim)

    def primary_only(self, tensor):
        if tensor is None:
            return None
        return self.manipulator.primary_only(tensor)

    def duplicate_or_shard_along(self, tensor, dim):
        if dim is None:
            return self.duplicate(tensor)
        return self.shard_along(tensor, dim)

    def transform_and_tile_weight_layout(self, tensors, weight_tiling=False, permute_order=[1, 2, 0, 3]):
        if tensors is None:
            return None

        if weight_tiling:
            new_tensors = []
            for tensor in tensors:
                K, N = tensor.shape
                assert K % constants.TILE_SIZE == 0 and N % constants.TILE_SIZE == 0, (f"Weight dimensions must be "
                                                                                       f"divisible by {constants.TILE_SIZE} "
                                                                                       f"but received weight with shape={K, N}."
                )
                reshape_sizes = [K // constants.TILE_SIZE,
                                 constants.TILE_SIZE,
                                 N // constants.TILE_SIZE,
                                 constants.TILE_SIZE]
                tensor = tensor.reshape(reshape_sizes) \
                               .permute(permute_order)
                tensor = tensor.contiguous()
                new_tensors.append(tensor)
            return new_tensors

        return tensors

    def shard_along_and_transform(self, tensor, dim, weight_tiling=False):
        if tensor is None:
            return None
        tensors = self.manipulator.shard_along_on_cpu(tensor, dim)
        tensors = self.transform_and_tile_weight_layout(tensors, weight_tiling)
        if not self.use_cpu:
            tensor = ops.parallel_to_nc(tensors)
        return tensor


class DecoderParameterBuilder:

    def __init__(self, scribe, parameter_number):
        self.scribe = scribe
        self.parameter_number = parameter_number
        self.dtype_converter = compiler.DataTypeConverter()

    def from_tensor(self, tensor, dim_size=None):
        if tensor is None:
            return None
        # Tensor may be a list of tensors (e.g. [tensor(...)]) during CPU compilation flow
        if isinstance(tensor, list):
            tensor = tensor[0]
        name = self.dtype_converter.torch2name(tensor.dtype)
        dtype = getattr(self.scribe, name)
        sizes = list(tensor.shape)
        if dim_size is not None:
            for dim, size in dim_size.items():
                sizes[dim] = size
        param = dtype[sizes].Parameter(parameter_number=self.parameter_number)
        self.parameter_number += 1
        return param

class DecoderProgram:

    def __init__(self, neuron_config, layers, hlo_modules : dict, debug_tensors: dict, num_inputs, tp_degree, n_positions_list, batch_sizes, prefixed_length=0, batch_size_for_shared_caches=False, tag=None, num_exec_repetition=1, on_cpu=False):
        # Each hlo module corresponds to one npos and one batch_size
        # hlo_modules is a 2D map (i,j) i is npos , j is batch_size
        self.neuron_config = neuron_config
        self.layers = layers
        self.batch_sizes = batch_sizes
        self.batch_size_for_shared_caches = batch_size_for_shared_caches
        self.n_positions_list = n_positions_list
        self.prefixed_length = prefixed_length
        first_hlo = hlo_modules[self.n_positions_list[0], self.batch_sizes[0]]
        hlos_for_input = list()
        hlos_for_input = [hlo_modules[self.n_positions_list[0],batch_size] for batch_size in self.batch_sizes]
        self.input_buffers = list()
        self.input_buffers = [[compiler.gen_zero_input(hlo,idx) for idx in range(num_inputs)] for hlo in hlos_for_input]
        self.kernels = dict()
        for npos, batch_size in itertools.product(self.n_positions_list, self.batch_sizes):
            kernel_tag = f"seqlen{npos}-batch{batch_size}"
            if tag is not None:
                if self.neuron_config.optimized_paged_attention and tag == 'token':
                    kernel_tag = f"{tag}-seqlen{npos}-block{batch_size}"
                else:
                    kernel_tag = f"{tag}-seqlen{npos}-batch{batch_size}"
                if self.neuron_config.enable_chunked_prefill:
                    kernel_tag = f"{tag}-chunked-prefill-chunksize{npos}-block{batch_size}"
            self.kernels[npos,batch_size] = compiler.ParallelKernel(hlo_modules[npos, batch_size], self.neuron_config.get_local_tp(tp_degree), self.neuron_config.get_g_start_device_id(tp_degree), self.neuron_config.get_g_device_count(tp_degree), tag=kernel_tag, num_exec_repetition=num_exec_repetition)
        self.debug_tensors = debug_tensors
        self.debug_output_buffers = dict()
        for npos, batch_size in itertools.product(self.n_positions_list, self.batch_sizes):
            debug_output_names = {tensor.metadata['output_index']: name for name, tensor in self.debug_tensors[npos,batch_size].items()}
            program_shape = hlo_modules[npos, batch_size].host_program_shape
            outputs = [program_shape.result] if len(program_shape.result.tuple_shapes) == 0 else program_shape.result.tuple_shapes
            debug_tensor_names_and_shapes = [(debug_output_names[idx], shape) for idx, shape in enumerate(outputs) if idx in debug_output_names]
            self.debug_output_buffers[npos,batch_size] = {name: compiler.gen_zero_output_from_shape_proto(shape) for name, shape in debug_tensor_names_and_shapes}
        # self.n_positions_list = [read_n_position(hm, num_inputs) for hm in hlo_modules]
        self.n_active_tokens = read_n_active_tokens(first_hlo)
        self.tp_degree = tp_degree
        self.need_reorder_cache = False
        self.tag = tag
        self._cpu_compile = on_cpu
        # Select manipulator based on device
        if self._cpu_compile:
            self.manipulator = parallel.CPUTensorManipulator(tp_degree, rank_id=self.neuron_config.rank_id, local_tp_degree=self.neuron_config.get_local_tp(tp_degree))
        else:
            self.manipulator = parallel.ParallelTensorManipulator(tp_degree, rank_id=self.neuron_config.rank_id, local_tp_degree=self.neuron_config.get_local_tp(tp_degree))

    def setup(self, layers, pre_layer_params, ln_lm_head_params, io_ring_cache_size=1):
        self.input_buffers = [[self.manipulator.duplicate(buf) for buf in input_buffers_for_batch_size] for input_buffers_for_batch_size in self.input_buffers]
        if self.logits_buffer:
            if self.neuron_config.log_softmax_scores or self.neuron_config.is_eagle_target:
                self.logits_buffer = [[self.manipulator.duplicate(buf) for buf in logits_buffer_batch_size] for logits_buffer_batch_size in self.logits_buffer]
            else:
                self.logits_buffer = [self.manipulator.duplicate(buf) for buf in self.logits_buffer]
        self.setup_debug_tensors()
        for kernel in self.kernels.values():
            kernel.load(io_ring_cache_size)

    def setup_debug_tensors(self):
        filled_debug_buffers = {}
        for (npos, bs), bufs in self.debug_output_buffers.items():
            filled_debug_buffers[npos,bs] = {name: self.manipulator.duplicate(buf) for name, buf in bufs.items()}
        self.debug_output_buffers = filled_debug_buffers

    def setup_reorder_cache(self, also_compile_now=True):
        self.need_reorder_cache = True
        self.reorder_cache_hlo_kernels = [self._create_reoder_cache_kernel(batch_size) for batch_size in self.batch_sizes]
        if also_compile_now:
            self.setup_reorder_cache_kernels()

    def find_bucket_id(self, length):
        return next(idx for idx, npos in enumerate(self.n_positions_list) if npos >= length+1)

    def find_block_bucket_size(self, context_length, block_size):
        n_active_blocks = ((context_length+block_size-1) // block_size).sum().item()
        active_block = bucket.find(self.batch_sizes, n_active_blocks)
        return active_block

    def inputs_host_to_device(self, input_tensors, batch_size):

        def process_input_tensors(tensor, idx):
            # process input_ids
            if idx == 0 and self.neuron_config.sequence_parallel_norm and not self.neuron_config.on_device_embedding:
                n_active_tokens = tensor.shape[1]
                if n_active_tokens > self.neuron_config.sequence_parallel_norm_threshold:
                    return self.manipulator.shard_along_on_cpu(tensor, 1)
            # handle the rest of the inputs
            return self.manipulator.duplicate_on_cpu(tensor)

        # This means there is a separate neff for embedding so the inputs will be the input_ids
        if self.neuron_config.on_device_embedding and isinstance(self, DecoderProgramMultiLayer):
            input_buffers = self.input_ids_buffer[self.batch_sizes.index(batch_size)]
        else:
            input_buffers = self.input_buffers[self.batch_sizes.index(batch_size)]
        # TODO: Check how to handle this corner condition.
        if len(input_tensors) == 5 and len(input_buffers) == 6:
            input_buffers.pop(3)
        for idx, (buf, tensor) in enumerate(zip(input_buffers, input_tensors)):
            tensor = tensor.to(buf.dtype)
            tensor = process_input_tensors(tensor, idx)
            assert buf.shape == tensor[0].shape, f"Copying tensor from host to device: buffer ({buf.shape}) and tensor ({tensor[0].shape}) have different shapes!"
            ops.parallel_write(buf, tensor)

    def run(self, bucket_id):
        raise NotImplementedError(DecoderProgram)

    def maybe_logits_device_to_host(self, batch_size, return_ranks):
        idx = self.batch_sizes.index(batch_size)
        if self.logits_buffer:
            if self.tp_degree == self.neuron_config.get_local_tp(self.tp_degree):

                if self.neuron_config.log_softmax_scores or self.neuron_config.is_eagle_target:
                    return [self.manipulator.unshard_along(val, dim=0) for val in self.logits_buffer[idx]]
                else:
                    logits = self.manipulator.unshard_along(self.logits_buffer[idx], dim=0)
                    if return_ranks > 0:
                        rank_size = logits.shape[0] // self.tp_degree
                        logits = logits[:rank_size * return_ranks]
                    return logits

            else:
                return ops.parallel_cpu(self.logits_buffer[idx])[0]
        else:
            return None

    def debug_tensors_to_host(self, bucket_id, batch_size):
        npos = self.n_positions_list[bucket_id]
        if global_debugger.debug_tensors is None:
            return
        for debug_tensor_name, debug_buffer in self.debug_output_buffers[npos,batch_size].items():
            unshard_dim = self.debug_tensors[npos,batch_size][debug_tensor_name].unshard_dim
            if unshard_dim is None:
                tensor = ops.parallel_cpu(debug_buffer)[0]
            else:
                tensor = self.manipulator.unshard_along(debug_buffer, dim=unshard_dim)
            global_debugger.debug_tensors[debug_tensor_name] = tensor

    def _fill_io_tensors(self, input_tensors, output_tensors, layers, npos, batch_size):
        end = npos
        if self.prefixed_length > 0:
            end = npos + self.prefixed_length
        if self.batch_size_for_shared_caches:
            batch_size = self.batch_size_for_shared_caches
        for layer in layers:
            for cache in layer.attn_k_cache[batch_size], layer.attn_v_cache[batch_size]:
                if self.neuron_config.paged_attention:
                    # don't slice because we pass full KV cache for each bucket
                    cache_slice = cache
                else:
                    cache_slice = self.manipulator.slice_on_nc(cache, 0, start=0, end=end, step=1)
                input_tensors.append(cache_slice)
                output_tensors.append(cache_slice)
        for layer in layers:
            input_tensors.extend(layer.valid_parameters())

    def _fill_debug_tensors(self, output_tensors, npos, batch_size):
        output_tensors.extend([None] * len(self.debug_output_buffers[npos,batch_size]))
        for debug_tensor_name, buf in self.debug_output_buffers[npos,batch_size].items():
            output_tensors[self.debug_tensors[npos,batch_size][debug_tensor_name].metadata['output_index']] = buf

    def _create_reoder_cache_kernel(self, batch_size):
        # assume each layer have same size of cache
        def _reorder_cache(scribe):
            reorder_ids = scribe.s64[batch_size].Parameter(parameter_number=0)
            caches = []
            param_builder = DecoderParameterBuilder(scribe, 1)
            for layer in self.layers:
                for cache in layer.attn_k_cache[batch_size], layer.attn_v_cache[batch_size]:
                    cache = param_builder.from_tensor(cache)
                    caches.append(cache)
            outputs = []
            # TODO: concat -> reorder -> indexing?
            # cache of shape [self.n_positions, self.batch_size, n_heads_kv_cache//self.tp_degree, self.attention_head_size]
            # we want to reorder on batch dimension
            for cache in caches:
                new_cache = hlo.index_select(cache, 1, reorder_ids)
                outputs.append(new_cache)
            root_shapes = [tensor.dtype[tensor.sizes] for tensor in outputs]
            return scribe.tuple(*root_shapes).Tuple(*outputs)

        return compiler.HLOKernel(_reorder_cache, self.tp_degree)

    def setup_reorder_cache_kernels(self):
        for bs_idx, batch_size in  enumerate(self.batch_sizes):
            reorder_cache_hlo_kernel = self.reorder_cache_hlo_kernels[bs_idx]
            self._setup_reorder_cache_kernel(reorder_cache_hlo_kernel, batch_size)

    def _setup_reorder_cache_kernel(self, reorder_cache_hlo_kernel, batch_size):
        reorder_cache_hlo_kernel.build()
        reorder_cache_hlo_kernel.load()
        # setup memory buffer
        reorder_ids = torch.zeros(self.layers[0].cache_shape[batch_size], dtype=torch.int64)
        self.reorder_ids_buffers = reorder_cache_hlo_kernel.manipulator.duplicate(reorder_ids)
        input_tensors = [self.reorder_ids_buffers]
        output_tensors = []
        for layer in self.layers:
            for cache in layer.attn_k_cache[batch_size], layer.attn_v_cache[batch_size]:
                input_tensors.append(cache)
                output_tensors.append(cache) # aliasing
        reorder_cache_hlo_kernel.setup(input_tensors, output_tensors)

    def reorder_cache_by_batch_size(self, reorder_ids, batch_size):
        assert self.need_reorder_cache, "DecoderProgram is not built with reorder_cache"
        reorder_ids_tensor = torch.tensor(reorder_ids, dtype=torch.int64)
        # TODO: if reorder_ids == range(batch_size), don't do anything
        idx = self.batch_sizes.index(batch_size)
        reorder_ids_tensors_cpu = self.reorder_cache_hlo_kernels[idx].manipulator.duplicate_on_cpu(reorder_ids_tensor)
        ops.parallel_write(self.reorder_ids_buffers, reorder_ids_tensors_cpu)
        self.reorder_cache_hlo_kernels[idx].run()

    def reorder_cache(self, reorder_ids):
        assert self.need_reorder_cache, "DecoderProgram is not built with reorder_cache"
        reorder_ids_tensor = torch.tensor(reorder_ids, dtype=torch.int64)
        # TODO: if reorder_ids == range(batch_size), don't do anything
        for bs_idx, batch_size in enumerate(self.batch_sizes):
            reorder_ids_tensors_cpu = self.reorder_cache_hlo_kernels[bs_idx].manipulator.duplicate_on_cpu(reorder_ids_tensor)
            ops.parallel_write(self.reorder_ids_buffers, reorder_ids_tensors_cpu)
            self.reorder_cache_hlo_kernels[bs_idx].run()

    def get_kernels(self):
        all_kernels = list()
        for npos, batch_size in itertools.product(self.n_positions_list, self.batch_sizes):
            all_kernels.append(self.kernels[npos,batch_size])
        return all_kernels


class DecoderProgramFullyUnrolled(DecoderProgram):

    def __init__(self, neuron_config, layers, hlo_modules, debug_tensors, num_inputs, tp_degree, n_positions_list, batch_sizes, prefixed_length=0, batch_size_for_shared_caches=None, tag=None, on_cpu=False):
        super().__init__(neuron_config, layers, hlo_modules, debug_tensors, num_inputs, tp_degree, n_positions_list, batch_sizes, prefixed_length, batch_size_for_shared_caches, tag=tag, on_cpu=on_cpu)
        hlos_for_input = list()
        hlos_for_input = [hlo_modules[self.n_positions_list[0],batch_size] for batch_size in self.batch_sizes]
        if self.neuron_config.log_softmax_scores or self.neuron_config.is_eagle_target:
            self.logits_buffer = [[compiler.gen_zero_output(hlo, 0), compiler.gen_zero_output(hlo, 1)] for hlo in hlos_for_input]
        else:
            self.logits_buffer = [compiler.gen_zero_output(hlo, 0) for hlo in hlos_for_input]
        self.memories = dict()
        self.executors = dict()


    def setup(self, layers, pre_layer_params, ln_lm_head_params):
        super().setup(layers, pre_layer_params, ln_lm_head_params)

        self.memories = dict()
        for npos,batch_size in itertools.product(self.n_positions_list, self.batch_sizes):
            self.memories[npos,batch_size] = self.kernels[npos,batch_size].build_memory()

        # Setup the memory with input and output buffers
        for bs_idx, batch_size in enumerate(self.batch_sizes):
            for npos in self.n_positions_list:
                input_tensors = [*self.input_buffers[bs_idx]]
                if self.neuron_config.log_softmax_scores or self.neuron_config.is_eagle_target:
                    output_tensors = [*self.logits_buffer[bs_idx]]
                else:
                    output_tensors = [self.logits_buffer[bs_idx]]
                self._fill_io_tensors(input_tensors, output_tensors, layers, npos, batch_size)
                self._fill_debug_tensors(output_tensors, npos, batch_size)
                input_tensors.extend(pre_layer_params)
                input_tensors.extend(ln_lm_head_params)
                self.memories[npos,batch_size].setup(input_tensors, output_tensors)

        # Warmup kernels to avoid unexpected initialization at runtime
        for kernel in self.get_kernels():
            kernel.warmup()

    def run(self, bucket_id, batch_size):
        npos = self.n_positions_list[bucket_id]
        self.kernels[npos,batch_size](self.memories[npos,batch_size])

    def enable_executor(self):
        for bs_idx, batch_size in enumerate(self.batch_sizes):
            for npos in self.n_positions_list:
                input_tensors = [*self.input_buffers[bs_idx]]
                if self.neuron_config.is_eagle_target:
                    output_tensors = [*self.logits_buffer[bs_idx]]
                else:
                    output_tensors = [self.logits_buffer[bs_idx]]
                executor = self.kernels[npos,batch_size].build_executor(self.memories[npos,batch_size], input_tensors, output_tensors)
                self.executors[npos,batch_size] = executor

    def execute(self, bucket_id, batch_size, *inputs, return_ranks=-1):
        """
        Execute a kernel with using the optimized ParallelExecutor.

        This is an alternative to the `run` method which requires that an
        executor has been constructed for each of the underlying kernels.

        Arguments:
            bucket_id: The kernel bucket to execute
            inputs: The set of CPU tensors to copy to each model
            return_ranks: The number of ranks to copy back to CPU
        """
        npos = self.n_positions_list[bucket_id]
        return self.executors[npos,batch_size](inputs, return_ranks)

    def get_kernels(self):
        all_kernels = super().get_kernels()
        # only true when reorder_cache called before to_neuron
        if self.need_reorder_cache:
            for hlo_kernel in self.reorder_cache_hlo_kernels:
                all_kernels.append(hlo_kernel.kernel)
        return all_kernels

class DecoderProgramMultiLayer(DecoderProgram):

    def __init__(self, neuron_config, layers, ode_hlo_modules, ode_num_inputs, hlo_modules, debug_tensors, ln_lm_head_hlo_modules, num_inputs, num_layers, unroll, tp_degree, n_positions_list, batch_sizes, prefixed_length=0, batch_size_for_shared_caches=None,tag=None,on_cpu=False):
        if num_layers % unroll:
            raise ValueError(f'unroll={unroll} does not divide num_layers={num_layers}')
        self.num_exec_repetition = num_layers // unroll
        super().__init__(neuron_config, layers, hlo_modules, debug_tensors, num_inputs, tp_degree, n_positions_list, batch_sizes, prefixed_length, batch_size_for_shared_caches, tag=tag, num_exec_repetition=self.num_exec_repetition, on_cpu=on_cpu)
        self.num_layers = num_layers
        assert len(ln_lm_head_hlo_modules) == len(batch_sizes)
        if self.neuron_config.log_softmax_scores or self.neuron_config.is_eagle_target:
            self.logits_buffer = [[compiler.gen_zero_output(hm, 0), compiler.gen_zero_output(hm, 1)] for hm in ln_lm_head_hlo_modules]
        else:
            self.logits_buffer = [compiler.gen_zero_output(hm) for hm in ln_lm_head_hlo_modules]
        self.unroll = unroll
        self.ln_lm_head_hlo_modules = ln_lm_head_hlo_modules

        self.ln_lm_head_kernels = [compiler.ParallelKernel(hm, self.neuron_config.get_local_tp(tp_degree), self.neuron_config.get_g_start_device_id(tp_degree), self.neuron_config.get_g_device_count(tp_degree)) for hm in ln_lm_head_hlo_modules]
        self.layer_executors = list()
        self.lm_head_executors = list()
        self.ode_kernels = []
        if self.neuron_config.on_device_embedding:
            for ode_hlo in ode_hlo_modules:
                self.ode_kernels.append(compiler.ParallelKernel(ode_hlo, self.neuron_config.get_local_tp(tp_degree), self.neuron_config.get_g_start_device_id(tp_degree), self.neuron_config.get_g_device_count(tp_degree), tag=f"ode-hlo"))
            self.ode_hlo_modules = ode_hlo_modules
            self.input_ids_buffer = []
            for i in range(len(ode_hlo_modules)):
                # While there is more than one input for this hlo, the rest of them will be shared with the input_buffers
                self.input_ids_buffer.append([compiler.gen_zero_input(ode_hlo_modules[i], 0)])
            self.ode_executors = []

    def setup(self, layers, pre_layer_params, ln_lm_head_params):
        super().setup(layers, pre_layer_params, ln_lm_head_params, io_ring_cache_size=self.num_exec_repetition)

        if self.neuron_config.on_device_embedding:
            for i in range(len(self.input_ids_buffer)):
                # Share the buffers here so we only need to move the inputs from host to device once (separate on-device embedding NEFF)
                self.input_ids_buffer[i] = [self.manipulator.duplicate(self.input_ids_buffer[i][0])] + self.input_buffers[i][1:]
            for kernel in self.ode_kernels:
                kernel.load()

            self.ode_memories = [ode_kernel.build_memory() for ode_kernel in self.ode_kernels]
            for ode, inp_ids, inp in zip(self.ode_memories, self.input_ids_buffer, self.input_buffers):
                ode_input_tensors = [*inp_ids]
                for weight in pre_layer_params:
                    ode_input_tensors.append(weight)
                # make the output of the ODE NEFF the input to the layers NEFF
                ode.setup(ode_input_tensors, [inp[0]])

        self.multi_layers_memories = []
        for _ in range(self.num_layers // self.unroll):
            memories = dict()
            for npos, batch_size in itertools.product(self.n_positions_list, self.batch_sizes):
                memories[npos,batch_size] = self.kernels[npos,batch_size].build_memory()
            self.multi_layers_memories.append(memories)

        self.ln_lm_head_memories = [ln_lm_head_kernel.build_memory() for ln_lm_head_kernel in self.ln_lm_head_kernels]

        hidden_buffers = list()
        last_token_id_buffers = list()
        for input_buffer in self.input_buffers:
            hidden_buffer, _, _, last_token_id_buffer, *_ = input_buffer
            hidden_buffers.append(hidden_buffer)
            last_token_id_buffers.append(last_token_id_buffer)

        multi_layer_starts = range(0, len(layers), self.unroll)
        multi_layers = [layers[start:start+self.unroll] for start in multi_layer_starts]

        for multi_layer_idx, multi_layer in enumerate(multi_layers):
            multi_layer_memory = self.multi_layers_memories[multi_layer_idx]
            for bs_idx, batch_size in enumerate(self.batch_sizes):
                for npos in self.n_positions_list:
                    input_tensors = [*self.input_buffers[bs_idx]]
                    output_tensors = [hidden_buffers[bs_idx]]
                    self._fill_io_tensors(input_tensors, output_tensors, multi_layer, npos, batch_size)
                    self._fill_debug_tensors(output_tensors, npos, batch_size)
                    input_tensors.extend(pre_layer_params)
                    multi_layer_memory[npos,batch_size].setup(input_tensors, output_tensors)

        if self.neuron_config.is_valid_lm_head():
            for head_idx in range(0,len(self.ln_lm_head_kernels)):
                output_tensors = [*self.logits_buffer[head_idx]] if self.neuron_config.log_softmax_scores or self.neuron_config.is_eagle_target else [self.logits_buffer[head_idx]]
                self.ln_lm_head_memories[head_idx].setup([hidden_buffers[head_idx], last_token_id_buffers[head_idx], *ln_lm_head_params], output_tensors)
                self.ln_lm_head_kernels[head_idx].build()
                self.ln_lm_head_kernels[head_idx].load()

        # Warmup kernels to avoid unexpected initialization at runtime
        for kernel in self.get_kernels():
            kernel.warmup()

    def run(self, bucket_id, batch_size):
        npos = self.n_positions_list[bucket_id]
        bs_idx = self.batch_sizes.index(batch_size)
        if self.neuron_config.on_device_embedding:
            self.ode_kernels[bs_idx](self.ode_memories[bs_idx])

        for memories in self.multi_layers_memories:
            self.kernels[npos,batch_size](memories[npos,batch_size])
        if self.neuron_config.is_valid_lm_head():
            self.ln_lm_head_kernels[bs_idx](self.ln_lm_head_memories[bs_idx])

    def enable_executor(self):
        if self.neuron_config.on_device_embedding:
            for i, (ode_memory, ode_kernel) in enumerate(zip(self.ode_memories, self.ode_kernels)):
                # Make the output of the ODE kernel share the same buffer as the input to the layer kernel
                self.ode_executors.append(ode_kernel.build_executor(ode_memory, [*self.input_ids_buffer[i]], [self.input_buffers[i][0]]))
        for layer_memories in self.multi_layers_memories:
            executors = dict()
            self.layer_executors.append(executors)
            for npos, batch_size in itertools.product(self.n_positions_list, self.batch_sizes):
                executors[npos,batch_size] = self.kernels[npos,batch_size].build_executor(layer_memories[npos,batch_size], [], [])
        # head
        for idx, kernel in enumerate(self.ln_lm_head_kernels):
            output_tensors = [self.logits_buffer[idx]]
            executor = kernel.build_executor(self.ln_lm_head_memories[idx], [], output_tensors)
            self.lm_head_executors.append(executor)

    def execute(self, bucket_id, batch_size, *inputs, return_ranks=-1):
        self.inputs_host_to_device(inputs, batch_size) # One-time input copy
        if self.neuron_config.on_device_embedding:
            self.ode_executors[self.batch_sizes.index(batch_size)]([], return_ranks=return_ranks)
        npos = self.n_positions_list[bucket_id]
        for layer_executor in self.layer_executors:
            layer_executor[npos,batch_size]([], return_ranks=0)
        return self.lm_head_executors[self.batch_sizes.index(batch_size)]([], return_ranks=return_ranks)

    def get_kernels(self):
        all_kernels = super().get_kernels()
        # Head kernel
        if self.neuron_config.is_valid_lm_head():
            for kernel in self.ln_lm_head_kernels:
                all_kernels.append(kernel)
        # only true when reorder_cache called before to_neuron
        if self.need_reorder_cache:
            for hlo_kernel in self.reorder_cache_hlo_kernels:
                all_kernels.append(hlo_kernel.kernel)
        if self.neuron_config.on_device_embedding:
            for kernel in self.ode_kernels:
                all_kernels.append(kernel)
        return all_kernels


def sync_tensor_program(sizes, element_dtype, replica_groups):

    def _sync_tensor_program(scribe):
            dtype = scribe.get_dtype(element_dtype)
            tensor = dtype[sizes].Parameter(parameter_number=0)
            add_func = hlo.gen_add_func(dtype)
            tensor = dtype[sizes].AllReduce(tensor, replica_groups=replica_groups, to_apply=add_func)
            return tensor

    return _sync_tensor_program


class PipelineParallelProgram(DecoderProgramMultiLayer):


    def init_pp_sync_programs(self):

        self.send_hidden_kernels = {}
        self.recv_hidden_kernels = {}

        self.send_logits_kernels = {}
        self.recv_logits_kernels = {}

        def replica_groups_helper(src_rank_id, dst_rank_id, tp):
            return [list(sorted([src_rank_id*tp+i, dst_rank_id*tp+i])) for i in range(tp)]

        for batch_idx, batch_size in enumerate(self.batch_sizes):

            for (npos, b), kernel in self.kernels.items():
                if b == batch_size:
                    break

            hidden = kernel.hlo_module.host_program_shape.parameters[0]

            hidden_sizes = hidden.dimensions
            hidden_dtype = compiler.primitive2name(hidden.element_type)

            logits = self.ln_lm_head_hlo_modules[batch_idx].host_program_shape.result

            logits_sizes = logits.dimensions
            logits_dtype = compiler.primitive2name(logits.element_type)

            if self.neuron_config.first_rank():
                # setup send hidden
                send_hidden_hlo = sync_tensor_program(hidden_sizes, hidden_dtype, replica_groups=replica_groups_helper(self.neuron_config.rank_id, self.neuron_config.rank_id+1, self.tp_degree))
                self.send_hidden_kernels[batch_size] = compiler.HLOKernel(send_hidden_hlo, self.tp_degree, self.neuron_config.rank_id*self.tp_degree, self.neuron_config.pp_stages*self.tp_degree, tag=self.tag)
                # setup receive logits
                recv_logits_hlo = sync_tensor_program(logits_sizes, logits_dtype, replica_groups=replica_groups_helper(self.neuron_config.pp_stages-1, self.neuron_config.rank_id, self.tp_degree))
                self.recv_logits_kernels[batch_size] = compiler.HLOKernel(recv_logits_hlo, self.tp_degree, self.neuron_config.rank_id*self.tp_degree, self.neuron_config.pp_stages*self.tp_degree, tag=self.tag)

            elif self.neuron_config.last_rank():
                # setup receive hidden
                recv_hidden_hlo = sync_tensor_program(hidden_sizes, hidden_dtype, replica_groups=replica_groups_helper(self.neuron_config.rank_id-1, self.neuron_config.rank_id, self.tp_degree))
                self.recv_hidden_kernels[batch_size] = compiler.HLOKernel(recv_hidden_hlo, self.tp_degree, self.neuron_config.rank_id*self.tp_degree, self.neuron_config.pp_stages*self.tp_degree, tag=self.tag)
                # setup send logits
                send_logits_hlo = sync_tensor_program(logits_sizes, logits_dtype, replica_groups=replica_groups_helper(self.neuron_config.rank_id, 0, self.tp_degree))
                self.send_logits_kernels[batch_size] = compiler.HLOKernel(send_logits_hlo, self.tp_degree, self.neuron_config.rank_id*self.tp_degree, self.neuron_config.pp_stages*self.tp_degree, tag=self.tag)
            else:
                # setup receive hidden
                recv_hidden_hlo = sync_tensor_program(hidden_sizes, hidden_dtype, replica_groups=replica_groups_helper(self.neuron_config.rank_id-1, self.neuron_config.rank_id, self.tp_degree))
                self.recv_hidden_kernels[batch_size] = compiler.HLOKernel(recv_hidden_hlo, self.tp_degree, self.neuron_config.rank_id*self.tp_degree, self.neuron_config.pp_stages*self.tp_degree, tag=self.tag)
                # setup send hidden
                send_hidden_hlo = sync_tensor_program(hidden_sizes, hidden_dtype, replica_groups=replica_groups_helper(self.neuron_config.rank_id, self.neuron_config.rank_id+1, self.tp_degree))
                self.send_hidden_kernels[batch_size] = compiler.HLOKernel(send_hidden_hlo, self.tp_degree, self.neuron_config.rank_id*self.tp_degree, self.neuron_config.pp_stages*self.tp_degree, tag=self.tag)
                pass

    def setup(self, layers, pre_layer_parameters, ln_lm_head_params):
        super().setup(layers, pre_layer_parameters, ln_lm_head_params)
        self.setup_pp_sync_programs()


    def get_pp_sync_kernels(self):
        get_kernels = lambda x: list(x.values())
        return get_kernels(self.send_hidden_kernels) + get_kernels(self.recv_hidden_kernels) + get_kernels(self.send_logits_kernels) + get_kernels(self.recv_logits_kernels)

    def setup_pp_sync_programs(self):

        for batch_idx, batch_size in enumerate(self.batch_sizes):
            if self.neuron_config.first_rank():
                # setup send hidden, only send from last layer group
                for (npos, batch), memory in self.multi_layers_memories[-1].items():
                    if batch == batch_size:
                        self.send_hidden_kernels[batch_size].setup([memory.output_tensors[0]], [])
                # setup recv logits, only recv at first layer group
                self.recv_logits_kernels[batch_size].setup([], [self.logits_buffer[batch_idx]])
            elif self.neuron_config.last_rank():
                # setup receive hidden, only recv at first layer group
                for (npos, batch), memory in self.multi_layers_memories[0].items():
                    if batch == batch_size:
                        self.recv_hidden_kernels[batch_size].setup([], [memory.input_tensors[0]])
                # setup send logits
                self.send_logits_kernels[batch_size].setup([self.logits_buffer[batch_idx]], [])
            else:
                # setup receive hidden, only recv at first layer group
                for (npos, batch), memory in self.multi_layers_memories[0].items():
                    if batch == batch_size:
                        self.recv_hidden_kernels[batch_size].setup([], [memory.input_tensors[0]])
                # set up send hidden, only send from last layer group
                for (npos, batch), memory in self.multi_layers_memories[-1].items():
                    if batch == batch_size:
                        self.send_hidden_kernels[batch_size].setup([memory.output_tensors[0]], [])


        for kernel in self.get_pp_sync_kernels():
            kernel.build()
            kernel.load()

    def run(self, bucket_id, batch_size):
        self.maybe_receive_hidden(bucket_id, batch_size)
        super().run(bucket_id, batch_size)
        self.maybe_send_hidden(bucket_id, batch_size)
        self.maybe_send_logits(bucket_id, batch_size)
        self.maybe_receive_logits(bucket_id, batch_size)

    def maybe_send_hidden(self, bucket_id, batch_size):
        if not self.neuron_config.last_rank():
            logging.debug("Running send_hidden")
            self.send_hidden_kernels[batch_size].run()

    def maybe_receive_hidden(self, bucket_id, batch_size):
        if not self.neuron_config.first_rank():
            logging.debug("Running recev_hidden")
            self.recv_hidden_kernels[batch_size].run()

    def maybe_receive_logits(self, bucket_id, batch_size):
        if self.neuron_config.first_rank():
            logging.debug("Running receiv_logits")
            self.recv_logits_kernels[batch_size].run()

    def maybe_send_logits(self, bucket_id, batch_size):
        if self.neuron_config.last_rank():
            logging.debug("Running send_logits")
            self.send_logits_kernels[batch_size].run()

class FastCacheBroadcaster(base.NeuronBaseSerializer):

    def __init__(self, n_positions, from_batch_size, to_batch_size, n_heads_tp, d_head, amp,
                 tp_degree, n_layer):
        cache_broadcast_impl = hlo.cache_broadcast(n_positions, from_batch_size, to_batch_size,
                                                   n_heads_tp, d_head, amp, n_layer)
        cache_broadcast_hlo_module = compiler.compile_py_func(cache_broadcast_impl)
        self.cache_broadcast_kernel = compiler.ParallelKernel(cache_broadcast_hlo_module, tp_degree)
        self._source_caches = None
        self._target_caches = None

    def set_source_caches(self, source_caches):
        self._source_caches = source_caches

    def set_target_caches(self, target_caches):
        self._target_caches = target_caches

    def setup(self):
        assert self._source_caches is not None and self._target_caches is not None, "need to call set_source_caches and set_target_caches"
        self.cache_broadcast_memory = self.cache_broadcast_kernel.build_memory()
        self.cache_broadcast_kernel.load()
        self.cache_broadcast_memory.setup(self._source_caches, self._target_caches)

    def run_broadcast(self):
        self.cache_broadcast_kernel(self.cache_broadcast_memory)

    def get_all_kernels(self):
        return [self.cache_broadcast_kernel]

# loads weights from safetensors files and puts them on neuron while assigning
# them to the corresponding attribute of decoder_layer_or_head
def presharded_weights_to_neuron(safetensors_file, decoder_layer_or_head, attr_names):
    for attr_name in attr_names:
        shards = []
        i = 0
        while True:
            if f"{attr_name}*{i}" not in safetensors_file.keys():
                break
            shards.append(safetensors_file.get_tensor(f"{attr_name}*{i}"))
            i+=1
        if all([torch.numel(shard) == 0 for shard in shards]):
            setattr(decoder_layer_or_head, attr_name, 'None')
        else:
            setattr(decoder_layer_or_head, attr_name, ops.parallel_to_nc(shards))

def get_attribute_names(safetensors_file):
    attr_names = set()
    for key in safetensors_file.keys():
        attr_name, shard_id = key.split('*')
        attr_names.add(attr_name)

    return attr_names


