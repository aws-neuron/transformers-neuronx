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
import pickle
import os
import torch
from transformers_neuronx import compiler
from transformers_neuronx import dtypes
from transformers_neuronx import hlo
from transformers_neuronx import ops
from transformers_neuronx import parallel
from transformers_neuronx import utils
from transformers_neuronx import quantize
from concurrent.futures import ProcessPoolExecutor

class DecoderLmHeadForSamplingNoEmbedding(torch.nn.Module):

    def __init__(self, tp_degree, n_positions_list, n_active_tokens, batch_size,
                 attention_head_size, amp, num_layers, unroll=None, neuron_config=None, allow_pad=True):
        super().__init__()
        if unroll is None:
            unroll = num_layers
        self.tp_degree = tp_degree
        self.n_positions_list = n_positions_list
        self.n_active_tokens = n_active_tokens
        self.batch_size = batch_size
        self.attention_head_size = attention_head_size
        self.amp = amp
        self.num_layers = num_layers
        self.unroll = unroll
        self.neuron_config = neuron_config
        self.layers = torch.nn.ModuleList()
        self.ln_f_weight = None
        self.ln_f_bias = None
        self.lm_head_weight = None
        self.lm_head_bias = None
        self.inputs_sdim = None
        self.inputs_builder = None
        self.layer_builder = None
        self.ln_lm_head_builder = None
        self.program = None
        self.compiler_artifacts_path = None
        self.pre_layer_parameters = []
        self.pre_layer_builder = None
        self.allow_pad = allow_pad
        self.use_executor = False
        self.return_ranks = -1

    def enable_executor(self, return_ranks=-1):
        self.use_executor = True
        self.return_ranks = return_ranks
        self.program.enable_executor()

    def add_inputs_builder(self, inputs_builder):
        self.inputs_builder = inputs_builder

    def add_pre_layer_parameter(self, param, sharding=None, allow_pad=False):
        self.pre_layer_parameters.append((param, sharding, allow_pad))

    def add_pre_layer_builder(self, builder):
        self.pre_layer_builder = builder

    def add_layer_builder(self, layer_builder):
        self.layer_builder = layer_builder

    def add_ln_lm_head_builder(self, ln_lm_head_builder):
        self.ln_lm_head_builder = ln_lm_head_builder

    def new_layer(self):
        *_, n_positions = self.n_positions_list
        layer = DecoderLayer(self.tp_degree, n_positions, self.batch_size, self.attention_head_size, self.amp, self.neuron_config, self.allow_pad)
        self.layers.append(layer)
        return layer

    def add_final_layer_norm(self, weight, bias):
        self.ln_f_weight = weight
        self.ln_f_bias = bias

    def add_lm_head(self, weight, bias=None):
        self.lm_head_weight = weight
        self.lm_head_bias = bias

    def to_neuron(self):
        manipulator = MaybeParallelTensorManipulator(self.tp_degree)

        extras = []
        for param, dim, allow_pad in self.pre_layer_parameters:
            if allow_pad:
                if param.shape[dim] % self.tp_degree != 0:
                    size = utils.round_up_to_divisor(param.shape[dim], self.tp_degree)
                    param = utils.pad(param, dim, size)
            extras.append(manipulator.duplicate_or_shard_along(param, dim))
        self.pre_layer_parameters = extras

        self.ln_f_weight = manipulator.duplicate(self.ln_f_weight)
        self.ln_f_bias = manipulator.duplicate(self.ln_f_bias)
        _, vocab_size = self.lm_head_weight.shape
        # Pad vocab size such that it can be divided by the following factor
        divisor = int(os.environ.get('NEURON_VOCAB_PAD_DIVISOR', str(self.tp_degree)))
        vocab_pad = utils.pad_vocab_size(vocab_size, divisor)
        lm_head_weight = torch.nn.functional.pad(self.lm_head_weight, (0, vocab_pad, 0, 0))
        self.lm_head_weight = manipulator.shard_along(lm_head_weight, dim=1)
        ln_lm_head_params = [*self.pre_layer_parameters, self.ln_f_weight, self.ln_f_bias, self.lm_head_weight]
        ln_lm_head_params = [param for param in ln_lm_head_params if param is not None]
        if self.lm_head_bias is not None:
            self.lm_head_bias = manipulator.shard_along(self.lm_head_bias, dim=0)
            ln_lm_head_params.append(self.lm_head_bias)

        self.program = self._build_program()
        self.program.setup(self.layers, ln_lm_head_params)

    def build_weight_shared(self, n_positions_list=None, n_active_tokens=None, batch_size=None,
                            unroll=None, share_caches=False):
        if n_positions_list is None:
            n_positions_list = self.n_positions_list
        if n_active_tokens is None:
            n_active_tokens = self.n_active_tokens
        if batch_size is None:
            batch_size = self.batch_size
        if unroll is None:
            unroll = self.unroll
        new = DecoderLmHeadForSamplingNoEmbedding(
            self.tp_degree, n_positions_list, n_active_tokens, batch_size, self.attention_head_size,
            self.amp, self.num_layers, unroll, neuron_config=self.neuron_config, allow_pad=self.allow_pad
        )
        new.add_inputs_builder(self.inputs_builder)
        new.add_pre_layer_builder(self.pre_layer_builder)
        new.add_layer_builder(self.layer_builder)
        new.add_ln_lm_head_builder(self.ln_lm_head_builder)
        for layer in self.layers:
            new_layer = new.new_layer()
            new_layer.assign_parameters(layer)
            if share_caches:
                new_layer.assign_caches(layer)
            else:
                new_layer.init_caches()
            new_layer.extra_parameters = layer.extra_parameters
        new.pre_layer_parameters = self.pre_layer_parameters
        new.add_final_layer_norm(self.ln_f_weight, self.ln_f_bias)
        new.add_lm_head(self.lm_head_weight, self.lm_head_bias)
        ln_lm_head_params = [*new.pre_layer_parameters, new.ln_f_weight, new.ln_f_bias, new.lm_head_weight]
        ln_lm_head_params = [param for param in ln_lm_head_params if param is not None]
        if new.lm_head_bias is not None:
            ln_lm_head_params.append(new.lm_head_bias)
        new.program = new._build_program()
        new.program.setup(new.layers, ln_lm_head_params)
        return new

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
        _, cache_ids, *_ = inputs
        bucket_id = self.program.find_bucket_id(cache_ids.item())
        if self.use_executor:
            return self.program.execute(bucket_id, *inputs, return_ranks=self.return_ranks)
        else:
            self.program.inputs_host_to_device(inputs)
            self.program.run(bucket_id)
            return self.program.logits_device_to_host()

    def forward(self, *inputs):
        hidden, *_ = inputs
        sequence_dim, *_ = self.inputs_sdim
        sequence_length = hidden.shape[sequence_dim]
        if sequence_length == 1:
            return self.forward_single(*inputs)
        if sequence_length % self.n_active_tokens:
            raise ValueError(f'sequence_length={sequence_length} cannot be divided by '
                             f'n_active_tokens={self.n_active_tokens}')
        outputs = None
        for start in range(0, sequence_length, self.n_active_tokens):
            slicing = slice(start, start + self.n_active_tokens)
            input_tensors = []
            for sdim, tensor in zip(self.inputs_sdim, inputs):
                if sdim is not None:
                    slices = [slice(None) for _ in tensor.shape]
                    slices[sdim] = slicing
                    tensor = tensor[tuple(slices)].contiguous()
                input_tensors.append(tensor)
            _, cache_ids, *_ = input_tensors
            max_id = cache_ids.max().item()
            min_id = cache_ids.min().item()
            # When context_length == m * n_active_tokens, bucket-size of n_active_tokens should be chosen.
            # This is useful for Fusion-In-Decoder case, where 2nd n_active_tokens don't need to attend to
            # 1st n_active_tokens.
            if max_id - min_id > 1:
                max_id -= min_id
                min_id = 0
            bucket_id = self.program.find_bucket_id(max_id)
            if self.program.find_bucket_id(min_id) != bucket_id:
                raise ValueError(f'given buckets {self.n_positions_list}, ids ranging from '
                                 f'{min_id} to {max_id} do not fall into the same bucket')
            if self.use_executor:
                outputs = self.program.execute(bucket_id, *input_tensors, return_ranks=self.return_ranks)
            else:
                self.program.inputs_host_to_device(input_tensors)
                self.program.run(bucket_id)

        if not self.use_executor:
            outputs = self.program.logits_device_to_host()
        return outputs

    def embed_positions_ids(self, position_ids, start_ids=None):
        batch_size = self.batch_size
        if start_ids is None:
            return position_ids, torch.zeros([batch_size], dtype=torch.int32)
        position_ids = position_ids.unsqueeze(0).repeat(batch_size, 1)
        position_ids -= start_ids.unsqueeze(1)
        position_ids.masked_fill_(position_ids < 0, 0)
        return position_ids, start_ids

    def save_compiler_artifacts(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.program.get_neff_bytes(), f)

    def load_compiler_artifacts_after_build(self, path):
        self.compiler_artifacts_path = path

    def _build_program(self):
        if self.unroll == self.num_layers:
            hlo_modules = [self._hlo_fully_unrolled(npos) for npos in self.n_positions_list]
            num_inputs = len(self.inputs_sdim)
            program = DecoderProgramFullyUnrolled(self.layers, hlo_modules, num_inputs, self.tp_degree)
        else:
            if utils.amp_is_u8(self.amp):
                raise NotImplementedError(f'amp={self.amp} only supports fully unrolled decoder')
            hlo_modules = [self._hlo_multi_layer(npos) for npos in self.n_positions_list]
            ln_lm_head_hlo_module = self._hlo_ln_lm_head()
            num_inputs = len(self.inputs_sdim)
            program = DecoderProgramMultiLayer(self.layers, hlo_modules, ln_lm_head_hlo_module, num_inputs,
                                               self.num_layers, self.unroll, self.tp_degree)
        if self.compiler_artifacts_path is not None:
            with open(self.compiler_artifacts_path, 'rb') as f:
                kernels_neff_bytes = pickle.load(f)
            program.set_neff_bytes(kernels_neff_bytes)
        return program

    def _hlo_fully_unrolled(self, n_positions):

        def fully_unrolled(scribe):
            amp, quantized, dequantized = utils.parse_amp(self.amp)
            dtype = getattr(scribe, amp)
            (hidden, *tensors), self.inputs_sdim = self.inputs_builder(
                scribe, dtype, n_positions, self.n_active_tokens, self.batch_size)
            param_builder = DecoderParameterBuilder(scribe, len(self.inputs_sdim))
            layers_caches, layers_weights = self._hlo_layers_params(param_builder, self.layers, n_positions)
            hidden, tensors = self._hlo_pre_layer(hidden, tensors, param_builder)
            ln_f_weight = param_builder.from_tensor(self.ln_f_weight)
            ln_f_bias = param_builder.from_tensor(self.ln_f_bias)
            head_weight = param_builder.from_tensor(self.lm_head_weight)
            head_bias = param_builder.from_tensor(self.lm_head_bias)
            hidden, out_caches = self._hlo_layers(hidden, tensors, self.layers, layers_caches, layers_weights)
            ln_f_weight = maybe_transfer_with_static_ring(ln_f_weight)
            ln_f_bias = maybe_transfer_with_static_ring(ln_f_bias)
            head_weight = maybe_transfer_with_static_ring(head_weight)
            head_bias = maybe_transfer_with_static_ring(head_bias)
            logits = self.ln_lm_head_builder(hidden, ln_f_weight, ln_f_bias, head_weight, head_bias)
            outputs = [logits, *out_caches]
            root_shapes = [shape.dtype[shape.sizes] for shape in outputs]
            return scribe.tuple(*root_shapes).Tuple(*outputs)

        return compiler.compile_py_func(fully_unrolled)

    def _hlo_multi_layer(self, n_positions):

        def multi_layer(scribe):
            dtype = getattr(scribe, self.amp)
            (hidden, *tensors), self.inputs_sdim = self.inputs_builder(
                scribe, dtype, n_positions, self.n_active_tokens, self.batch_size)
            param_builder = DecoderParameterBuilder(scribe, len(self.inputs_sdim))
            # use the first `unroll` layers to build the HLO -- assuming all layers are same
            layers = self.layers[:self.unroll]
            layers_caches, layers_weights = self._hlo_layers_params(param_builder, layers, n_positions)
            hidden, tensors = self._hlo_pre_layer(hidden, tensors, param_builder)
            out_hidden, out_caches = self._hlo_layers(hidden, tensors, layers, layers_caches, layers_weights)
            out_hidden.set_alias_to(hidden)
            outputs = [out_hidden, *out_caches]
            root_shapes = [shape.dtype[shape.sizes] for shape in outputs]
            return scribe.tuple(*root_shapes).Tuple(*outputs)

        return compiler.compile_py_func(multi_layer)

    def _hlo_pre_layer(self, hidden, tensors, param_builder):
        params = []
        if self.pre_layer_builder is not None:
            for param in self.pre_layer_parameters:
                param = param_builder.from_tensor(param)
                param = hlo.transfer_with_static_ring(param)
                params.append(param)
            (hidden, *tensors) = self.pre_layer_builder(hidden, *tensors, *params)
        return hidden, tensors

    def _hlo_layers_params(self, param_builder, layers, n_positions):
        layers_caches = []
        for layer in layers:
            layer_caches = []
            for cache in layer.attn_k_cache, layer.attn_v_cache:
                par = param_builder.from_tensor(cache, dim_size={0: n_positions})
                layer_caches.append(par)
            layers_caches.append(layer_caches)
        layers_weights = []
        for layer in layers:
            layer_weights = [param_builder.from_tensor(weight) for weight in layer.all_parameters()]
            layers_weights.append(layer_weights)
        return layers_caches, layers_weights

    def _hlo_layers(self, hidden, tensors, layers, layers_caches, layers_weights):
        output_caches = []
        for layer, caches, weights in zip(layers, layers_caches, layers_weights):
            in_caches = [hlo.transfer_with_static_ring(cache) for cache in caches]
            weights = [maybe_transfer_with_static_ring(weight) for weight in weights]
            weights = layer.hlo_maybe_dequantize_weights(weights)
            hidden, *out_caches = self.layer_builder(hidden, *tensors, *in_caches, *weights)
            for out_cache, cache in zip(out_caches, caches):
                out_cache.set_alias_to(cache, must=True)
            output_caches.extend(out_caches)
        return hidden, output_caches

    def _hlo_ln_lm_head(self):
        hidden_sizes = []

        def capture_hidden_sizes(scribe):
            dtype = getattr(scribe, self.amp)
            *_, n_positions = self.n_positions_list
            (hidden, *_), _ = self.inputs_builder(
                scribe, dtype, n_positions, self.n_active_tokens, self.batch_size)
            hidden_sizes.clear()
            hidden_sizes.extend(hidden.sizes)
            return hidden

        compiler.compile_py_func(capture_hidden_sizes)

        def ln_lm_head(scribe):
            dtype = getattr(scribe, self.amp)
            hidden = dtype[tuple(hidden_sizes)].Parameter(parameter_number=0)
            param_builder = DecoderParameterBuilder(scribe, 1)
            ln_f_weight = param_builder.from_tensor(self.ln_f_weight)
            ln_f_bias = param_builder.from_tensor(self.ln_f_bias)
            head_weight = param_builder.from_tensor(self.lm_head_weight)
            head_bias = param_builder.from_tensor(self.lm_head_bias)
            return self.ln_lm_head_builder(hidden, ln_f_weight, ln_f_bias, head_weight, head_bias)

        return compiler.compile_py_func(ln_lm_head)

    def reorder_cache(self, reorder_ids):
        self.program.reorder_cache(reorder_ids)


    def setup_reorder_cache(self):
        self.program.setup_reorder_cache()


def read_n_position(hlo_module, num_inputs):
    return hlo_module.host_program_shape.parameters[num_inputs].dimensions[0]


def read_n_active_tokens(hlo_module):
    return hlo_module.host_program_shape.parameters[0].dimensions[1]


def maybe_transfer_with_static_ring(shape):
    if shape is None:
        return None
    return hlo.transfer_with_static_ring(shape)


class MaybePadder:

    def __init__(self, size) -> None:
        self.size = size

    def __call__(self, weight, dim):
        return utils.pad(weight, dim, self.size)


class DecoderLayer(torch.nn.Module):

    def __init__(self, tp_degree, n_positions, batch_size, attention_head_size, amp, neuron_config=None, allow_pad=False):
        super().__init__()
        self.pre_attn_ln_weight = None
        self.pre_attn_ln_bias = None
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
        self.attn_k_cache = None
        self.attn_v_cache = None
        self.tp_degree = tp_degree
        self.n_positions = n_positions
        self.batch_size = batch_size
        self.attention_head_size = attention_head_size
        self.amp = amp
        dtype, _, _ = utils.parse_amp(amp)
        self.cache_dtype = dtypes.to_torch_dtype(dtype)
        self.neuron_config = neuron_config
        self.extra_parameters = []
        self.allow_pad = allow_pad

    def add_parameter(self, param, sharding=None, allow_pad=False, allow_quantize=False):
        self.extra_parameters.append((param, sharding, allow_pad, allow_quantize))

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

    def add_attention_output(self, weight, bias, sharding=0):
        self.attn_out_weight = weight
        self.attn_out_bias = bias
        self.attn_out_sharding = sharding

    def add_post_attention_layer_norm(self, weight, bias):
        self.post_attn_ln_weight = weight
        self.post_attn_ln_bias = bias

    def add_pre_mlp_layer_norm(self, weight, bias):
        self.pre_mlp_ln_weight = weight
        self.pre_mlp_ln_bias = bias

    def add_mlp_input(self, weight, bias):
        self.mlp_in_weight = weight
        self.mlp_in_bias = bias

    def add_mlp_output(self, weight, bias):
        self.mlp_out_weight = weight
        self.mlp_out_bias = bias

    def add_post_mlp_layer_norm(self, weight, bias):
        self.post_mlp_ln_weight = weight
        self.post_mlp_ln_bias = bias

    def to_neuron(self):

        # If we allow padding then we need to pad non-sharded QKV weight dimensions
        if self.allow_pad:

            # Hidden size padding
            hidden_size, _ = self.attn_q_weight.shape
            n_heads = hidden_size // self.attention_head_size
            n_heads_padded = utils.round_up_to_divisor(n_heads, self.tp_degree)
            hidden_size_padded = n_heads_padded * self.attention_head_size
            maybe_pad = MaybePadder(hidden_size_padded)

            self.attn_q_weight = maybe_pad(self.attn_q_weight, dim=1)
            self.attn_q_bias = maybe_pad(self.attn_q_bias, dim=0)

            self.attn_k_weight = maybe_pad(self.attn_k_weight, dim=1)
            self.attn_k_bias = maybe_pad(self.attn_k_bias, dim=0)

            self.attn_v_weight = maybe_pad(self.attn_v_weight, dim=1)
            self.attn_v_bias = maybe_pad(self.attn_v_bias, dim=0)

            self.attn_out_weight = maybe_pad(self.attn_out_weight, dim=self.attn_out_sharding)

            # Intermediate MLP layer padding
            if self.mlp_in_weight is not None:
                _, intermediate_size = self.mlp_in_weight.shape
                intermediate_size_padded = utils.round_up_to_divisor(intermediate_size, self.tp_degree)
                maybe_pad = MaybePadder(intermediate_size_padded)

                self.mlp_in_weight = maybe_pad(self.mlp_in_weight, dim=1)
                self.mlp_in_bias = maybe_pad(self.mlp_in_bias, dim=0)
                self.mlp_out_weight = maybe_pad(self.mlp_out_weight, dim=0)

        if utils.amp_is_u8(self.amp):
            self.attn_q_weight, self.attn_q_min, self.attn_q_max = utils.u8_encode(self.attn_q_weight)
            self.attn_k_weight, self.attn_k_min, self.attn_k_max = utils.u8_encode(self.attn_k_weight)
            self.attn_v_weight, self.attn_v_min, self.attn_v_max = utils.u8_encode(self.attn_v_weight)
            self.attn_out_weight, self.attn_out_min, self.attn_out_max = utils.u8_encode(self.attn_out_weight)
            self.mlp_in_weight, self.mlp_in_min, self.mlp_in_max = utils.u8_encode(self.mlp_in_weight)
            self.mlp_out_weight, self.mlp_out_min, self.mlp_out_max = utils.u8_encode(self.mlp_out_weight)
        if self.neuron_config and self.neuron_config.quant:
            self.mlp_in_weight, self.mlp_in_scales = \
                quantize.maybe_quantize_weights(self.mlp_in_weight, self.neuron_config.quant)
            self.mlp_out_weight, self.mlp_out_scales = \
                quantize.maybe_quantize_weights(self.mlp_out_weight, self.neuron_config.quant)

            if self.neuron_config.quant.quantize_attn:
                self.attn_q_weight, self.attn_q_scales = \
                    quantize.maybe_quantize_weights(self.attn_q_weight, self.neuron_config.quant)
                self.attn_k_weight, self.attn_k_scales = \
                    quantize.maybe_quantize_weights(self.attn_k_weight, self.neuron_config.quant)
                self.attn_v_weight, self.attn_v_scales = \
                    quantize.maybe_quantize_weights(self.attn_v_weight, self.neuron_config.quant)
                self.attn_out_weight, self.attn_out_scales = \
                    quantize.maybe_quantize_weights(self.attn_out_weight, self.neuron_config.quant)


        maybe_manipulator = MaybeParallelTensorManipulator(self.tp_degree)
        maybe_duplicate = maybe_manipulator.duplicate
        maybe_shard_along = maybe_manipulator.shard_along
        maybe_primary_only = maybe_manipulator.primary_only
        self.pre_attn_ln_weight = maybe_duplicate(self.pre_attn_ln_weight)
        self.pre_attn_ln_bias = maybe_duplicate(self.pre_attn_ln_bias)
        self.attn_q_weight = maybe_shard_along(self.attn_q_weight, dim=1)
        self.attn_q_scales = maybe_shard_along(self.attn_q_scales, dim=0)
        self.attn_q_bias = maybe_shard_along(self.attn_q_bias, dim=0)
        self.attn_k_weight = maybe_shard_along(self.attn_k_weight, dim=1)
        self.attn_k_scales = maybe_shard_along(self.attn_k_scales, dim=0)
        self.attn_k_bias = maybe_shard_along(self.attn_k_bias, dim=0)
        self.attn_v_weight = maybe_shard_along(self.attn_v_weight, dim=1)
        self.attn_v_scales = maybe_shard_along(self.attn_v_scales, dim=0)
        self.attn_v_bias = maybe_shard_along(self.attn_v_bias, dim=0)
        self.attn_out_weight = maybe_shard_along(self.attn_out_weight, dim=self.attn_out_sharding)
        self.attn_out_scales = maybe_duplicate(self.attn_out_scales)
        self.attn_out_bias = maybe_primary_only(self.attn_out_bias)
        self.post_attn_ln_weight = maybe_duplicate(self.post_attn_ln_weight)
        self.post_attn_ln_bias = maybe_duplicate(self.post_attn_ln_bias)
        self.pre_mlp_ln_weight = maybe_duplicate(self.pre_mlp_ln_weight)
        self.pre_mlp_ln_bias = maybe_duplicate(self.pre_mlp_ln_bias)
        self.mlp_in_weight = maybe_shard_along(self.mlp_in_weight, dim=1)
        self.mlp_in_scales = maybe_shard_along(self.mlp_in_scales, dim=0)
        self.mlp_in_bias = maybe_shard_along(self.mlp_in_bias, dim=0)
        self.mlp_out_weight = maybe_shard_along(self.mlp_out_weight, dim=0)
        self.mlp_out_scales = maybe_duplicate(self.mlp_out_scales)
        self.mlp_out_bias = maybe_primary_only(self.mlp_out_bias)
        self.post_mlp_ln_weight = maybe_duplicate(self.post_mlp_ln_weight)
        self.post_mlp_ln_bias = maybe_duplicate(self.post_mlp_ln_bias)

        extras = []
        for param, dim, allow_pad, allow_quantize in self.extra_parameters:
            if allow_pad:
                size = utils.round_up_to_divisor(param.shape[dim], self.tp_degree)
                param = utils.pad(param, dim, size)

            if allow_quantize and self.neuron_config and self.neuron_config.quant:
                param, scales = quantize.maybe_quantize_weights(param, self.neuron_config.quant)
                scales_dim = 0 if dim == 1 else None
                extras.extend([maybe_manipulator.duplicate_or_shard_along(param, dim),
                               maybe_manipulator.duplicate_or_shard_along(scales, scales_dim)])
            elif allow_quantize:
                # If the parameter is quantizable but the quantization is not enabled, we still need
                # to add a scale placeholder to match the layer arguments
                extras.extend([maybe_manipulator.duplicate_or_shard_along(param, dim), None])
            else:
                extras.append(maybe_manipulator.duplicate_or_shard_along(param, dim))
        self.extra_parameters = extras

        self.init_caches()

    def init_caches(self):

        hidden_size, _ = self.attn_q_weight.shape

        # When padding, compute the hidden size based on the padding. We must
        # allow the KV cache to be padded so it can be evenly divisible across
        # NeuronCores.
        if self.allow_pad:
            hidden_size, _ = self.attn_q_weight.shape
            n_heads = hidden_size // self.attention_head_size
            n_heads_padded = utils.round_up_to_divisor(n_heads, self.tp_degree)
            hidden_size = n_heads_padded * self.attention_head_size

        n_heads = hidden_size // self.attention_head_size
        n_heads_kv_cache = n_heads * self.attn_k_weight.shape[-1] // self.attn_q_weight.shape[-1]
        cache_shape = [self.n_positions, self.batch_size, n_heads_kv_cache, self.attention_head_size]
        cpu_cache = torch.zeros(cache_shape, dtype=self.cache_dtype)
        self.cache_shape = [self.n_positions, self.batch_size, n_heads_kv_cache//self.tp_degree, self.attention_head_size]
        manipulator = parallel.ParallelTensorManipulator(self.tp_degree)
        self.attn_k_cache = manipulator.shard_along(cpu_cache, dim=2)
        self.attn_v_cache = manipulator.shard_along(cpu_cache, dim=2)

    def all_parameters(self):
        return [
            self.pre_attn_ln_weight,
            self.pre_attn_ln_bias,
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
        zero_cache = torch.zeros(self.attn_k_cache.shape, dtype=self.attn_k_cache.dtype)
        zero_cache = [zero_cache for _ in range(self.tp_degree)]
        ops.parallel_write(self.attn_k_cache, zero_cache)
        ops.parallel_write(self.attn_v_cache, zero_cache)

    def assign_parameters(self, layer):
        self.pre_attn_ln_weight = layer.pre_attn_ln_weight
        self.pre_attn_ln_bias = layer.pre_attn_ln_bias
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

    def assign_caches(self, layer):
        self.attn_k_cache = layer.attn_k_cache
        self.attn_v_cache = layer.attn_v_cache
        self.cache_shape = layer.cache_shape


class MaybeParallelTensorManipulator:

    def __init__(self, tp_degree):
        self.manipulator = parallel.ParallelTensorManipulator(tp_degree)

    def duplicate(self, tensor):
        if tensor is None:
            return None
        return self.manipulator.duplicate(tensor)

    def shard_along(self, tensor, dim):
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


class DecoderParameterBuilder:

    def __init__(self, scribe, parameter_number):
        self.scribe = scribe
        self.parameter_number = parameter_number
        self.dtype_converter = compiler.DataTypeConverter()

    def from_tensor(self, tensor, dim_size=None):
        if tensor is None:
            return None
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

    def __init__(self, layers, hlo_modules, num_inputs, tp_degree, prefixed_length=0):
        self.layers = layers
        first_hlo, *_ = hlo_modules
        self.prefixed_length = prefixed_length
        self.input_buffers = [compiler.gen_zero_input(first_hlo, idx) for idx in range(num_inputs)]
        self.kernels = [compiler.ParallelKernel(hm, tp_degree) for hm in hlo_modules]
        self.n_positions_list = [read_n_position(hm, num_inputs) for hm in hlo_modules]
        self.n_active_tokens = read_n_active_tokens(first_hlo)
        self.manipulator = parallel.ParallelTensorManipulator(tp_degree)
        self.tp_degree = tp_degree
        self.need_reorder_cache = False

    def setup(self, layers, ln_lm_head_params):
        self.input_buffers = [self.manipulator.duplicate(buf) for buf in self.input_buffers]
        self.logits_buffer = self.manipulator.duplicate(self.logits_buffer)

        # Compile modules in parallel
        with ProcessPoolExecutor(max_workers=len(self.n_positions_list)) as executor:
            neff_bytes_futures = []
            for kernel, bucket_size in zip(self.kernels, self.n_positions_list):
                future = executor.submit(kernel.compile, bucket_size)
                neff_bytes_futures.append(future)
            for kernel, future in zip(self.kernels, neff_bytes_futures):
                kernel.neff_bytes = future.result()

        for kernel in self.kernels:
            kernel.load()

    def setup_reorder_cache(self):
        self.need_reorder_cache = True
        self.reorder_cache_hlo_kernel = self._create_reoder_cache_kernel()
        self._setup_reorder_cache_kernel()


    def get_neff_bytes(self):
        neff_bytes_arr = [kernel.neff_bytes for kernel in self.kernels]
        if self.need_reorder_cache:
            neff_bytes_arr += [self.reorder_cache_hlo_kernel.kernel.neff_bytes]
        return neff_bytes_arr

    def set_neff_bytes(self, kernels_neff_bytes):
        if self.need_reorder_cache:
            kernels_neff_bytes, reorder_cache_kernel_bytes = kernels_neff_bytes[:-1], kernels_neff_bytes[-1]
            self.reorder_cache_hlo_kernel.kernel.neff_bytes = reorder_cache_kernel_bytes

        for kernel, neff_bytes in zip(self.kernels, kernels_neff_bytes):
            kernel.neff_bytes = neff_bytes


    def find_bucket_id(self, length):
        return next(idx for idx, npos in enumerate(self.n_positions_list) if npos >= length)

    def inputs_host_to_device(self, input_tensors):
        for buf, tensor in zip(self.input_buffers, input_tensors):
            assert buf.shape == tensor.shape, f"Copying tensor from host to device: buffer ({buf.shape}) and tensor ({tensor.shape}) have different shapes!"
            tensor = tensor.to(buf.dtype)
            tensor = self.manipulator.duplicate_on_cpu(tensor)
            ops.parallel_write(buf, tensor)

    def run(self, bucket_id):
        raise NotImplementedError(DecoderProgram)

    def logits_device_to_host(self):
        return self.manipulator.unshard_along(self.logits_buffer, dim=0)

    def _fill_io_tensors(self, input_tensors, output_tensors, layers, npos):
        end = npos
        if self.prefixed_length > 0:
            end = npos + self.prefixed_length
        for layer in layers:
            for cache in layer.attn_k_cache, layer.attn_v_cache:
                cache_slice = self.manipulator.slice_on_nc(cache, 0, start=0, end=end, step=1)
                input_tensors.append(cache_slice)
                output_tensors.append(cache_slice)
        for layer in layers:
            input_tensors.extend(layer.valid_parameters())

    def _create_reoder_cache_kernel(self):
        # assume each layer have same size of cache
        def _reorder_cache(scribe):
            reorder_ids = scribe.s64[self.layers[0].batch_size].Parameter(parameter_number=0)
            caches = []
            param_builder = DecoderParameterBuilder(scribe, 1)
            for layer in self.layers:
                for cache in layer.attn_k_cache, layer.attn_v_cache:
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

    def _setup_reorder_cache_kernel(self):
        self.reorder_cache_hlo_kernel.build()
        self.reorder_cache_hlo_kernel.load()

        # setup memory buffer
        reorder_ids = torch.zeros(self.layers[0].cache_shape, dtype=torch.int64)
        self.reorder_ids_buffers = self.reorder_cache_hlo_kernel.manipulator.duplicate(reorder_ids)
        input_tensors = [self.reorder_ids_buffers]
        output_tensors = []
        for layer in self.layers:
            for cache in layer.attn_k_cache, layer.attn_v_cache:
                input_tensors.append(cache)
                output_tensors.append(cache) # aliasing
        self.reorder_cache_hlo_kernel.setup(input_tensors, output_tensors)

    def reorder_cache(self, reorder_ids):
        assert self.need_reorder_cache, "DecoderProgram is not built with reorder_cache"
        reorder_ids_tensor = torch.tensor(reorder_ids, dtype=torch.int64)
        # TODO: if reorder_ids == range(batch_size), don't do anything
        reorder_ids_tensors_cpu = self.reorder_cache_hlo_kernel.manipulator.duplicate_on_cpu(reorder_ids_tensor)
        ops.parallel_write(self.reorder_ids_buffers, reorder_ids_tensors_cpu)
        self.reorder_cache_hlo_kernel.run()


class DecoderProgramFullyUnrolled(DecoderProgram):

    def __init__(self, layers, hlo_modules, num_inputs, tp_degree, prefixed_length=0):
        super().__init__(layers, hlo_modules, num_inputs, tp_degree, prefixed_length)
        first_hlo, *_ = hlo_modules
        self.logits_buffer = compiler.gen_zero_output(first_hlo, 0)
        self.memories = [kernel.build_memory() for kernel in self.kernels]
        self.executors = list()

    def setup(self, layers, ln_lm_head_params):
        super().setup(layers, ln_lm_head_params)
        for npos, memory in zip(self.n_positions_list, self.memories):
            input_tensors = [*self.input_buffers]
            output_tensors = [self.logits_buffer]
            self._fill_io_tensors(input_tensors, output_tensors, layers, npos)
            input_tensors.extend(ln_lm_head_params)
            memory.setup(input_tensors, output_tensors)

    def run(self, bucket_id):
        self.kernels[bucket_id](self.memories[bucket_id])

    def enable_executor(self):
        for kernel, memory in zip(self.kernels, self.memories):
            input_tensors = [*self.input_buffers]
            output_tensors = [self.logits_buffer]
            executor = kernel.build_executor(memory, input_tensors, output_tensors)
            self.executors.append(executor)

    def execute(self, bucket_id, *inputs, return_ranks=-1):
        """
        Execute a kernel with using the optimized ParallelExecutor.

        This is an alternative to the `run` method which requires that an
        executor has been constructed for each of the underlying kernels.

        Arguments:
            bucket_id: The kernel bucket to execute
            inputs: The set of CPU tensors to copy to each model
            return_ranks: The number of ranks to copy back to CPU
        """
        return self.executors[bucket_id](inputs, return_ranks)



class DecoderProgramMultiLayer(DecoderProgram):

    def __init__(self, layers, hlo_modules, ln_lm_head_hlo_module, num_inputs, num_layers, unroll, tp_degree, prefixed_length=0):
        super().__init__(layers, hlo_modules, num_inputs, tp_degree, prefixed_length)
        if num_layers % unroll:
            raise ValueError(f'unroll={unroll} does not divide num_layers={num_layers}')
        self.logits_buffer = compiler.gen_zero_output(ln_lm_head_hlo_module)
        self.unroll = unroll
        self.multi_layers_memories = []
        for _ in range(num_layers // unroll):
            memories = [kernel.build_memory() for kernel in self.kernels]
            self.multi_layers_memories.append(memories)
        self.ln_lm_head_kernel = compiler.ParallelKernel(ln_lm_head_hlo_module, tp_degree)
        self.ln_lm_head_memory = self.ln_lm_head_kernel.build_memory()
        self.layer_executors = list()
        self.lm_head_executor = None

    def setup(self, layers, ln_lm_head_params):
        super().setup(layers, ln_lm_head_params)
        hidden_buffer, *_ = self.input_buffers
        multi_layer_starts = range(0, len(layers), self.unroll)
        multi_layers = [layers[start:start+self.unroll] for start in multi_layer_starts]
        for memories, multi_layer in zip(self.multi_layers_memories, multi_layers):
            for npos, memory in zip(self.n_positions_list, memories):
                input_tensors = [*self.input_buffers]
                output_tensors = [hidden_buffer]
                self._fill_io_tensors(input_tensors, output_tensors, multi_layer, npos)
                memory.setup(input_tensors, output_tensors)
        self.ln_lm_head_memory.setup([hidden_buffer, *ln_lm_head_params], [self.logits_buffer])
        self.ln_lm_head_kernel.build()
        self.ln_lm_head_kernel.load()

    def run(self, bucket_id):
        for memories in self.multi_layers_memories:
            self.kernels[bucket_id](memories[bucket_id])
        self.ln_lm_head_kernel(self.ln_lm_head_memory)

    def enable_executor(self):
        for layer_memories in self.multi_layers_memories:
            executors = list()
            self.layer_executors.append(executors)
            for kernel, memories in zip(self.kernels,layer_memories):
                # NOTE: No input/output returns from layers. Do one-time copy
                executor = kernel.build_executor(memories, [], [])
                executors.append(executor)
        output_tensors = [self.logits_buffer]
        executor = self.ln_lm_head_kernel.build_executor(self.ln_lm_head_memory, [], output_tensors)
        self.lm_head_executor = executor

    def execute(self, bucket_id, *inputs, return_ranks=-1):
        self.inputs_host_to_device(inputs) # One-time input copy
        for layer_executor in self.layer_executors:
            layer_executor[bucket_id]([], return_ranks=0) # No returns from intermediate layers
        return self.lm_head_executor([], return_ranks=return_ranks)


class FastCacheBroadcaster:

    def __init__(self, n_positions, from_batch_size, to_batch_size, n_heads_tp, d_head, amp,
                 tp_degree, n_layer):
        cache_broadcast_impl = hlo.cache_broadcast(n_positions, from_batch_size, to_batch_size,
                                                   n_heads_tp, d_head, amp, n_layer)
        cache_broadcast_hlo_module = compiler.compile_py_func(cache_broadcast_impl)
        self.cache_broadcast_kernel = compiler.ParallelKernel(cache_broadcast_hlo_module, tp_degree)
        self.cache_broadcast_memory = self.cache_broadcast_kernel.build_memory()
        self.cache_broadcast_kernel.build()
        self.cache_broadcast_kernel.load()

    def setup(self, source_caches, target_caches):
        self.cache_broadcast_memory.setup(source_caches, target_caches)

    def run_broadcast(self):
        self.cache_broadcast_kernel(self.cache_broadcast_memory)
