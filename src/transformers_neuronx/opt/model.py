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
from torch.nn.parameter import UninitializedParameter
from transformers_neuronx import compiler
from transformers_neuronx import dtypes
from transformers_neuronx import module
from transformers_neuronx import ops
from transformers_neuronx import parallel
from transformers_neuronx import utils
from transformers_neuronx.opt import hlo
from transformers_neuronx.opt.config import OPTConfig


class OPTForSampling(module.PretrainedModel):

    def __init__(self, config, batch_size=1, amp='f32', tp_degree=2, n_positions=2048,
                 unroll=None, init_n_active_tokens=None, **kwargs):
        super().__init__()
        config = OPTConfig(config, n_positions, batch_size, amp, tp_degree, **kwargs)
        self.config = config
        if unroll is None:
            unroll = config.num_hidden_layers
        self.init_n_active_tokens = init_n_active_tokens
        dtype = dtypes.to_torch_dtype(config.amp)
        self.model = OPTModel(config)
        self.lm_head = module.LowMemoryLazyLinear(config.vocab_size, dtype=dtype, bias=False)
        self.ln_lm_head = OPTLnLmHead(config, self.model.decoder.final_layer_norm, self.lm_head)
        n_positions_list = utils.power_of_two_bucket_sizes(128, n_positions)
        self.n_positions_list = n_positions_list
        self.program = build_opt_program(config, 1, n_positions_list, unroll)
        self.init_program = OPTProgramDoNothing()
        if init_n_active_tokens is not None:
            self.init_program = build_opt_program(config, init_n_active_tokens, n_positions_list, unroll)
        self.manipulator = parallel.ParallelTensorManipulator(tp_degree)
        self.to_neuron_hooks = []

    def to_neuron(self):
        ops.init()
        decoder = self.model.decoder
        decoder.embed_tokens.materialize()
        decoder.embed_positions.materialize()
        parallel.layers_to_neuron(16, decoder.layers, self.n_positions_list, self.to_neuron_hooks)
        self.ln_lm_head.to_neuron()
        self.program.setup(self)
        self.init_program.setup(self)

    def reset(self):
        for layer in self.model.decoder.layers:
            layer.reset()

    def forward(self, input_ids, cache_offset, mask):
        last_offset = cache_offset[-1].item()
        bucket_id = find_first_ge_index(self.n_positions_list, last_offset)
        this_length = input_ids.shape[-1]
        init_length = self.init_program.init_length(this_length, self.init_n_active_tokens)
        init_step = self.init_program.init_step(self.init_n_active_tokens)
        for cur_len in range(0, init_length, init_step):
            slicing = slice(cur_len, cur_len + init_step)
            inputs = input_ids[:, slicing], cache_offset[slicing], mask[slicing]
            logits = self._run_program(self.init_program, bucket_id, *inputs)
        for cur_len in range(init_length, this_length):
            slicing = slice(cur_len, cur_len + 1)
            inputs = input_ids[:, slicing], cache_offset[slicing], mask[slicing]
            logits = self._run_program(self.program, bucket_id, *inputs)
        logits = self.manipulator.unshard_along(logits, dim=0)
        logits = logits.to(torch.float32)
        logits = logits[:self.config.vocab_size]
        logits = logits.transpose(0, -1)
        logits = logits[:, -1, :]
        return logits

    def _run_program(self, program, bucket_id, input_ids, cache_offset, mask):
        active_n_positions = self.n_positions_list[bucket_id]
        mask = mask[:, :active_n_positions].contiguous()
        inputs_embeds = self.model.decoder.embed_tokens(input_ids)
        past_length = cache_offset[0].item()
        this_length = input_ids.shape[-1]
        position_ids = torch.arange(past_length, past_length + this_length)
        position_ids = position_ids.unsqueeze(0).view(-1, this_length)
        position_embeds = self.model.decoder.embed_positions(position_ids)
        hidden = inputs_embeds + position_embeds
        hidden = hidden.transpose(0, -1).contiguous()
        hidden = hidden.to(self.program.buffers.hidden_buffer.dtype)
        hidden = self.manipulator.duplicate_on_cpu(hidden)
        cache_offset = self.manipulator.duplicate_on_cpu(cache_offset)
        mask = self.manipulator.duplicate_on_cpu(mask)
        ops.parallel_write(program.buffers.hidden_buffer, hidden)
        ops.parallel_write(program.buffers.cache_offset_buffer, cache_offset)
        ops.parallel_write(program.buffers.mask_buffers[bucket_id], mask)
        return program.run(bucket_id)

    @torch.no_grad()
    def sample(self, input_ids, sequence_length):
        config = self.config
        filter_value = -float('inf')
        min_length = max_length = sequence_length
        top_k = 50
        self.reset()
        start = input_ids.shape[1]

        # populate key/value caches according to the prompt text
        cache_offset = torch.arange(start, dtype=torch.int32)
        mask = torch.ones([start, config.n_positions])
        mask = torch.tril(mask)
        next_token_scores = self(input_ids, cache_offset, mask)

        # auto-regressive generation
        tokens = [input_ids]
        for cur_len in range(start, max_length):
            next_len = cur_len + 1

            # pre-process distribution
            if cur_len < min_length:
                next_token_scores[:, config.eos_token_id] = -float('inf')

            # Remove all tokens with a probability less than the last token of the top-k
            topk_values, _ = torch.topk(next_token_scores, top_k)
            indices_to_remove = next_token_scores < topk_values[:, -1, None]
            next_token_scores = next_token_scores.masked_fill(indices_to_remove, filter_value)

            # sample
            probs = torch.nn.functional.softmax(next_token_scores, dim=-1)
            inputs = torch.multinomial(probs, num_samples=1)
            tokens.append(inputs)

            # stop when eos_token is found
            if (inputs == config.eos_token_id).all():
                break

            # forward pass to get next token
            cache_offset = torch.as_tensor([cur_len], dtype=torch.int32)
            mask = torch.zeros([1, config.n_positions])
            mask[:, :next_len] = 1.0
            next_token_scores = self(inputs, cache_offset, mask)
        return torch.cat(tokens, dim=-1)

    def register_to_neuron_hook(self, hook):
        self.to_neuron_hooks.append(hook)


class OPTBuffers:

    def __init__(self, opt_hlo_modules):
        first_hlo_module, *_ = opt_hlo_modules
        self.hidden_buffer = compiler.gen_zero_input(first_hlo_module, 0)
        self.cache_offset_buffer = compiler.gen_zero_input(first_hlo_module, 1)
        self.mask_buffers = [compiler.gen_zero_input(hlo, 2) for hlo in opt_hlo_modules]
        self.logits_buffer = compiler.gen_zero_output(first_hlo_module, 0)

    def to_neuron(self, manipulator):
        duplicate = manipulator.duplicate
        self.hidden_buffer = duplicate(self.hidden_buffer)
        self.cache_offset_buffer = duplicate(self.cache_offset_buffer)
        self.mask_buffers = [duplicate(mask_buffer) for mask_buffer in self.mask_buffers]
        self.logits_buffer = duplicate(self.logits_buffer)


def find_first_ge_index(values, target):
    return next(idx for idx, val in enumerate(values) if val >= target)


class OPTModel(module.LowMemoryModule):

    def __init__(self, config):
        super().__init__()
        self.decoder = OPTDecoder(config)


class OPTDecoder(module.LowMemoryModule):

    def __init__(self, config):
        super().__init__()
        self.embed_tokens = module.LowMemoryEmbedding(config.vocab_size, config.hidden_size,
                                                      padding_idx=config.pad_token_id)
        self.embed_positions = OPTLearnedPositionalEmbedding(config.max_position_embeddings,
                                                             config.hidden_size)
        self.layers = module.LowMemoryModuleList()
        for _ in range(config.num_hidden_layers):
            self.layers.append(OPTDecoderLayer(config))
        self.final_layer_norm = module.LowMemoryLayerNorm(config.hidden_size)


class OPTLearnedPositionalEmbedding(module.LowMemoryEmbedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, num_embeddings, embedding_dim):
        # OPT is set up so that if padding_idx is specified then offset the embedding ids by 2
        # and adjust num_embeddings appropriately. Other models don't have this hack
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def forward(self, position_ids):
        return super().forward(position_ids + self.offset)


class OPTDecoderLayer(module.LowMemoryModule):

    def __init__(self, config):
        super().__init__()
        self.self_attn_layer_norm = module.LowMemoryLayerNorm(config.hidden_size)
        self.self_attn = OPTAttention(config)
        dtype = dtypes.to_torch_dtype(config.amp)
        self.fc1 = module.LowMemoryLazyLinear(config.hidden_size, dtype=dtype)
        self.fc2 = module.LowMemoryLazyLinear(config.ffn_dim, dtype=dtype)
        self.final_layer_norm = module.LowMemoryLayerNorm(config.hidden_size)
        self.config = config
        self.ln_1_weight = None
        self.ln_1_bias = None
        self.attn_q_weight = None
        self.attn_k_weight = None
        self.attn_v_weight = None
        self.attn_out_weight = None
        self.ln_2_weight = None
        self.ln_2_bias = None
        self.mlp_in_weight = None
        self.mlp_in_bias = None
        self.mlp_out_weight = None
        self.mlp_out_bias = None
        self.key_cache = None
        self.key_cache_slices = None
        self.value_cache = None
        self.value_cache_slices = None

    def to_neuron(self, n_positions_list):
        config = self.config
        manipulator = parallel.ParallelTensorManipulator(config.tp_degree)
        duplicate = manipulator.duplicate
        shard_along = manipulator.shard_along
        primary_only = manipulator.primary_only
        self.self_attn_layer_norm.materialize()
        self.ln_1_weight = duplicate(self.self_attn_layer_norm.weight.detach())
        self.ln_1_bias = duplicate(self.self_attn_layer_norm.bias.detach())
        q_proj = self.self_attn.q_proj
        k_proj = self.self_attn.k_proj
        v_proj = self.self_attn.v_proj
        out_proj = self.self_attn.out_proj
        q_proj.materialize()
        k_proj.materialize()
        v_proj.materialize()
        out_proj.materialize()
        self.attn_q_weight = shard_along(q_proj.weight.detach().T, dim=1)
        self.attn_q_bias = shard_along(q_proj.bias.detach(), dim=0)
        self.attn_k_weight = shard_along(k_proj.weight.detach().T, dim=1)
        self.attn_k_bias = shard_along(k_proj.bias.detach(), dim=0)
        self.attn_v_weight = shard_along(v_proj.weight.detach().T, dim=1)
        self.attn_v_bias = shard_along(v_proj.bias.detach(), dim=0)
        self.attn_out_weight = shard_along(out_proj.weight.detach().T, dim=0)
        self.attn_out_bias = primary_only(out_proj.bias.detach())
        q_proj.weight = UninitializedParameter()
        q_proj.bias = UninitializedParameter()
        k_proj.weight = UninitializedParameter()
        k_proj.bias = UninitializedParameter()
        v_proj.weight = UninitializedParameter()
        v_proj.bias = UninitializedParameter()
        out_proj.weight = UninitializedParameter()
        out_proj.bias = UninitializedParameter()
        self.final_layer_norm.materialize()
        self.ln_2_weight = duplicate(self.final_layer_norm.weight.detach())
        self.ln_2_bias = duplicate(self.final_layer_norm.bias.detach())
        fc1 = self.fc1
        fc2 = self.fc2
        fc1.materialize()
        fc2.materialize()
        self.mlp_in_weight = shard_along(fc1.weight.detach().T, dim=1)
        self.mlp_in_bias = shard_along(fc1.bias.detach(), dim=0)
        self.mlp_out_weight = shard_along(fc2.weight.detach().T, dim=0)
        self.mlp_out_bias = primary_only(fc2.bias.detach())
        fc1.weight = UninitializedParameter()
        fc2.weight = UninitializedParameter()

        slice_on_nc = manipulator.slice_on_nc
        n_positions = config.n_positions
        batch_size = config.batch_size
        n_head = config.num_attention_heads
        s_head = config.hidden_size // n_head
        cache_shape = [n_positions, batch_size, n_head, s_head]
        dtype = dtypes.to_torch_dtype(config.amp)
        key_cache = torch.zeros(cache_shape, dtype=dtype)
        key_cache = manipulator.shard_along(key_cache, dim=2)
        self.key_cache = key_cache
        self.key_cache_slices = [slice_on_nc(key_cache, 0, start=0, end=npos, step=1)
                                 for npos in n_positions_list]
        value_cache = torch.zeros(cache_shape, dtype=dtype)
        value_cache = manipulator.shard_along(value_cache, dim=2)
        self.value_cache = value_cache
        self.value_cache_slices = [slice_on_nc(value_cache, 0, start=0, end=npos, step=1)
                                   for npos in n_positions_list]

    def get_parameters(self):
        return [
            self.ln_1_weight,
            self.ln_1_bias,
            self.attn_q_weight,
            self.attn_q_bias,
            self.attn_k_weight,
            self.attn_k_bias,
            self.attn_v_weight,
            self.attn_v_bias,
            self.attn_out_weight,
            self.attn_out_bias,
            self.ln_2_weight,
            self.ln_2_bias,
            self.mlp_in_weight,
            self.mlp_in_bias,
            self.mlp_out_weight,
            self.mlp_out_bias,
        ]

    def get_cache_slices(self):
        return [self.key_cache_slices, self.value_cache_slices]

    def reset(self):
        zero_cache = torch.zeros(self.key_cache.shape, dtype=self.key_cache.dtype)
        zero_cache = [zero_cache for _ in range(self.config.tp_degree)]
        ops.parallel_write(self.key_cache, zero_cache)
        ops.parallel_write(self.value_cache, zero_cache)


class OPTAttention(module.LowMemoryModule):

    def __init__(self, config):
        super().__init__()
        hidden_size = config.hidden_size
        dtype = dtypes.to_torch_dtype(config.amp)
        self.q_proj = module.LowMemoryLazyLinear(hidden_size, dtype=dtype)
        self.k_proj = module.LowMemoryLazyLinear(hidden_size, dtype=dtype)
        self.v_proj = module.LowMemoryLazyLinear(hidden_size, dtype=dtype)
        self.out_proj = module.LowMemoryLazyLinear(hidden_size, dtype=dtype)


class OPTLnLmHead:

    def __init__(self, config, ln_f, lm_head):
        self.ln_f = ln_f
        self.lm_head = lm_head
        self.ln_f_weight = None
        self.ln_f_bias = None
        self.lm_head_weight = None
        tp_degree = config.tp_degree
        self.manipulator = parallel.ParallelTensorManipulator(tp_degree)
        vocab_size = config.vocab_size
        self.vocab_pad = pad_vocab_size(vocab_size, tp_degree)

    def to_neuron(self):
        manipulator = self.manipulator
        self.ln_f.materialize()
        self.ln_f_weight = manipulator.duplicate(self.ln_f.weight.detach())
        self.ln_f_bias = manipulator.duplicate(self.ln_f.bias.detach())
        self.lm_head.materialize()
        lm_head_weight = self.lm_head.weight.detach()
        lm_head_weight = torch.nn.functional.pad(lm_head_weight, (0, 0, 0, self.vocab_pad))
        self.lm_head_weight = manipulator.shard_along(lm_head_weight.T, dim=1)
        self.lm_head.weight = UninitializedParameter()

    def get_parameters(self):
        return [self.ln_f_weight, self.ln_f_bias, self.lm_head_weight]


def pad_vocab_size(vocab_size, tp_degree):
    return ((vocab_size // tp_degree + 1) * tp_degree - vocab_size) % tp_degree


def build_opt_program(config, n_active, n_positions_list, n_layers):
    hlo_modules = [hlo.build_opt_hlo_module(config, n_active, npos) for npos in n_positions_list]
    buffers = OPTBuffers(hlo_modules)
    if n_layers == config.num_hidden_layers:
        return OPTProgramFullyUnrolled(config, hlo_modules, buffers)
    else:
        build_func = hlo.build_opt_multi_layer_hlo_module
        hlo_modules = [build_func(config, n_active, npos, n_layers) for npos in n_positions_list]
        return OPTProgramPartiallyUnrolled(config, hlo_modules, n_layers, buffers)


class OPTProgramDoNothing:

    def setup(self, model):
        pass

    def init_length(self, this_length, init_n_active_tokens):
        return 0

    def init_step(self, init_n_active_tokens):
        return 1

    def run(self, bucket_id):
        pass


class OPTProgramBase:

    def setup(self, model):
        raise NotImplementedError(OPTProgramBase)

    def init_length(self, this_length, init_n_active_tokens):
        return this_length // init_n_active_tokens * init_n_active_tokens

    def init_step(self, init_n_active_tokens):
        return init_n_active_tokens

    def run(self, bucket_id):
        raise NotImplementedError(OPTProgramBase)


class OPTProgramPartiallyUnrolled(OPTProgramBase):

    def __init__(self, config, multi_layer_hlo_modules, n_layers, buffers):
        num_hidden_layers = config.num_hidden_layers
        if num_hidden_layers % n_layers:
            raise ValueError(f'n_layers={n_layers} does not divide num_hidden_layers={num_hidden_layers}')
        self.n_layers = n_layers
        tp_degree = config.tp_degree
        self.multi_layer_kernels = [compiler.build_parallel_kernel(hm, tp_degree)
                                    for hm in multi_layer_hlo_modules]
        head_hlo_module = hlo.build_lm_head_hlo_module(config, 1)
        self.head_kernel = compiler.build_parallel_kernel(head_hlo_module, tp_degree)
        self.multi_layers_memories = []
        for _ in range(num_hidden_layers // n_layers):
            memories = [compiler.ParallelMemory(hm, tp_degree) for hm in multi_layer_hlo_modules]
            self.multi_layers_memories.append(memories)
        self.head_memory = compiler.ParallelMemory(head_hlo_module, tp_degree)
        self.buffers = buffers

    def setup(self, model):
        for kernel in self.multi_layer_kernels:
            kernel.load()
        self.head_kernel.load()
        for memories in self.multi_layers_memories:
            for memory in memories:
                memory.init()
        self.head_memory.init()
        buffers = self.buffers
        buffers.to_neuron(model.manipulator)
        multi_layer_starts = range(0, model.config.num_hidden_layers, self.n_layers)
        layers = model.model.decoder.layers
        multi_layers = [layers[start:start+self.n_layers] for start in multi_layer_starts]
        for memories, multi_layer in zip(self.multi_layers_memories, multi_layers):
            cache_slices, params = cache_slices_and_parameters(multi_layer)
            setup_opt_memories(memories, buffers, cache_slices, params, buffers.hidden_buffer)
        head_inputs = [buffers.hidden_buffer, *model.ln_lm_head.get_parameters()]
        for index, input_buffer in enumerate(head_inputs):
            self.head_memory.inputs.add(index, input_buffer)
        self.head_memory.outputs.add(0, buffers.logits_buffer)

    def run(self, bucket_id):
        for memories in self.multi_layers_memories:
            self.multi_layer_kernels[bucket_id](memories[bucket_id])
        self.head_kernel(self.head_memory)
        return self.buffers.logits_buffer


class OPTProgramFullyUnrolled(OPTProgramBase):

    def __init__(self, config, opt_hlo_modules, buffers):
        self.kernels = [compiler.build_parallel_kernel(hm, config.tp_degree) for hm in opt_hlo_modules]
        self.memories = [compiler.ParallelMemory(hm, config.tp_degree) for hm in opt_hlo_modules]
        self.buffers = buffers

    def setup(self, model):
        for kernel in self.kernels:
            kernel.load()
        for memory in self.memories:
            memory.init()
        cache_slices, params = cache_slices_and_parameters(model.model.decoder.layers)
        params.extend(model.ln_lm_head.get_parameters())
        buffers = self.buffers
        buffers.to_neuron(model.manipulator)
        setup_opt_memories(self.memories, buffers, cache_slices, params, buffers.logits_buffer)

    def run(self, bucket_id):
        self.kernels[bucket_id](self.memories[bucket_id])
        return self.buffers.logits_buffer


def cache_slices_and_parameters(layers):
    cache_slices = []
    params = []
    for layer in layers:
        cache_slices.extend(layer.get_cache_slices())
        params.extend(layer.get_parameters())
    return cache_slices, params


def setup_opt_memories(memories, buffers, cache_slices, params, output_buffer):
    for bucket_id, memory in enumerate(memories):
        mask_buffer = buffers.mask_buffers[bucket_id]
        input_buffers = [buffers.hidden_buffer, buffers.cache_offset_buffer, mask_buffer]
        for index, input_buffer in enumerate(input_buffers):
            memory.inputs.add(index, input_buffer)
        cache_start_index = len(input_buffers)
        for index, bucketed_caches in enumerate(cache_slices, start=cache_start_index):
            memory.inputs.add(index, bucketed_caches[bucket_id])
        param_start_index = cache_start_index + len(cache_slices)
        for index, param in enumerate(params, start=param_start_index):
            memory.inputs.add(index, param)
        memory.outputs.add(0, output_buffer)
        for index, bucketed_caches in enumerate(cache_slices, start=1):
            memory.outputs.add(index, bucketed_caches[bucket_id])
