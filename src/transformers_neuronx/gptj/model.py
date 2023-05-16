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
from transformers_neuronx import compiler
from transformers_neuronx import dtypes
from transformers_neuronx import module
from transformers_neuronx import ops
from transformers_neuronx import parallel
from transformers_neuronx import program
from transformers_neuronx import utils
from transformers_neuronx.gptj import hlo
from transformers_neuronx.gptj.config import GPTJConfig
from transformers_neuronx.sampling import simple_sample


class GPTJForSampling(module.PretrainedModel):

    def __init__(self, config, batch_size=1, amp='f32', tp_degree=2,
                 unroll=None, init_n_active_tokens=None, **kwargs):
        super().__init__()
        config = GPTJConfig(config, batch_size, amp, tp_degree, **kwargs)
        self.config = config
        # Check if input sequence length is allowed given position embedding dimensions
        sequence_length = kwargs.get("n_positions", None)
        if sequence_length:
            max_allowed_sequence_length = config.n_ctx
            if sequence_length > max_allowed_sequence_length:
                raise ValueError(f"Sequence length ({sequence_length}) cannot be larger than position embedding's context size ({max_allowed_sequence_length})!")
        if unroll is None:
            unroll = config.n_layer
        self.unroll = unroll
        if init_n_active_tokens is not None:
            raise NotImplementedError(f'init_n_active_tokens={init_n_active_tokens}')
        self.init_n_active_tokens = init_n_active_tokens
        dtype = dtypes.to_torch_dtype(config.amp)
        self.transformer = GPTJTransformer(config)
        self.lm_head = module.LowMemoryLazyLinear(config.vocab_size, dtype=dtype)
        self.ln_lm_head = GPTJLnLmHead(config, self.transformer.ln_f, self.lm_head)
        self.n_positions_list = utils.power_of_two_bucket_sizes(128, config.n_positions)
        self.program = None
        self.init_program = program.DoNothingDecoder()
        self.manipulator = parallel.ParallelTensorManipulator(config.tp_degree)

    def to_neuron(self):
        ops.init()
        config = self.config
        n_positions_list = self.n_positions_list
        unroll = self.unroll
        self.program = build_gptj_program(config, 1, n_positions_list, unroll)
        init_n_active = self.init_n_active_tokens
        if init_n_active is not None:
            self.init_program = build_gptj_program(config, init_n_active, n_positions_list, unroll)
        self.transformer.wte.materialize()
        for idx, block in enumerate(self.transformer.h):
            block.to_neuron(n_positions_list)
        self.ln_lm_head.to_neuron()
        self.program.setup(self.transformer.h, self.ln_lm_head)
        self.init_program.setup(self.transformer.h, self.ln_lm_head)

    def reset(self):
        for block in self.transformer.h:
            block.reset()

    def forward(self, input_ids, cache_offset, start_ids=None):
        last_offset = cache_offset[-1].item()
        bucket_id = find_first_ge_index(self.n_positions_list, last_offset)
        this_length = input_ids.shape[-1]
        init_length = self.init_program.init_length(this_length, self.init_n_active_tokens)
        init_step = self.init_program.init_step(self.init_n_active_tokens)
        for cur_len in range(0, init_length, init_step):
            slicing = slice(cur_len, cur_len + init_step)
            inputs = input_ids[:, slicing], cache_offset[slicing]
            logits = self._run_program(self.init_program, bucket_id, *inputs)
        for cur_len in range(init_length, this_length):
            slicing = slice(cur_len, cur_len + 1)
            inputs = input_ids[:, slicing], cache_offset[slicing]
            logits = self._run_program(self.program, bucket_id, *inputs)
        logits = self.manipulator.unshard_along(logits, dim=0)
        logits = logits.to(torch.float32)
        logits = logits[:self.config.vocab_size]
        logits = logits.transpose(0, -1)
        logits = logits[:, -1, :]
        return logits

    def _run_program(self, program, bucket_id, input_ids, cache_offset):
        active_n_positions = self.n_positions_list[bucket_id]
        hidden = self.transformer.wte(input_ids)
        hidden = hidden.transpose(0, -1).contiguous()
        input_buffers = program.buffers.get_input_buffers(bucket_id)
        hidden_buffer, pos_embd_buffer, *_ = input_buffers
        hidden = hidden.to(hidden_buffer.dtype)
        config = self.config
        rotary_dim = config.rotary_dim
        s_head = config.n_embd // config.n_head
        last_offset = cache_offset[-1].item()
        pos_embd = self._fixed_pos_embedding(rotary_dim, s_head, offset=last_offset)
        pos_embd = pos_embd.unsqueeze(0)
        pos_embd = pos_embd.to(pos_embd_buffer.dtype)
        hidden = self.manipulator.duplicate_on_cpu(hidden)
        pos_embd = self.manipulator.duplicate_on_cpu(pos_embd)
        cache_offset = self.manipulator.duplicate_on_cpu(cache_offset)
        for in_buffer, in_tensor in zip(input_buffers, [hidden, pos_embd, cache_offset]):
            ops.parallel_write(in_buffer, in_tensor)
        return program.run(bucket_id)

    def sample(self, input_ids, sequence_length, start_ids=None, top_k=50):
        return simple_sample(self, input_ids, start_ids, sequence_length,
                             eos_token_id=self.config.eos_token_id, top_k=top_k)

    def _fixed_pos_embedding(self, dim, head_dim, offset):
        # TODO: init_n_active > 1
        seq_len = offset + 1
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2) / dim))
        sinusoid_inp = torch.einsum("i , j -> i j", torch.arange(seq_len), inv_freq).float()
        sin = torch.sin(sinusoid_inp)
        cos = torch.cos(sinusoid_inp)
        sin = sin[None, offset:seq_len, None, :].repeat_interleave(2, 3)
        sin[..., ::2] *= -1
        cos = cos[None, offset:seq_len, None, :].repeat_interleave(2, 3)
        sin_diag = torch.diagflat(sin)
        rotate = torch.eye(sin.shape[-1])
        rotate[::2, :], rotate[1::2, :] = rotate[1::2, :].clone(), rotate[::2, :].clone()
        cos_diag = torch.diagflat(cos)
        sincos = torch.matmul(rotate, sin_diag) + cos_diag
        # 256x256x256 matmul where only the beginning 64x64x64 is useful; can optimize if necessary
        pos_embd = torch.eye(head_dim)
        pos_embd[:dim, :dim] = sincos
        return pos_embd


class GPTJBuffers:

    def __init__(self, hlo_modules, tp_degree):
        first_hlo_module, *_ = hlo_modules
        hidden_buffer = compiler.gen_zero_input(first_hlo_module, 0)
        pos_embd_buffer = compiler.gen_zero_input(first_hlo_module, 1)
        cache_offset_buffer = compiler.gen_zero_input(first_hlo_module, 2)
        self.input_buffers = [hidden_buffer, pos_embd_buffer, cache_offset_buffer]
        self.output_buffer = compiler.gen_zero_output(first_hlo_module, 0)
        self.manipulator = parallel.ParallelTensorManipulator(tp_degree)

    def to_neuron(self):
        hidden_buffer, pos_embd_buffer, cache_offset_buffer = self.input_buffers
        hidden_buffer = self.manipulator.duplicate(hidden_buffer)
        pos_embd_buffer = self.manipulator.duplicate(pos_embd_buffer)
        cache_offset_buffer = self.manipulator.duplicate(cache_offset_buffer)
        self.input_buffers = [hidden_buffer, pos_embd_buffer, cache_offset_buffer]
        self.output_buffer = self.manipulator.duplicate(self.output_buffer)

    def get_input_buffers(self, bucket_id):
        hidden_buffer, pos_embd_buffer, cache_offset_buffer = self.input_buffers
        return [hidden_buffer, pos_embd_buffer, cache_offset_buffer]


def find_first_ge_index(values, target):
    return next(idx for idx, val in enumerate(values) if val >= target)


class GPTJTransformer(module.LowMemoryModule):

    def __init__(self, config):
        super().__init__()
        self.wte = module.LowMemoryEmbedding(config.vocab_size, config.n_embd)
        self.h = module.LowMemoryModuleList()
        for _ in range(config.n_layer):
            block = GPTJBlock(config)
            self.h.append(block)
        self.ln_f = module.LowMemoryLayerNorm(config.n_embd)


class GPTJBlock(module.LowMemoryModule):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = module.LowMemoryLayerNorm(config.n_embd)
        self.attn = GPTJAttention(config)
        self.mlp = GPTJMLP(config)
        self.config = config
        self.ln_1_weight = None
        self.ln_1_bias = None
        self.attn_q_weight = None
        self.attn_k_weight = None
        self.attn_v_weight = None
        self.attn_out_weight = None
        self.mlp_in_weight = None
        self.mlp_in_bias = None
        self.mlp_out_weight = None
        self.mlp_out_bias = None
        self.key_cache = None
        self.key_cache_slices = None
        self.value_cache = None
        self.value_cache_slices = None

    def to_neuron(self, n_positions_list):
        self.materialize()
        config = self.config
        manipulator = parallel.ParallelTensorManipulator(config.tp_degree)
        duplicate = manipulator.duplicate
        shard_along = manipulator.shard_along
        primary_only = manipulator.primary_only
        self.ln_1_weight = duplicate(self.ln_1.weight.detach())
        self.ln_1_bias = duplicate(self.ln_1.bias.detach())
        attn = self.attn
        self.attn_q_weight = shard_along(attn.q_proj.weight.detach().T, dim=1)
        self.attn_k_weight = shard_along(attn.k_proj.weight.detach().T, dim=1)
        self.attn_v_weight = shard_along(attn.v_proj.weight.detach().T, dim=1)
        self.attn_out_weight = shard_along(attn.out_proj.weight.detach().T, dim=0)
        mlp = self.mlp
        self.mlp_in_weight = shard_along(mlp.fc_in.weight.detach().T, dim=1)
        self.mlp_in_bias = shard_along(mlp.fc_in.bias.detach(), dim=0)
        self.mlp_out_weight = shard_along(mlp.fc_out.weight.detach().T, dim=0)
        self.mlp_out_bias = primary_only(mlp.fc_out.bias.detach())
        self.nullify()

        slice_on_nc = manipulator.slice_on_nc
        n_positions = config.n_positions
        batch_size = config.batch_size
        n_head = config.n_head
        s_head = config.n_embd // n_head
        cache_shape = [n_positions, batch_size, n_head, s_head]
        dtype = dtypes.to_torch_dtype(config.amp)
        key_cache = torch.zeros(cache_shape, dtype=dtype)
        key_cache = shard_along(key_cache, dim=2)
        self.key_cache = key_cache
        self.key_cache_slices = [slice_on_nc(key_cache, 0, start=0, end=npos, step=1)
                                 for npos in n_positions_list]
        value_cache = torch.zeros(cache_shape, dtype=dtype)
        value_cache = shard_along(value_cache, dim=2)
        self.value_cache = value_cache
        self.value_cache_slices = [slice_on_nc(value_cache, 0, start=0, end=npos, step=1)
                                   for npos in n_positions_list]

    def get_parameters(self):
        return [
            self.ln_1_weight,
            self.ln_1_bias,
            self.attn_q_weight,
            self.attn_k_weight,
            self.attn_v_weight,
            self.attn_out_weight,
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


class GPTJAttention(module.LowMemoryModule):

    def __init__(self, config):
        super().__init__()
        n_embd = config.n_embd
        dtype = dtypes.to_torch_dtype(config.amp)
        self.q_proj = module.LowMemoryLazyLinear(n_embd, bias=False, dtype=dtype)
        self.k_proj = module.LowMemoryLazyLinear(n_embd, bias=False, dtype=dtype)
        self.v_proj = module.LowMemoryLazyLinear(n_embd, bias=False, dtype=dtype)
        self.out_proj = module.LowMemoryLazyLinear(n_embd, bias=False, dtype=dtype)


class GPTJMLP(module.LowMemoryModule):

    def __init__(self, config):
        super().__init__()
        dtype = dtypes.to_torch_dtype(config.amp)
        self.fc_in = module.LowMemoryLazyLinear(config.intermediate_dim, dtype=dtype)
        self.fc_out = module.LowMemoryLazyLinear(config.n_embd, dtype=dtype)


class GPTJLnLmHead:

    def __init__(self, config, ln_f, lm_head):
        self.tp_degree = config.tp_degree
        self.ln_f = ln_f
        self.lm_head = lm_head
        self.ln_f_weight = None
        self.ln_f_bias = None
        self.lm_head_weight = None
        self.lm_head_bias = None
        self.manipulator = parallel.ParallelTensorManipulator(config.tp_degree)

    def to_neuron(self):
        duplicate = self.manipulator.duplicate
        shard_along = self.manipulator.shard_along
        self.ln_f.materialize()
        self.ln_f_weight = duplicate(self.ln_f.weight.detach())
        self.ln_f_bias = duplicate(self.ln_f.bias.detach())
        self.lm_head.materialize()
        # Pad the lm_head_weight and lm_head_bias if vocab_size % tp_degree != 0
        lm_head_weight = self.lm_head.weight.detach().T
        _, vocab_size = lm_head_weight.shape
        vocab_pad = utils.pad_vocab_size(vocab_size, self.tp_degree)
        lm_head_weight = torch.nn.functional.pad(lm_head_weight, (0, vocab_pad, 0, 0))
        self.lm_head_weight = shard_along(lm_head_weight, dim=1)
        lm_head_bias = self.lm_head.bias.detach()
        lm_head_bias = torch.nn.functional.pad(lm_head_bias, (0, vocab_pad))
        self.lm_head_bias = shard_along(lm_head_bias, dim=0)
        self.lm_head.nullify()

    def get_parameters(self):
        return [self.ln_f_weight, self.ln_f_bias, self.lm_head_weight, self.lm_head_bias]


def build_gptj_program(config, n_active, n_positions_list, n_blocks):
    hlo_modules = [hlo.build_gptj_hlo_module(config, n_active, npos) for npos in n_positions_list]
    buffers = GPTJBuffers(hlo_modules, config.tp_degree)
    if n_blocks == config.n_layer:
        return program.FullyUnrolledDecoder(config.tp_degree, hlo_modules, buffers)
    else:
        build_func = hlo.build_gptj_multi_block_hlo_module
        hlo_modules = [build_func(config, n_active, npos, n_blocks) for npos in n_positions_list]
        head_hlo_module = hlo.build_ln_lm_head_hlo_module(config, n_active)
        return program.MultiLayerDecoder(config.n_layer, config.tp_degree, hlo_modules,
                                         head_hlo_module, n_blocks, buffers)
