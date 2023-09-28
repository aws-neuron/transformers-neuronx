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
import warnings

import torch
from transformers_neuronx import compiler
from transformers_neuronx import dtypes
from transformers_neuronx import module
from transformers_neuronx import ops
from transformers_neuronx import parallel
from transformers_neuronx import program
from transformers_neuronx import utils
from transformers_neuronx.gptneox import hlo
from transformers_neuronx.gptneox.config import GPTNeoXConfig
from transformers_neuronx.sampling import simple_sample


class GPTNeoXForSampling(module.PretrainedModel):

    def __init__(self, config, batch_size=1, amp='f32', tp_degree=2,
                 unroll=None, init_n_active_tokens=None, neuron_config=None, **kwargs):
        super().__init__()
        config = GPTNeoXConfig(config, batch_size, amp, tp_degree, **kwargs)
        self.debug = kwargs.get('debug', False)
        self.config = config
        self.neuron_config = neuron_config
        if self.config.activation_function not in ["gelu_new"]: # TODO see if we actually need to implement any other activation func variants
            warnings.warn(f'hidden_act="{self.config.activation_function}" ignored in favor of hidden_act="gelu_new"')
            self.config.activation_function = "gelu_new"
        if not self.config.use_parallel_residual: # TODO implement use_parallel_residual = False
            raise NotImplementedError(f'use_parallel_residual=False is not yet implemented')
        if neuron_config and neuron_config.quant:
            raise NotImplementedError(f'Support for quantization is not yet implemented')
        if unroll is not None: # TODO add support for unroll
            raise NotImplementedError(f'unroll={unroll} is not yet implemented')
        if unroll is None:
            unroll = config.n_layer
        self.unroll = unroll
        if init_n_active_tokens is not None: # TODO add support for init_n_active_tokens
            raise NotImplementedError(f'init_n_active_tokens={init_n_active_tokens} is not yet implemented')
        self.init_n_active_tokens = init_n_active_tokens
        dtype = dtypes.to_torch_dtype(config.amp)
        self.gpt_neox = GPTNeoXTransformer(config)
        self.embed_out = module.LowMemoryLazyLinear(config.vocab_size, dtype=dtype)
        self.ln_lm_head = GPTNeoXLnLmHead(config, self.gpt_neox.final_layer_norm, self.embed_out)
        self.n_positions_list = utils.power_of_two_bucket_sizes(128, config.n_positions)
        self.program = None
        self.init_program = program.DoNothingDecoder()
        self.manipulator = parallel.ParallelTensorManipulator(config.tp_degree)

    def to_neuron(self):
        ops.init()
        config = self.config
        n_positions_list = self.n_positions_list
        unroll = self.unroll
        self.program = build_gptneox_program(config, 1, n_positions_list, unroll, self.debug)
        self.gpt_neox.embed_in.materialize()
        for idx, block in enumerate(self.gpt_neox.layers):
            block.to_neuron(n_positions_list)
        self.ln_lm_head.to_neuron()
        self.program.setup(self.gpt_neox.layers, self.ln_lm_head)
        self.init_program.setup(self.gpt_neox.layers, self.ln_lm_head)

    def reset(self):
        for block in self.gpt_neox.layers:
            block.reset()

    def forward(self, input_ids, cache_offset, start_ids=None):
        batch_size, context_length = input_ids.shape
        if start_ids is None:
            start_ids = torch.zeros([self.config.batch_size], dtype=torch.int32)
        if cache_offset is None:
            cache_offset = torch.arange(context_length, dtype=torch.int32)
        last_offset = cache_offset[-1].item()
        bucket_id = find_first_ge_index(self.n_positions_list, last_offset)
        this_length = input_ids.shape[-1]
        init_length = self.init_program.init_length(this_length, self.init_n_active_tokens)
        init_step = self.init_program.init_step(self.init_n_active_tokens)
        for cur_len in range(0, init_length, init_step):
            slicing = slice(cur_len, cur_len + init_step)
            inputs = input_ids[:, slicing], cache_offset[slicing], start_ids
            logits = self._run_program(self.init_program, bucket_id, *inputs)
        for cur_len in range(init_length, this_length):
            slicing = slice(cur_len, cur_len + 1)
            inputs = input_ids[:, slicing], cache_offset[slicing], start_ids
            logits = self._run_program(self.program, bucket_id, *inputs)
            if self.debug:
                debug_tensors, debug_names = compiler.get_debug_outputs(self.program, bucket_id)
                for tensor, name in zip(debug_tensors, debug_names):
                    print("====================================")
                    print(f"Var name: {name}\nTensor:\n{tensor}")
                logits_cpu = self.manipulator.unshard_along(logits, dim=0)
        logits = self.manipulator.unshard_along(logits, dim=0)
        logits = logits.to(torch.float32)
        logits = logits[:self.config.vocab_size, -1, :]
        logits = logits.transpose(0, 1)
        return logits

    def _run_program(self, program, bucket_id, input_ids, cache_offset, start_ids):
        active_n_positions = self.n_positions_list[bucket_id]
        hidden = self.gpt_neox.embed_in(input_ids)
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
        start_ids = self.manipulator.duplicate_on_cpu(start_ids)
        for in_buffer, in_tensor in zip(input_buffers, [hidden, pos_embd, cache_offset, start_ids]):
            ops.parallel_write(in_buffer, in_tensor)
        return program.run(bucket_id)

    def sample(self, input_ids, sequence_length, start_ids=None, top_k=50):
        return simple_sample(self, input_ids, start_ids, sequence_length,
                             eos_token_id=self.config.eos_token_id, top_k=top_k)

    def _fixed_pos_embedding(self, dim, head_dim, offset):
        """
        GPT-NeoX Rotary Positional Embeddings Reference:
            rotary function defitions: https://github.com/huggingface/transformers/blob/v4.26-release/src/transformers/models/gpt_neox/modeling_gpt_neox.py#L242-L283
            rotary function usage: https://github.com/huggingface/transformers/blob/v4.26-release/src/transformers/models/gpt_neox/modeling_gpt_neox.py#L127-L142

        Differences compared to GPT-J:
            1. sin and cos:
                GPT-J uses
                    [
                        A, A, B, B, C, C, D, D,
                        E, E, F, F, G, G, H, H,
                        ...
                    ]
                GPT-NeoX uses
                    [
                        A, B, C, D, A, B, C, D,
                        E, F, G, H, E, F, G, H,
                        ...
                    ]
            2. rotation:
                GPT-J swaps every two elements
                GPT-NeoX swaps halves
        """
        rotary_emb_base = self.config.rotary_emb_base
        seq_len = offset + 1
        inv_freq = 1.0 / (rotary_emb_base ** (torch.arange(0, dim, 2) / dim))
        sinusoid_inp = torch.einsum("i , j -> i j", torch.arange(seq_len), inv_freq).float()
        sin = torch.sin(sinusoid_inp)
        cos = torch.cos(sinusoid_inp)

        # Stack sin and cos
        sin = torch.cat((sin[None, offset:seq_len, None, :], sin[None, offset:seq_len, None, :]), dim=-1)
        sin[..., : sin.shape[-1] // 2] *= -1 # multiply second half by -1
        cos = torch.cat((cos[None, offset:seq_len, None, :], cos[None, offset:seq_len, None, :]), dim=-1)

        sin_diag = torch.diagflat(sin)
        cos_diag = torch.diagflat(cos)

        # Swap halves
        rotate = torch.eye(sin.shape[-1])
        rotate[: sin.shape[-1] // 2, :], rotate[sin.shape[-1] // 2 :, :] = rotate[sin.shape[-1] // 2 :, :].clone(), rotate[: sin.shape[-1] // 2, :].clone()
        sincos = torch.matmul(rotate, sin_diag) + cos_diag
        # Only rotary_pct of this is used - we can optimize if necessary
        pos_embd = torch.eye(head_dim)
        pos_embd[:dim, :dim] = sincos
        return pos_embd


class GPTNeoXBuffers:

    def __init__(self, hlo_modules, dbg_outputs, tp_degree):
        first_hlo_module, *_ = hlo_modules
        hidden_buffer = compiler.gen_zero_input(first_hlo_module, 0)
        pos_embd_buffer = compiler.gen_zero_input(first_hlo_module, 1)
        cache_offset_buffer = compiler.gen_zero_input(first_hlo_module, 2)
        start_ids_buffer = compiler.gen_zero_input(first_hlo_module, 3)
        self.input_buffers = [hidden_buffer, pos_embd_buffer, cache_offset_buffer, start_ids_buffer]
        self.output_buffer = compiler.gen_zero_output(first_hlo_module, 0)
        self.debug_outputs = [compiler.gen_zero_output_from_shape(x) for x in dbg_outputs.get_tensors()]
        self.manipulator = parallel.ParallelTensorManipulator(tp_degree)

    def to_neuron(self):
        hidden_buffer, pos_embd_buffer, cache_offset_buffer, start_ids_buffer = self.input_buffers
        hidden_buffer = self.manipulator.duplicate(hidden_buffer)
        pos_embd_buffer = self.manipulator.duplicate(pos_embd_buffer)
        cache_offset_buffer = self.manipulator.duplicate(cache_offset_buffer)
        start_ids_buffer = self.manipulator.duplicate(start_ids_buffer)
        self.input_buffers = [hidden_buffer, pos_embd_buffer, cache_offset_buffer, start_ids_buffer]
        self.output_buffer = self.manipulator.duplicate(self.output_buffer)
        self.debug_outputs = [self.manipulator.duplicate(x) for x in self.debug_outputs]

    def get_input_buffers(self, bucket_id):
        hidden_buffer, pos_embd_buffer, cache_offset_buffer, start_ids_buffer = self.input_buffers
        return [hidden_buffer, pos_embd_buffer, cache_offset_buffer, start_ids_buffer]


def find_first_ge_index(values, target):
    return next(idx for idx, val in enumerate(values) if val >= target)


class GPTNeoXTransformer(module.LowMemoryModule):

    def __init__(self, config):
        super().__init__()
        self.embed_in = module.LowMemoryEmbedding(config.vocab_size, config.n_embd)
        self.layers = module.LowMemoryModuleList()
        for _ in range(config.n_layer):
            layer = GPTNeoXLayer(config)
            self.layers.append(layer)
        self.final_layer_norm = module.LowMemoryLayerNorm(config.n_embd)


class GPTNeoXLayer(module.LowMemoryModule):

    def __init__(self, config):
        super().__init__()
        self.input_layernorm = module.LowMemoryLayerNorm(config.n_embd)
        self.attention = GPTNeoXAttention(config)
        self.post_attention_layernorm = module.LowMemoryLayerNorm(config.n_embd)
        self.mlp = GPTNeoXMLP(config)
        self.config = config
        self.ln_1_weight = None
        self.ln_1_bias = None
        self.attn_q_weight = None
        self.attn_q_bias = None
        self.attn_k_weight = None
        self.attn_k_bias = None
        self.attn_v_weight = None
        self.attn_v_bias = None
        self.attn_out_weight = None
        self.attn_out_bias = None
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
        self.materialize()
        config = self.config

        manipulator = parallel.ParallelTensorManipulator(config.tp_degree)
        duplicate = manipulator.duplicate
        shard_along = manipulator.shard_along
        primary_only = manipulator.primary_only

        self.ln_1_weight = duplicate(self.input_layernorm.weight.detach())
        self.ln_1_bias = duplicate(self.input_layernorm.bias.detach())

        attention = self.attention

        n_head = self.config.n_head
        hidden_size = self.config.n_embd

        qkv = attention.query_key_value.weight
        qkv = qkv.reshape(n_head, 3, hidden_size // n_head, hidden_size).transpose(1, 0)
        qkv = qkv.reshape(3, hidden_size, hidden_size)

        qkv_bias = attention.query_key_value.bias
        qkv_bias = qkv_bias.reshape(n_head, 3, hidden_size // n_head).transpose(1, 0)
        qkv_bias = qkv_bias.reshape(3, hidden_size)

        q = qkv[0].T
        k = qkv[1].T
        v = qkv[2].T

        q_bias = qkv_bias[0]
        k_bias = qkv_bias[1]
        v_bias = qkv_bias[2]

        self.attn_q_weight = shard_along(q, dim=1)
        self.attn_q_bias = shard_along(q_bias, dim=0)
        self.attn_k_weight = shard_along(k, dim=1)
        self.attn_k_bias = shard_along(k_bias, dim=0)
        self.attn_v_weight = shard_along(v, dim=1)
        self.attn_v_bias = shard_along(v_bias, dim=0)
        self.attn_out_weight = shard_along(attention.dense.weight.detach().T, dim=0)
        self.attn_out_bias = primary_only(attention.dense.bias.detach())

        self.ln_2_weight = duplicate(self.post_attention_layernorm.weight.detach())
        self.ln_2_bias = duplicate(self.post_attention_layernorm.bias.detach())

        mlp = self.mlp
        self.mlp_in_weight = shard_along(mlp.dense_h_to_4h.weight.detach().T, dim=1)
        self.mlp_in_bias = shard_along(mlp.dense_h_to_4h.bias.detach(), dim=0)
        self.mlp_out_weight = shard_along(mlp.dense_4h_to_h.weight.detach().T, dim=0)
        self.mlp_out_bias = primary_only(mlp.dense_4h_to_h.bias.detach())

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


class GPTNeoXAttention(module.LowMemoryModule):

    def __init__(self, config):
        super().__init__()
        n_embd = config.n_embd
        dtype = dtypes.to_torch_dtype(config.amp)
        self.query_key_value = module.LowMemoryLazyLinear(3 * n_embd, bias=True, dtype=dtype)
        self.dense = module.LowMemoryLazyLinear(n_embd, bias=True, dtype=dtype)


class GPTNeoXMLP(module.LowMemoryModule):

    def __init__(self, config):
        super().__init__()
        dtype = dtypes.to_torch_dtype(config.amp)
        self.dense_h_to_4h = module.LowMemoryLazyLinear(config.intermediate_dim, dtype=dtype) # [intermediate_dim, n_embd]
        self.dense_4h_to_h = module.LowMemoryLazyLinear(config.n_embd, dtype=dtype) # [n_embd, intermediate_dim]


class GPTNeoXLnLmHead:

    def __init__(self, config, final_layer_norm, embed_out):
        self.tp_degree = config.tp_degree
        self.final_layer_norm = final_layer_norm
        self.embed_out = embed_out # Linear layer without bias (analogous to lm_head in GPT-J)
        self.ln_f_weight = None
        self.ln_f_bias = None
        self.embed_out_weight = None
        self.manipulator = parallel.ParallelTensorManipulator(self.tp_degree)

    def to_neuron(self):
        duplicate = self.manipulator.duplicate
        shard_along = self.manipulator.shard_along
        self.final_layer_norm.materialize()
        self.ln_f_weight = duplicate(self.final_layer_norm.weight.detach())
        self.ln_f_bias = duplicate(self.final_layer_norm.bias.detach())
        self.embed_out.materialize()
        # Pad the lm_head_weight if vocab_size % tp_degree != 0
        embed_out_weight = self.embed_out.weight.detach().T
        _, vocab_size = embed_out_weight.shape
        vocab_pad = utils.pad_vocab_size(vocab_size, self.tp_degree)
        embed_out_weight = torch.nn.functional.pad(embed_out_weight, (0, vocab_pad, 0, 0))
        self.embed_out_weight = shard_along(embed_out_weight, dim=1)
        self.embed_out.nullify()

    def get_parameters(self):
        return [self.ln_f_weight, self.ln_f_bias, self.embed_out_weight]


def build_gptneox_program(config, n_active, n_positions_list, n_blocks, debug):
    debugger = program.Debugger(debug)
    hlo_modules = [hlo.build_gptneox_hlo_module(config, n_active, npos, debugger=debugger) for npos in n_positions_list]
    buffers = GPTNeoXBuffers(hlo_modules, debugger, config.tp_degree)
    if n_blocks == config.n_layer:
        return program.FullyUnrolledDecoder(config.tp_degree, hlo_modules, buffers, debugger)
    else:
        # We are here if unroll is specified and it's not equal to n_layer
        raise NotImplementedError('unroll != n_layer is not yet implemented') # TODO implement unroll