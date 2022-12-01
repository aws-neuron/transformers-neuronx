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
from transformers_neuronx import program
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
        self.unroll = unroll
        self.init_n_active_tokens = init_n_active_tokens
        dtype = dtypes.to_torch_dtype(config.amp)
        self.model = OPTModel(config)
        self.lm_head = module.LowMemoryLazyLinear(config.vocab_size, dtype=dtype, bias=False)
        self.ln_lm_head = OPTLnLmHead(config, self.model.decoder.final_layer_norm, self.lm_head)
        self.n_positions_list = utils.power_of_two_bucket_sizes(128, n_positions)
        self.program = None
        self.init_program = program.DoNothingDecoder()
        self.manipulator = parallel.ParallelTensorManipulator(tp_degree)
        self.to_neuron_hooks = []

    def to_neuron(self):
        ops.init()
        config = self.config
        n_positions_list = self.n_positions_list
        unroll = self.unroll
        self.program = build_opt_program(config, 1, n_positions_list, unroll)
        init_n_active = self.init_n_active_tokens
        if init_n_active is not None:
            self.init_program = build_opt_program(config, init_n_active, n_positions_list, unroll)
        decoder = self.model.decoder
        decoder.embed_tokens.materialize()
        decoder.embed_positions.materialize()
        parallel.layers_to_neuron(16, decoder.layers, n_positions_list, self.to_neuron_hooks)
        self.ln_lm_head.to_neuron()
        self.program.setup(self.model.decoder.layers, self.ln_lm_head)
        self.init_program.setup(self.model.decoder.layers, self.ln_lm_head)

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
        input_buffers = program.buffers.get_input_buffers(bucket_id)
        hidden_buffer, *_ = input_buffers
        hidden = hidden.to(hidden_buffer.dtype)
        hidden = self.manipulator.duplicate_on_cpu(hidden)
        cache_offset = self.manipulator.duplicate_on_cpu(cache_offset)
        mask = self.manipulator.duplicate_on_cpu(mask)
        for in_buffer, in_tensor in zip(input_buffers, [hidden, cache_offset, mask]):
            ops.parallel_write(in_buffer, in_tensor)
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
            topk_values, topk_indices = torch.topk(next_token_scores, top_k)

            # sample
            probs = torch.nn.functional.softmax(topk_values, dim=-1)
            inputs_in_topk = torch.multinomial(probs, num_samples=1, replacement=True)
            inputs = torch.gather(topk_indices, 1, inputs_in_topk)
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

    def __init__(self, opt_hlo_modules, tp_degree):
        first_hlo_module, *_ = opt_hlo_modules
        hidden_buffer = compiler.gen_zero_input(first_hlo_module, 0)
        cache_offset_buffer = compiler.gen_zero_input(first_hlo_module, 1)
        mask_buffers = [compiler.gen_zero_input(hlo, 2) for hlo in opt_hlo_modules]
        self.input_buffers = [hidden_buffer, cache_offset_buffer, mask_buffers]
        self.output_buffer = compiler.gen_zero_output(first_hlo_module, 0)
        self.manipulator = parallel.ParallelTensorManipulator(tp_degree)

    def to_neuron(self):
        hidden_buffer, cache_offset_buffer, mask_buffers = self.input_buffers
        hidden_buffer = self.manipulator.duplicate(hidden_buffer)
        cache_offset_buffer = self.manipulator.duplicate(cache_offset_buffer)
        mask_buffers = [self.manipulator.duplicate(mask_buffer) for mask_buffer in mask_buffers]
        self.input_buffers = [hidden_buffer, cache_offset_buffer, mask_buffers]
        self.output_buffer = self.manipulator.duplicate(self.output_buffer)

    def get_input_buffers(self, bucket_id):
        hidden_buffer, cache_offset_buffer, mask_buffers = self.input_buffers
        return [hidden_buffer, cache_offset_buffer, mask_buffers[bucket_id]]


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
        self.vocab_pad = utils.pad_vocab_size(vocab_size, tp_degree)

    def to_neuron(self):
        self.ln_f.materialize()
        self.ln_f_weight = self.manipulator.duplicate(self.ln_f.weight.detach())
        self.ln_f_bias = self.manipulator.duplicate(self.ln_f.bias.detach())
        self.lm_head.materialize()
        lm_head_weight = self.lm_head.weight.detach()
        lm_head_weight = torch.nn.functional.pad(lm_head_weight, (0, 0, 0, self.vocab_pad))
        self.lm_head_weight = self.manipulator.shard_along(lm_head_weight.T, dim=1)
        self.lm_head.weight = UninitializedParameter()

    def get_parameters(self):
        return [self.ln_f_weight, self.ln_f_bias, self.lm_head_weight]


def build_opt_program(config, n_active, n_positions_list, n_layers):
    hlo_modules = [hlo.build_opt_hlo_module(config, n_active, npos) for npos in n_positions_list]
    buffers = OPTBuffers(hlo_modules, config.tp_degree)
    if n_layers == config.num_hidden_layers:
        return program.FullyUnrolledDecoder(config, hlo_modules, buffers)
    else:
        build_func = hlo.build_opt_multi_layer_hlo_module
        hlo_modules = [build_func(config, n_active, npos, n_layers) for npos in n_positions_list]
        head_hlo_module = hlo.build_ln_lm_head_hlo_module(config, n_active)
        return program.MultiLayerDecoder(config, hlo_modules, head_hlo_module, n_layers, buffers)
