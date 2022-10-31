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
from transformers_neuronx import dtypes
from transformers_neuronx import module
from transformers_neuronx import parallel
from transformers_neuronx.gpt2.model import GPT2LnLmHead
from transformers_neuronx.opt import hlo
from transformers_neuronx.opt.config import OPTConfig


class OPTForSampling(module.PretrainedModel):

    def __init__(self, config, batch_size=1, init_n_active_tokens=8, amp='f32', tp_degree=2,
                 n_positions=2048, unroll=None, fast_init=False, **kwargs):
        super().__init__()
        config = OPTConfig(config, n_positions, batch_size, amp, tp_degree, **kwargs)
        # power-of-2 bucket sizes
        n_positions_list = []
        bucket_size = 128
        while bucket_size < n_positions:
            n_positions_list.append(bucket_size)
            bucket_size *= 2
        n_positions_list.append(n_positions)
        self.n_positions_list = n_positions_list
        self.active_n_positions_index = None
        self.opt_kernels = None
        self.opt_init_kernels = None
        block_kernels = None
        ln_lm_head_kernel = None
        block_init_kernels = None
        ln_lm_head_init_kernel = None
        n_blocks = unroll
        self.fast_init = fast_init
        if unroll is None:
            block_kernels = [hlo.build_opt_block_kernel(config, n_active_tokens=1, n_positions=npos)
                             for npos in n_positions_list]
            ln_lm_head_kernel = hlo.build_lm_head_kernel(config, n_active_tokens=1)
            if fast_init:
                block_init_kernels = [hlo.build_opt_block_kernel(config, init_n_active_tokens, npos)
                                      for npos in n_positions_list]
                ln_lm_head_init_kernel = hlo.build_lm_head_kernel(config, init_n_active_tokens)
        elif unroll == config.num_hidden_layers:
            self.opt_kernels = [hlo.build_opt_kernel(config, n_active_tokens=1, n_positions=npos)
                                for npos in n_positions_list]
            if fast_init:
                self.opt_init_kernels = [hlo.build_opt_kernel(config, init_n_active_tokens, npos)
                                         for npos in n_positions_list]
        else:
            if config.num_hidden_layers % unroll != 0:
                raise ValueError(
                    f'unroll={unroll} does not divide num_hidden_layers={config.num_hidden_layers}')
            block_kernels = [hlo.build_opt_multi_block_kernel(config, 1, npos, n_blocks)
                             for npos in n_positions_list]
            ln_lm_head_kernel = hlo.build_lm_head_kernel(config, n_active_tokens=1)
            if fast_init:
                block_init_kernels = []
                for npos in n_positions_list:
                    kernel = hlo.build_opt_multi_block_kernel(config, init_n_active_tokens, npos,
                                                              n_blocks)
                    block_init_kernels.append(kernel)
                ln_lm_head_init_kernel = hlo.build_lm_head_kernel(config, init_n_active_tokens)
        self.init_n_active_tokens = init_n_active_tokens
        self.model = OPTModel(config, block_kernels, block_init_kernels, n_blocks=n_blocks)
        dtype = dtypes.to_torch_dtype(config.amp)
        self.lm_head = module.LowMemoryLazyLinear(config.vocab_size, dtype=dtype, bias=False)
        self.config = config
        ln_f = self.model.decoder.final_layer_norm
        self.ln_lm_head = GPT2LnLmHead(config, ln_lm_head_kernel, ln_lm_head_init_kernel,
                                       ln_f, self.lm_head)
        self.manipulator = parallel.TensorManipulator(config.tp_degree)
        self.opt_caches = None
        self.opt_params = None
        self.active_opt_kernel = None
        self.active_opt_init_kernel = None
        self.to_neuron_hooks = []

    def to_neuron(self):
        self.model.decoder.embed_tokens.materialize()
        self.model.decoder.embed_positions.materialize()
        if self.opt_kernels is not None:
            for kernel in self.opt_kernels:
                kernel.load()
        if self.opt_init_kernels is not None:
            for kernel in self.opt_init_kernels:
                kernel.load()
        first_block, *_ = self.model.decoder.layers
        first_block.maybe_load_kernels()
        if self.model.decoder.multi_blocks is not None:
            first_multi_block, *_ = self.model.decoder.multi_blocks
            first_multi_block.maybe_load_kernels()
        for idx, block in enumerate(self.model.decoder.layers):
            block.to_neuron()
            for hook in self.to_neuron_hooks:
                hook(idx)
        self.ln_lm_head.to_neuron()

    def reset(self):
        for block in self.model.decoder.layers:
            block.reset(self.n_positions_list)
        self.activate_n_positions_index(0)
        if self.model.decoder.multi_blocks is not None:
            for multi_block in self.model.decoder.multi_blocks:
                multi_block.reset()
        self.opt_params = []
        for block in self.model.decoder.layers:
            block_params = [
                block.ln_1_weight,
                block.ln_1_bias,
                block.attn_q_weight,
                block.attn_q_bias,
                block.attn_k_weight,
                block.attn_k_bias,
                block.attn_v_weight,
                block.attn_v_bias,
                block.attn_out_weight,
                block.attn_out_bias,
                block.ln_2_weight,
                block.ln_2_bias,
                block.mlp_in_weight,
                block.mlp_in_bias,
                block.mlp_out_weight,
                block.mlp_out_bias,
            ]
            self.opt_params.extend(block_params)
        ln_lm_head = self.ln_lm_head
        ln_lm_head_params = [
            ln_lm_head.ln_f_weight,
            ln_lm_head.ln_f_bias,
            ln_lm_head.lm_head_weight,
        ]
        self.opt_params.extend(ln_lm_head_params)

    def forward(self, input_ids, cache_offset, mask):
        last_offset = cache_offset[-1].item()
        n_positions_index = find_first_ge_index(self.n_positions_list, last_offset)
        if n_positions_index != self.active_n_positions_index:
            self.activate_n_positions_index(n_positions_index)
        this_length = input_ids.shape[-1]
        if this_length > 1:
            return self._forward_init_dynamic_shape(input_ids, cache_offset, mask)
        return self._forward_one_token(input_ids, cache_offset, mask)

    def activate_n_positions_index(self, n_positions_index):
        self.opt_caches = []
        for block in self.model.decoder.layers:
            block.activate_n_positions_index(n_positions_index)
            self.opt_caches.append(block.active_key_cache)
            self.opt_caches.append(block.active_value_cache)
        if self.model.decoder.multi_blocks is not None:
            for multi_block in self.model.decoder.multi_blocks:
                multi_block.activate_n_positions_index(n_positions_index)
        self.active_n_positions_index = n_positions_index
        if self.opt_kernels is not None:
            self.active_opt_kernel = self.opt_kernels[n_positions_index]
        if self.opt_init_kernels is not None:
            self.active_opt_init_kernel = self.opt_init_kernels[n_positions_index]

    def _forward_init_dynamic_shape(self, input_ids, cache_offset, mask):
        this_length = input_ids.shape[-1]
        init_length = 0
        if self.fast_init:
            init_n_active_tokens = self.init_n_active_tokens
            init_length = this_length // init_n_active_tokens * init_n_active_tokens
            for cur_len in range(0, init_length, init_n_active_tokens):
                next_len = cur_len + init_n_active_tokens
                input_ids_slice = input_ids[:, cur_len:next_len]
                cache_offset_slice = cache_offset[cur_len:next_len]
                mask_slice = mask[cur_len:next_len]
                logits = self._forward_init(input_ids_slice, cache_offset_slice, mask_slice)
        for cur_len in range(init_length, this_length):
            next_len = cur_len + 1
            input_ids_slice = input_ids[:, cur_len:next_len]
            cache_offset_slice = cache_offset[cur_len:next_len]
            mask_slice = mask[cur_len:next_len]
            logits = self._forward_one_token(input_ids_slice, cache_offset_slice, mask_slice)
        return logits

    def _forward_init(self, input_ids, cache_offset, mask):
        if self.model.decoder.multi_blocks is not None:
            return self._forward_init_multi_block(input_ids, cache_offset, mask)
        if self.opt_kernels is None:
            return self._forward_init_layered(input_ids, cache_offset, mask)
        hidden, cache_offset, mask = self._process_inputs(input_ids, cache_offset, mask)
        inputs_cores = hidden, cache_offset, mask, *self.opt_caches, *self.opt_params
        logits, *_ = self.active_opt_init_kernel(inputs_cores)
        return self._process_outputs(logits)

    def _forward_init_layered(self, input_ids, cache_offset, mask):
        hidden, cache_offset, mask = self._process_inputs(input_ids, cache_offset, mask)
        for block in self.model.decoder.layers:
            hidden = block.forward_init(hidden, cache_offset, mask)
        logits = self.ln_lm_head.call_init(hidden)
        return self._process_outputs(logits)

    def _forward_init_multi_block(self, input_ids, cache_offset, mask):
        hidden, cache_offset, mask = self._process_inputs(input_ids, cache_offset, mask)
        for multi_block in self.model.decoder.multi_blocks:
            hidden = multi_block.call_init(hidden, cache_offset, mask)
        logits = self.ln_lm_head.call_init(hidden)
        return self._process_outputs(logits)

    def _forward_one_token(self, input_ids, cache_offset, mask):
        if self.model.decoder.multi_blocks is not None:
            return self._forward_one_token_multi_block(input_ids, cache_offset, mask)
        if self.opt_kernels is None:
            return self._forward_one_token_layered(input_ids, cache_offset, mask)
        hidden, cache_offset, mask = self._process_inputs(input_ids, cache_offset, mask)
        inputs_cores = hidden, cache_offset, mask, *self.opt_caches, *self.opt_params
        logits, *_ = self.active_opt_kernel(inputs_cores)
        return self._process_outputs(logits)

    def _forward_one_token_layered(self, input_ids, cache_offset, mask):
        hidden, cache_offset, mask = self._process_inputs(input_ids, cache_offset, mask)
        for block in self.model.decoder.layers:
            hidden = block(hidden, cache_offset, mask)
        logits = self.ln_lm_head(hidden)
        return self._process_outputs(logits)

    def _forward_one_token_multi_block(self, input_ids, cache_offset, mask):
        hidden, cache_offset, mask = self._process_inputs(input_ids, cache_offset, mask)
        for multi_block in self.model.decoder.multi_blocks:
            hidden = multi_block(hidden, cache_offset, mask)
        logits = self.ln_lm_head(hidden)
        return self._process_outputs(logits)

    def _process_inputs(self, input_ids, cache_offset, mask):
        active_n_positions = self.n_positions_list[self.active_n_positions_index]
        mask = mask[:, :active_n_positions].contiguous()
        inputs_embeds = self.model.decoder.embed_tokens(input_ids)
        past_length = cache_offset[0].item()
        this_length = input_ids.shape[-1]
        position_ids = torch.arange(past_length, past_length + this_length)
        position_ids = position_ids.unsqueeze(0).view(-1, this_length)
        position_embeds = self.model.decoder.embed_positions(position_ids)
        hidden = inputs_embeds + position_embeds
        hidden = hidden.transpose(0, -1)
        dtype = dtypes.to_torch_dtype(self.config.amp)
        hidden = hidden.to(dtype)
        duplicate = self.manipulator.duplicate
        hidden = duplicate(hidden)
        cache_offset = duplicate(cache_offset)
        mask = duplicate(mask)
        return hidden, cache_offset, mask

    def _process_outputs(self, logits):
        logits = self.manipulator.unshard_along(logits, dim=0)
        logits = logits.to(torch.float32)
        logits = logits[:self.config.vocab_size]
        logits = logits.transpose(0, -1)
        logits = logits[:, -1, :]
        return logits

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
            print('cur_len=', cur_len)
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


def find_first_ge_index(values, target):
    return next(idx for idx, val in enumerate(values) if val >= target)


class OPTDecoder(module.LowMemoryModule):

    def __init__(self, config, block_kernels, block_init_kernels, n_blocks):
        super().__init__()
        self.embed_tokens = module.LowMemoryEmbedding(config.vocab_size, config.hidden_size,
                                                      padding_idx=config.pad_token_id)
        self.embed_positions = OPTLearnedPositionalEmbedding(config.max_position_embeddings,
                                                             config.hidden_size)
        self.layers = module.LowMemoryModuleList()
        enable_multi_block = n_blocks is not None and n_blocks < config.num_hidden_layers
        for _ in range(config.num_hidden_layers):
            block = OPTBlock(config, block_kernels, block_init_kernels)
            if enable_multi_block:
                block = OPTBlock(config, None, None)
            self.layers.append(block)
        self.multi_blocks = None
        if enable_multi_block:
            self.multi_blocks = []
            for start in range(0, config.num_hidden_layers, n_blocks):
                blocks = self.layers[start:start+n_blocks]
                multi_block = OPTMultiBlock(config, blocks, block_kernels, block_init_kernels)
                self.multi_blocks.append(multi_block)
        self.final_layer_norm = module.LowMemoryLayerNorm(config.hidden_size)


class OPTModel(module.LowMemoryModule):

    def __init__(self, config, block_kernels, block_init_kernels, n_blocks):
        super().__init__()
        self.decoder = OPTDecoder(config, block_kernels, block_init_kernels, n_blocks)


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


class OPTBlock(module.LowMemoryModule):

    def __init__(self, config, kernels, init_kernels):
        super().__init__()
        self.self_attn_layer_norm = module.LowMemoryLayerNorm(config.hidden_size)
        self.self_attn = OPTAttention(config)
        dtype = dtypes.to_torch_dtype(config.amp)
        self.fc1 = module.LowMemoryLazyLinear(config.hidden_size, dtype=dtype)
        self.fc2 = module.LowMemoryLazyLinear(config.ffn_dim, dtype=dtype)
        self.final_layer_norm = module.LowMemoryLayerNorm(config.hidden_size)
        self.config = config
        self.kernels = kernels
        self.init_kernels = init_kernels
        self.active_kernel = None
        self.active_init_kernel = None
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
        self.value_cache = None
        self.active_key_cache = None
        self.active_value_cache = None
        self.params = None

    def to_neuron(self):
        manipulator = parallel.TensorManipulator(self.config.tp_degree)
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

    def maybe_load_kernels(self):
        if self.kernels is not None:
            for kernel in self.kernels:
                kernel.load()
        if self.init_kernels is not None:
            for kernel in self.init_kernels:
                kernel.load()

    def reset(self, n_positions_list):
        config = self.config
        manipulator = parallel.TensorManipulator(config.tp_degree)
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
        self.params = [
            self.ln_1_weight, self.ln_1_bias, self.attn_q_weight, self.attn_q_bias,
            self.attn_k_weight, self.attn_k_bias, self.attn_v_weight, self.attn_v_bias,
            self.attn_out_weight, self.attn_out_bias, self.ln_2_weight, self.ln_2_bias,
            self.mlp_in_weight, self.mlp_in_bias, self.mlp_out_weight, self.mlp_out_bias,
        ]
        self.activate_n_positions_index(0)

    def forward(self, hidden, cache_offset, mask):
        inputs_cores = [hidden, cache_offset, mask, self.active_key_cache, self.active_value_cache,
                        *self.params]
        hidden, *_ = self.active_kernel(inputs_cores)
        return hidden

    def forward_init(self, hidden, cache_offset, mask):
        inputs_cores = [hidden, cache_offset, mask, self.active_key_cache, self.active_value_cache,
                        *self.params]
        hidden, *_ = self.active_init_kernel(inputs_cores)
        return hidden

    def activate_n_positions_index(self, n_positions_index):
        self.active_key_cache = self.key_cache_slices[n_positions_index]
        self.active_value_cache = self.value_cache_slices[n_positions_index]
        if self.kernels is not None:
            self.active_kernel = self.kernels[n_positions_index]
        if self.init_kernels is not None:
            self.active_init_kernel = self.init_kernels[n_positions_index]


class OPTAttention(module.LowMemoryModule):

    def __init__(self, config):
        super().__init__()
        hidden_size = config.hidden_size
        dtype = dtypes.to_torch_dtype(config.amp)
        self.q_proj = module.LowMemoryLazyLinear(hidden_size, dtype=dtype)
        self.k_proj = module.LowMemoryLazyLinear(hidden_size, dtype=dtype)
        self.v_proj = module.LowMemoryLazyLinear(hidden_size, dtype=dtype)
        self.out_proj = module.LowMemoryLazyLinear(hidden_size, dtype=dtype)


class OPTMultiBlock:

    def __init__(self, config, blocks, kernels, init_kernels):
        self.blocks = blocks
        self.kernels = kernels
        self.init_kernels = init_kernels
        self.active_kernel = None
        self.active_init_kernel = None
        self.caches = None
        self.params = None
        self.manipulator = parallel.TensorManipulator(config.tp_degree)

    def reset(self):
        self.params = []
        for block in self.blocks:
            block_params = [
                block.ln_1_weight,
                block.ln_1_bias,
                block.attn_q_weight,
                block.attn_q_bias,
                block.attn_k_weight,
                block.attn_k_bias,
                block.attn_v_weight,
                block.attn_v_bias,
                block.attn_out_weight,
                block.attn_out_bias,
                block.ln_2_weight,
                block.ln_2_bias,
                block.mlp_in_weight,
                block.mlp_in_bias,
                block.mlp_out_weight,
                block.mlp_out_bias,
            ]
            self.params.extend(block_params)
        self.activate_n_positions_index(0)

    def maybe_load_kernels(self):
        if self.kernels is not None:
            for kernel in self.kernels:
                kernel.load()
        if self.init_kernels is not None:
            for kernel in self.init_kernels:
                kernel.load()

    def __call__(self, hidden, cache_offset, mask):
        inputs_cores = [hidden, cache_offset, mask, *self.caches, *self.params]
        hidden, *_ = self.active_kernel(inputs_cores)
        return hidden

    def call_init(self, hidden, cache_offset, mask):
        inputs_cores = [hidden, cache_offset, mask, *self.caches, *self.params]
        hidden, *_ = self.active_init_kernel(inputs_cores)
        return hidden

    def activate_n_positions_index(self, n_positions_index):
        if self.kernels is not None:
            self.active_kernel = self.kernels[n_positions_index]
        if self.init_kernels is not None:
            self.active_init_kernel = self.init_kernels[n_positions_index]
        self.caches = []
        for block in self.blocks:
            self.caches.append(block.active_key_cache)
            self.caches.append(block.active_value_cache)
