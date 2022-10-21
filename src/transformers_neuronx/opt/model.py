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
                 n_positions=2048, unroll=False, **kwargs):
        super().__init__()
        config = OPTConfig(config, n_positions, batch_size, amp, tp_degree, **kwargs)
        self.opt_kernel = None
        self.opt_init_kernel = None
        if unroll:
            block_kernel = None
            ln_lm_head_kernel = None
            block_init_kernel = None
            ln_lm_head_init_kernel = None
            self.opt_kernel = hlo.build_opt_kernel(config, n_active_tokens=1, n_positions=n_positions)
            self.opt_init_kernel = hlo.build_opt_kernel(config, init_n_active_tokens, n_positions)
        else:
            block_kernel = hlo.build_opt_block_kernel(config, n_active_tokens=1,
                                                      n_positions=n_positions)
            ln_lm_head_kernel = hlo.build_lm_head_kernel(config, n_active_tokens=1)
            block_init_kernel = hlo.build_opt_block_kernel(config, init_n_active_tokens, n_positions)
            ln_lm_head_init_kernel = hlo.build_lm_head_kernel(config, init_n_active_tokens)
        self.init_n_active_tokens = init_n_active_tokens
        self.model = OPTModel(config, block_kernel, block_init_kernel)
        dtype = dtypes.to_torch_dtype(config.amp)
        self.lm_head = module.LowMemoryLazyLinear(config.vocab_size, dtype=dtype)
        self.config = config
        ln_f = self.model.decoder.final_layer_norm
        self.ln_lm_head = GPT2LnLmHead(config, ln_lm_head_kernel, ln_lm_head_init_kernel,
                                       ln_f, self.lm_head)
        self.manipulator = parallel.TensorManipulator(config.tp_degree)
        self.opt_caches = None
        self.opt_params = None

    def to_neuron(self):
        if self.opt_kernel is not None:
            self.opt_kernel.load()
        if self.opt_init_kernel is not None:
            self.opt_init_kernel.load()
        for idx, block in enumerate(self.model.decoder.layers):
            block.to_neuron()
        self.ln_lm_head.to_neuron()

    def reset(self):
        for block in self.model.decoder.layers:
            block.reset()
        self.opt_caches = []
        for block in self.model.decoder.layers:
            self.opt_caches.append(block.key_cache)
            self.opt_caches.append(block.value_cache)
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
        this_length = input_ids.shape[-1]
        if this_length > 1:
            return self._forward_init_dynamic_shape(input_ids, cache_offset, mask)
        return self._forward_one_token(input_ids, cache_offset, mask)

    def _forward_init_dynamic_shape(self, input_ids, cache_offset, mask):
        this_length = input_ids.shape[-1]
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
        if self.opt_kernel is None:
            return self._forward_init_layered(input_ids, cache_offset, mask)
        hidden, cache_offset, mask = self._process_inputs(input_ids, cache_offset, mask)
        inputs_cores = hidden, cache_offset, mask, *self.opt_caches, *self.opt_params
        logits, *_ = self.opt_init_kernel(inputs_cores)
        return self._process_outputs(logits)

    def _forward_init_layered(self, input_ids, cache_offset, mask):
        hidden, cache_offset, mask = self._process_inputs(input_ids, cache_offset, mask)
        for block in self.model.decoder.layers:
            hidden = block.forward_init(hidden, cache_offset, mask)
        logits = self.ln_lm_head.call_init(hidden)
        return self._process_outputs(logits)

    def _forward_one_token(self, input_ids, cache_offset, mask):
        if self.opt_kernel is None:
            return self._forward_one_token_layered(input_ids, cache_offset, mask)
        hidden, cache_offset, mask = self._process_inputs(input_ids, cache_offset, mask)
        inputs_cores = hidden, cache_offset, mask, *self.opt_caches, *self.opt_params
        logits, *_ = self.opt_kernel(inputs_cores)
        return self._process_outputs(logits)

    def _forward_one_token_layered(self, input_ids, cache_offset, mask):
        hidden, cache_offset, mask = self._process_inputs(input_ids, cache_offset, mask)
        for block in self.model.decoder.layers:
            hidden = block(hidden, cache_offset, mask)
        logits = self.ln_lm_head(hidden)
        return self._process_outputs(logits)

    def _process_inputs(self, input_ids, cache_offset, mask):
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
        min_length = max_length = top_k = sequence_length
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


class OPTDecoder(module.LowMemoryModule):

    def __init__(self, config, block_kernel, block_init_kernel):
        super().__init__()
        self.embed_tokens = module.LowMemoryEmbedding(config.vocab_size, config.hidden_size,
                                                      padding_idx=config.pad_token_id)
        self.embed_positions = OPTLearnedPositionalEmbedding(config.max_position_embeddings,
                                                             config.hidden_size)
        self.layers = module.LowMemoryModuleList()
        for _ in range(config.num_hidden_layers):
            self.layers.append(OPTBlock(config, block_kernel, block_init_kernel))
        self.final_layer_norm = module.LowMemoryLayerNorm(config.hidden_size)


class OPTModel(module.LowMemoryModule):

    def __init__(self, config, block_kernel, block_init_kernel):
        super().__init__()
        self.decoder = OPTDecoder(config, block_kernel, block_init_kernel)


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

    def __init__(self, config, kernel, init_kernel):
        super().__init__()
        self.self_attn_layer_norm = module.LowMemoryLayerNorm(config.hidden_size)
        self.self_attn = OPTAttention(config)
        dtype = dtypes.to_torch_dtype(config.amp)
        self.fc1 = module.LowMemoryLazyLinear(config.hidden_size, dtype=dtype)
        self.fc2 = module.LowMemoryLazyLinear(config.ffn_dim, dtype=dtype)
        self.final_layer_norm = module.LowMemoryLayerNorm(config.hidden_size)
        self.config = config
        self.kernel = kernel
        self.init_kernel = init_kernel
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
        self.params = None

    def to_neuron(self):
        if self.kernel is not None:
            self.kernel.load()
        if self.init_kernel is not None:
            self.init_kernel.load()
        manipulator = parallel.TensorManipulator(self.config.tp_degree)
        duplicate = manipulator.duplicate
        shard_along = manipulator.shard_along
        primary_only = manipulator.primary_only
        self.ln_1_weight = duplicate(self.self_attn_layer_norm.weight.detach())
        self.ln_1_bias = duplicate(self.self_attn_layer_norm.bias.detach())
        q_proj = self.self_attn.q_proj
        k_proj = self.self_attn.k_proj
        v_proj = self.self_attn.v_proj
        out_proj = self.self_attn.out_proj
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
        self.ln_2_weight = duplicate(self.final_layer_norm.weight.detach())
        self.ln_2_bias = duplicate(self.final_layer_norm.bias.detach())
        fc1 = self.fc1
        fc2 = self.fc2
        self.mlp_in_weight = shard_along(fc1.weight.detach().T, dim=1)
        self.mlp_in_bias = shard_along(fc1.bias.detach(), dim=0)
        self.mlp_out_weight = shard_along(fc2.weight.detach().T, dim=0)
        self.mlp_out_bias = primary_only(fc2.bias.detach())
        fc1.weight = UninitializedParameter()
        fc2.weight = UninitializedParameter()

    def reset(self):
        config = self.config
        manipulator = parallel.TensorManipulator(config.tp_degree)
        n_positions = config.n_positions
        batch_size = config.batch_size
        n_head = config.num_attention_heads
        s_head = config.hidden_size // n_head
        cache_shape = [n_positions, batch_size, n_head, s_head]
        dtype = dtypes.to_torch_dtype(config.amp)
        self.key_cache = torch.zeros(cache_shape, dtype=dtype)
        self.key_cache = manipulator.shard_along(self.key_cache, dim=2)
        self.value_cache = torch.zeros(cache_shape, dtype=dtype)
        self.value_cache = manipulator.shard_along(self.value_cache, dim=2)
        self.params = [
            self.ln_1_weight, self.ln_1_bias, self.attn_q_weight, self.attn_q_bias,
            self.attn_k_weight, self.attn_k_bias, self.attn_v_weight, self.attn_v_bias,
            self.attn_out_weight, self.attn_out_bias, self.ln_2_weight, self.ln_2_bias,
            self.mlp_in_weight, self.mlp_in_bias, self.mlp_out_weight, self.mlp_out_bias,
        ]

    def forward(self, hidden, cache_offset, mask):
        inputs_cores = [hidden, cache_offset, mask, self.key_cache, self.value_cache, *self.params]
        hidden, self.key_cache, self.value_cache = self.kernel(inputs_cores)
        return hidden

    def forward_init(self, hidden, cache_offset, mask):
        inputs_cores = [hidden, cache_offset, mask, self.key_cache, self.value_cache, *self.params]
        hidden, self.key_cache, self.value_cache = self.init_kernel(inputs_cores)
        return hidden


class OPTAttention(module.LowMemoryModule):

    def __init__(self, config):
        super().__init__()
        hidden_size = config.hidden_size
        dtype = dtypes.to_torch_dtype(config.amp)
        self.q_proj = module.LowMemoryLazyLinear(hidden_size, dtype=dtype)
        self.k_proj = module.LowMemoryLazyLinear(hidden_size, dtype=dtype)
        self.v_proj = module.LowMemoryLazyLinear(hidden_size, dtype=dtype)
        self.out_proj = module.LowMemoryLazyLinear(hidden_size, dtype=dtype)
