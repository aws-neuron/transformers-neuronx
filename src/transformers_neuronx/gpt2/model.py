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
from transformers_neuronx.gpt2 import hlo
from transformers_neuronx.gpt2.config import GPT2Config


class GPT2ForSampling(module.PretrainedModel):

    def __init__(self, config, batch_size=1, n_active_tokens=1, amp='f32', tp_degree=2,
                 unroll=False, **kwargs):
        super().__init__()
        config = GPT2Config(config, batch_size, amp, tp_degree, **kwargs)
        n_positions = config.n_positions
        self.gpt2_kernel = None
        if unroll:
            block_kernel = None
            ln_lm_head_kernel = None
            self.gpt2_kernel = hlo.build_gpt2_kernel(config, n_active_tokens, n_positions)
        else:
            block_kernel = hlo.build_gpt2_block_kernel(config, n_active_tokens, n_positions)
            ln_lm_head_kernel = hlo.build_lm_head_kernel(config, n_active_tokens)
        self.transformer = GPT2Transformer(config, block_kernel)
        dtype = dtypes.to_torch_dtype(config.amp)
        self.lm_head = module.LowMemoryLazyLinear(config.vocab_size, dtype=dtype)
        self.config = config
        ln_f = self.transformer.ln_f
        self.ln_lm_head = GPT2LnLmHead(config, ln_lm_head_kernel, None, ln_f, self.lm_head)
        self.manipulator = parallel.TensorManipulator(config.tp_degree)
        self.gpt2_caches = None
        self.gpt2_params = None

    def to_neuron(self):
        if self.gpt2_kernel is not None:
            self.gpt2_kernel.load()
        for idx, block in enumerate(self.transformer.h):
            block.to_neuron()
        self.ln_lm_head.to_neuron()

    def reset(self):
        for block in self.transformer.h:
            block.reset()
        self.gpt2_caches = []
        for block in self.transformer.h:
            self.gpt2_caches.append(block.key_cache)
            self.gpt2_caches.append(block.value_cache)
        self.gpt2_params = []
        for block in self.transformer.h:
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
            self.gpt2_params.extend(block_params)
        ln_lm_head = self.ln_lm_head
        ln_lm_head_params = [
            ln_lm_head.ln_f_weight,
            ln_lm_head.ln_f_bias,
            ln_lm_head.lm_head_weight,
        ]
        self.gpt2_params.extend(ln_lm_head_params)

    def forward(self, input_ids, cache_offset, mask):
        inputs_embeds = self.transformer.wte(input_ids)
        past_length = cache_offset.item()
        this_length = input_ids.shape[-1]
        position_ids = torch.arange(past_length, past_length + this_length)
        position_ids = position_ids.unsqueeze(0).view(-1, this_length)
        position_embeds = self.transformer.wpe(position_ids)
        hidden = inputs_embeds + position_embeds
        hidden = hidden.transpose(0, -1)
        dtype = dtypes.to_torch_dtype(self.config.amp)
        hidden = hidden.to(dtype)
        duplicate = self.manipulator.duplicate
        hidden = duplicate(hidden)
        cache_offset = duplicate(cache_offset)
        mask = duplicate(mask)
        if self.gpt2_kernel is None:
            for block in self.transformer.h:
                hidden = block(hidden, cache_offset, mask)
            logits = self.ln_lm_head(hidden)
        else:
            inputs_cores = hidden, cache_offset, mask, *self.gpt2_caches, *self.gpt2_params
            logits, *_ = self.gpt2_kernel(inputs_cores)
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
        inputs = input_ids[:, 0:1]
        tokens = [input_ids]
        # auto-regressive generation
        for cur_len in range(1, max_length):
            print('cur_len=', cur_len)
            offset = cur_len - 1
            seq_len = cur_len
            mask = torch.zeros([1, config.n_positions])
            mask[:, :cur_len] = 1.0

            # forward pass to get next token
            cache_offset = torch.as_tensor([offset], dtype=torch.int32)
            next_token_scores = self(inputs, cache_offset, mask)
            if cur_len < start:
                inputs = input_ids[:, cur_len:cur_len+1]
                continue

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
        return torch.cat(tokens, dim=-1)


class GPT2Transformer(module.LowMemoryModule):

    def __init__(self, config, block_kernel):
        super().__init__()
        self.wte = module.LowMemoryEmbedding(config.vocab_size, config.n_embd)
        self.wpe = module.LowMemoryEmbedding(config.n_ctx, config.n_embd)
        self.h = module.LowMemoryModuleList()
        for _ in range(config.n_layer):
            self.h.append(GPT2Block(config, block_kernel))
        self.ln_f = module.LowMemoryLayerNorm(config.n_embd)


class GPT2Block(module.LowMemoryModule):

    def __init__(self, config, kernel):
        super().__init__()
        self.ln_1 = module.LowMemoryLayerNorm(config.n_embd)
        self.attn = GPT2Attention(config)
        self.ln_2 = module.LowMemoryLayerNorm(config.n_embd)
        self.mlp = GPT2MLP(config)
        self.config = config
        self.kernel = kernel
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
        manipulator = parallel.TensorManipulator(self.config.tp_degree)
        duplicate = manipulator.duplicate
        shard_along = manipulator.shard_along
        primary_only = manipulator.primary_only
        self.ln_1_weight = duplicate(self.ln_1.weight.detach())
        self.ln_1_bias = duplicate(self.ln_1.bias.detach())
        c_attn = self.attn.c_attn
        c_proj = self.attn.c_proj
        c_attn_weight = c_attn.weight.detach()
        c_attn_bias = c_attn.bias.detach()
        n_embd = self.config.n_embd
        self.attn_q_weight = shard_along(c_attn_weight[:, :n_embd], dim=1)
        self.attn_q_bias = shard_along(c_attn_bias[:n_embd], dim=0)
        self.attn_k_weight = shard_along(c_attn_weight[:, n_embd:n_embd*2], dim=1)
        self.attn_k_bias = shard_along(c_attn_bias[n_embd:n_embd*2], dim=0)
        self.attn_v_weight = shard_along(c_attn_weight[:, n_embd*2:n_embd*3], dim=1)
        self.attn_v_bias = shard_along(c_attn_bias[n_embd*2:n_embd*3], dim=0)
        self.attn_out_weight = shard_along(c_proj.weight.detach(), dim=0)
        self.attn_out_bias = primary_only(c_proj.bias.detach())
        c_attn.weight = UninitializedParameter()
        c_proj.weight = UninitializedParameter()
        self.ln_2_weight = duplicate(self.ln_2.weight.detach())
        self.ln_2_bias = duplicate(self.ln_2.bias.detach())
        c_fc = self.mlp.c_fc
        c_proj = self.mlp.c_proj
        self.mlp_in_weight = shard_along(c_fc.weight.detach(), dim=1)
        self.mlp_in_bias = shard_along(c_fc.bias.detach(), dim=0)
        self.mlp_out_weight = shard_along(c_proj.weight.detach(), dim=0)
        self.mlp_out_bias = primary_only(c_proj.bias.detach())
        c_fc.weight = UninitializedParameter()
        c_proj.weight = UninitializedParameter()

    def reset(self):
        config = self.config
        manipulator = parallel.TensorManipulator(config.tp_degree)
        n_positions = config.n_positions
        batch_size = config.batch_size
        n_head = config.n_head
        s_head = config.n_embd // n_head
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


class GPT2Attention(module.LowMemoryModule):

    def __init__(self, config):
        super().__init__()
        n_embd = config.n_embd
        dtype = dtypes.to_torch_dtype(config.amp)
        self.c_attn = module.LowMemoryLazyLinear(n_embd, dtype=dtype)
        self.c_proj = module.LowMemoryLazyLinear(n_embd, dtype=dtype)


class GPT2MLP(module.LowMemoryModule):

    def __init__(self, config):
        super().__init__()
        dtype = dtypes.to_torch_dtype(config.amp)
        self.c_fc = module.LowMemoryLazyLinear(config.n_embd, dtype=dtype)
        self.c_proj = module.LowMemoryLazyLinear(config.intermediate_dim, dtype=dtype)


class GPT2LnLmHead:

    def __init__(self, config, kernel, init_kernel, ln_f, lm_head):
        self.kernel = kernel
        self.init_kernel = init_kernel
        self.ln_f = ln_f
        self.lm_head = lm_head
        self.ln_f_weight = None
        self.ln_f_bias = None
        self.lm_head_weight = None
        tp_degree = config.tp_degree
        self.manipulator = parallel.TensorManipulator(tp_degree)
        vocab_size = config.vocab_size
        self.vocab_pad = ((vocab_size // tp_degree + 1) * tp_degree - vocab_size) % tp_degree

    def to_neuron(self):
        if self.kernel is not None:
            self.kernel.load()
        if self.init_kernel is not None:
            self.init_kernel.load()
        duplicate = self.manipulator.duplicate
        shard_along = self.manipulator.shard_along
        self.ln_f_weight = duplicate(self.ln_f.weight.detach())
        self.ln_f_bias = duplicate(self.ln_f.bias.detach())
        lm_head_weight = self.lm_head.weight.detach()
        lm_head_weight = torch.nn.functional.pad(lm_head_weight, (0, 0, 0, self.vocab_pad))
        self.lm_head_weight = shard_along(lm_head_weight.T, dim=1)
        self.lm_head.weight = UninitializedParameter()

    def __call__(self, hidden):
        inputs = hidden, self.ln_f_weight, self.ln_f_bias, self.lm_head_weight
        logits, = self.kernel(inputs)
        return logits

    def call_init(self, hidden):
        inputs = hidden, self.ln_f_weight, self.ln_f_bias, self.lm_head_weight
        logits, = self.init_kernel(inputs)
        return logits
