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
from transformers_neuronx.gptj import hlo
from transformers_neuronx.gptj.config import GPTJConfig
from transformers_neuronx.sampling import simple_sample


class GPTJForSampling(module.PretrainedModel):

    def __init__(self, config, batch_size=1, n_active_tokens=1, amp='f32', tp_degree=2,
                 unroll=False, fast_init=False, **kwargs):
        super().__init__()
        config = GPTJConfig(config, batch_size, n_active_tokens, amp, tp_degree, **kwargs)
        block_kernel = hlo.build_gptj_block_kernel(config)
        self.transformer = GPTJTransformer(config, block_kernel)
        dtype = dtypes.to_torch_dtype(config.amp)
        self.lm_head = module.LowMemoryLazyLinear(config.vocab_size, dtype=dtype)
        self.config = config
        ln_lm_head_kernel = hlo.build_lm_head_kernel(config)
        ln_f = self.transformer.ln_f
        self.ln_lm_head = GPTJLnLmHead(config, ln_lm_head_kernel, ln_f, self.lm_head)
        self.manipulator = parallel.TensorManipulator(config.tp_degree)

    def to_neuron(self):
        self.transformer.wte.materialize()
        for idx, block in enumerate(self.transformer.h):
            block.to_neuron()
        self.ln_lm_head.to_neuron()

    def reset(self):
        for block in self.transformer.h:
            block.reset()

    def forward(self, input_ids, cache_offset, mask):
        last_offset = cache_offset[-1].item()
        inputs_embeds = self.transformer.wte(input_ids)
        hidden = inputs_embeds.transpose(0, -1)
        dtype = dtypes.to_torch_dtype(self.config.amp)
        hidden = hidden.to(dtype)
        config = self.config
        rotary_dim = config.rotary_dim
        s_head = config.n_embd // config.n_head
        pos_embd = self._fixed_pos_embedding(rotary_dim, s_head, offset=last_offset)
        pos_embd = pos_embd.unsqueeze(0)
        pos_embd = pos_embd.to(dtype)
        duplicate = self.manipulator.duplicate
        hidden = duplicate(hidden)
        pos_embd = duplicate(pos_embd)
        cache_offset = duplicate(cache_offset)
        mask = duplicate(mask)
        for block in self.transformer.h:
            hidden = block(hidden, pos_embd, cache_offset, mask)
        logits = self.ln_lm_head(hidden)
        logits = self.manipulator.unshard_along(logits, dim=0)
        logits = logits.to(torch.float32)
        logits = logits.transpose(0, -1)
        logits = logits[:, -1, :]
        return logits

    def sample(self, input_ids, sequence_length):
        return simple_sample(self, input_ids, sequence_length, self.config.n_positions,
                             eos_token_id=self.config.eos_token_id, top_k=50)

    def _fixed_pos_embedding(self, dim, head_dim, offset):
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


class GPTJTransformer(module.LowMemoryModule):

    def __init__(self, config, block_kernel):
        super().__init__()
        self.wte = module.LowMemoryEmbedding(config.vocab_size, config.n_embd)
        self.h = module.LowMemoryModuleList()
        for _ in range(config.n_layer):
            block = GPTJBlock(config, block_kernel)
            self.h.append(block)
        self.ln_f = module.LowMemoryLayerNorm(config.n_embd)


class GPTJBlock(module.LowMemoryModule):

    def __init__(self, config, kernel):
        super().__init__()
        self.ln_1 = module.LowMemoryLayerNorm(config.n_embd)
        self.attn = GPTJAttention(config)
        self.mlp = GPTJMLP(config)
        self.config = config
        self.kernel = kernel
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
        self.value_cache = None

    def to_neuron(self):
        if self.kernel is not None:
            self.kernel.load()
        manipulator = parallel.TensorManipulator(self.config.tp_degree)
        duplicate = manipulator.duplicate
        shard_along = manipulator.shard_along
        primary_only = manipulator.primary_only
        self.ln_1.materialize()
        self.ln_1_weight = duplicate(self.ln_1.weight.detach())
        self.ln_1_bias = duplicate(self.ln_1.bias.detach())
        attn = self.attn
        attn.q_proj.materialize()
        attn.k_proj.materialize()
        attn.v_proj.materialize()
        attn.out_proj.materialize()
        self.attn_q_weight = shard_along(attn.q_proj.weight.detach().T, dim=1)
        self.attn_k_weight = shard_along(attn.k_proj.weight.detach().T, dim=1)
        self.attn_v_weight = shard_along(attn.v_proj.weight.detach().T, dim=1)
        self.attn_out_weight = shard_along(attn.out_proj.weight.detach().T, dim=0)
        attn.q_proj.weight = UninitializedParameter()
        attn.k_proj.weight = UninitializedParameter()
        attn.v_proj.weight = UninitializedParameter()
        attn.out_proj.weight = UninitializedParameter()
        mlp = self.mlp
        mlp.fc_in.materialize()
        mlp.fc_out.materialize()
        self.mlp_in_weight = shard_along(mlp.fc_in.weight.detach().T, dim=1)
        self.mlp_in_bias = shard_along(mlp.fc_in.bias.detach(), dim=0)
        self.mlp_out_weight = shard_along(mlp.fc_out.weight.detach().T, dim=0)
        self.mlp_out_bias = primary_only(mlp.fc_out.bias.detach())
        mlp.fc_in.weight = UninitializedParameter()
        mlp.fc_out.weight = UninitializedParameter()

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

    def forward(self, hidden, pos_embd, cache_offset, mask):
        inputs_cores = \
            hidden, self.ln_1_weight, self.ln_1_bias, \
            self.attn_q_weight, self.attn_k_weight, self.attn_v_weight, self.attn_out_weight, \
            pos_embd, self.key_cache, self.value_cache, cache_offset, mask, \
            self.mlp_in_weight, self.mlp_in_bias, self.mlp_out_weight, self.mlp_out_bias
        hidden, self.key_cache, self.value_cache = self.kernel(inputs_cores)
        return hidden


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

    def __init__(self, config, kernel, ln_f, lm_head):
        self.kernel = kernel
        self.ln_f = ln_f
        self.lm_head = lm_head
        self.ln_f_weight = None
        self.ln_f_bias = None
        self.lm_head_weight = None
        self.lm_head_bias = None
        self.manipulator = parallel.TensorManipulator(config.tp_degree)

    def to_neuron(self):
        if self.kernel is not None:
            self.kernel.load()
        duplicate = self.manipulator.duplicate
        shard_along = self.manipulator.shard_along
        self.ln_f.materialize()
        self.ln_f_weight = duplicate(self.ln_f.weight.detach())
        self.ln_f_bias = duplicate(self.ln_f.bias.detach())
        self.lm_head.materialize()
        self.lm_head_weight = shard_along(self.lm_head.weight.detach().T, dim=1)
        self.lm_head_bias = shard_along(self.lm_head.bias.detach(), dim=0)
        self.lm_head.weight = UninitializedParameter()

    def __call__(self, hidden):
        inputs = hidden, self.ln_f_weight, self.ln_f_bias, self.lm_head_weight, self.lm_head_bias
        logits, = self.kernel(inputs)
        return logits
