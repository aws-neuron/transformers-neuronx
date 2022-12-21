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
from transformers_neuronx import dtypes
from transformers_neuronx import hlo
from transformers_neuronx import module
from transformers_neuronx import ops
from transformers_neuronx import utils
from transformers_neuronx.decoder import DecoderLmHeadForSamplingNoEmbedding
from transformers_neuronx.gpt2.config import GPT2Config
from transformers_neuronx.opt.model import OPTForSamplingNoEmbeddingHlo
from transformers_neuronx.sampling import simple_sample


class GPT2ForSampling(module.PretrainedModel):

    def __init__(self, config, batch_size=1, amp='f32', tp_degree=2,
                 unroll=None, init_n_active_tokens=None, **kwargs):
        super().__init__()
        config = GPT2Config(config, batch_size, amp, tp_degree, **kwargs)
        self.chkpt_model = GPT2CheckpointCompatible(config)
        self.config = config
        if unroll is None:
            unroll = config.n_layer
        n_positions_list = utils.power_of_two_bucket_sizes(128, config.n_positions)
        attention_head_size = config.n_embd // config.n_head
        self.decoder_lm_head = DecoderLmHeadForSamplingNoEmbedding(
            tp_degree, n_positions_list, 1, batch_size, attention_head_size, amp,
            config.n_layer, unroll,
        )
        hlo_builder = OPTForSamplingNoEmbeddingHlo(tp_degree, batch_size, config.n_embd, 'gelu_new')
        self.decoder_lm_head.add_inputs_builder(hlo_builder.inputs)
        self.decoder_lm_head.add_layer_builder(hlo_builder.layer)
        self.decoder_lm_head.add_ln_lm_head_builder(hlo_builder.ln_lm_head)

    def load_state_dict_dir(self, state_dict_dir):
        self.chkpt_model.load_state_dict_dir(state_dict_dir)

    def load_state_dict_low_memory(self, state_dict):
        self.chkpt_model.load_state_dict_low_memory(state_dict)

    def to_neuron(self):
        ops.init()
        self.chkpt_model.transformer.wte.materialize()
        self.chkpt_model.transformer.wpe.materialize()
        n_embd = self.config.n_embd
        for layer in self.chkpt_model.transformer.h:
            layer.materialize()
            attn = layer.attn
            mlp = layer.mlp
            c_attn_weight = attn.c_attn.weight.detach()
            c_attn_bias = attn.c_attn.bias.detach()
            new_layer = self.decoder_lm_head.new_layer()
            new_layer.add_pre_attention_layer_norm(layer.ln_1.weight.detach(),
                                                   layer.ln_1.bias.detach())
            new_layer.add_attention_query(c_attn_weight[:, :n_embd], c_attn_bias[:n_embd])
            new_layer.add_attention_key(c_attn_weight[:, n_embd:n_embd*2],
                                        c_attn_bias[n_embd:n_embd*2])
            new_layer.add_attention_value(c_attn_weight[:, n_embd*2:n_embd*3],
                                          c_attn_bias[n_embd*2:n_embd*3])
            new_layer.add_attention_output(attn.c_proj.weight.detach(), attn.c_proj.bias.detach())
            new_layer.add_pre_mlp_layer_norm(layer.ln_2.weight.detach(), layer.ln_2.bias.detach())
            new_layer.add_mlp_input(mlp.c_fc.weight.detach(), mlp.c_fc.bias.detach())
            new_layer.add_mlp_output(mlp.c_proj.weight.detach(), mlp.c_proj.bias.detach())
            new_layer.to_neuron()
            layer.nullify()
        ln_f = self.chkpt_model.transformer.ln_f
        ln_f.materialize()
        self.decoder_lm_head.add_final_layer_norm(ln_f.weight.detach(), ln_f.bias.detach())
        lm_head = self.chkpt_model.lm_head
        lm_head.materialize()
        self.decoder_lm_head.add_lm_head(lm_head.weight.detach().T)
        lm_head.nullify()
        self.decoder_lm_head.to_neuron()

    def reset(self):
        self.decoder_lm_head.reset()

    def forward(self, input_ids, position_ids):
        inputs_embeds = self.chkpt_model.transformer.wte(input_ids)
        position_embeds = self.chkpt_model.transformer.wpe(position_ids)
        hidden = inputs_embeds + position_embeds
        hidden = hidden.transpose(0, -1)
        logits = self.decoder_lm_head(hidden, position_ids)
        logits = logits.to(torch.float32)
        logits = logits[:self.config.vocab_size]
        logits = logits.transpose(0, -1)
        logits = logits[:, -1, :]
        return logits

    def sample(self, input_ids, sequence_length):
        return simple_sample(self, input_ids, sequence_length, self.config.n_positions,
                             eos_token_id=self.config.eos_token_id, top_k=50)


class GPT2CheckpointCompatible(module.PretrainedModel):

    def __init__(self, config):
        super().__init__()
        dtype, _, _ = utils.parse_amp(config.amp)
        dtype = dtypes.to_torch_dtype(dtype)
        self.transformer = GPT2Transformer(config)
        self.lm_head = module.LowMemoryLazyLinear(config.vocab_size, dtype=dtype, bias=False)


class GPT2Transformer(module.LowMemoryModule):

    def __init__(self, config):
        super().__init__()
        self.wte = module.LowMemoryEmbedding(config.vocab_size, config.n_embd)
        self.wpe = module.LowMemoryEmbedding(config.n_ctx, config.n_embd)
        self.h = module.LowMemoryModuleList()
        for _ in range(config.n_layer):
            self.h.append(GPT2Block(config))
        self.ln_f = module.LowMemoryLayerNorm(config.n_embd)


class GPT2Block(module.LowMemoryModule):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = module.LowMemoryLayerNorm(config.n_embd)
        self.attn = GPT2Attention(config)
        self.ln_2 = module.LowMemoryLayerNorm(config.n_embd)
        self.mlp = GPT2MLP(config)


class GPT2Attention(module.LowMemoryModule):

    def __init__(self, config):
        super().__init__()
        n_embd = config.n_embd
        dtype, _, _ = utils.parse_amp(config.amp)
        dtype = dtypes.to_torch_dtype(dtype)
        self.c_attn = module.LowMemoryLazyLinear(n_embd, dtype=dtype)
        self.c_proj = module.LowMemoryLazyLinear(n_embd, dtype=dtype)


class GPT2MLP(module.LowMemoryModule):

    def __init__(self, config):
        super().__init__()
        dtype, _, _ = utils.parse_amp(config.amp)
        dtype = dtypes.to_torch_dtype(dtype)
        self.c_fc = module.LowMemoryLazyLinear(config.n_embd, dtype=dtype)
        self.c_proj = module.LowMemoryLazyLinear(config.intermediate_dim, dtype=dtype)
