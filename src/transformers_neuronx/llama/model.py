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

from transformers_neuronx import decoder
from transformers_neuronx import module
from transformers_neuronx import ops
from transformers_neuronx import sampling
from transformers_neuronx import utils
from transformers_neuronx.llama.config import LlamaConfig
from transformers_neuronx.llama.modules import LlamaForCausalLM
from transformers_neuronx.llama.hlo import LlamaForSamplingNoEmbeddingHlo


class LlamaForSampling(module.WrappingCheckpointCompatibleModel):

    def __init__(self, config, *, n_positions=2048, batch_size=1, amp='f32', tp_degree=2,
                 context_length_estimate=None, context_unroll=None, unroll=None,
                 neuron_config=None, **kwargs):
        config = LlamaConfig(config, n_positions, batch_size, amp, tp_degree)
        super().__init__(LlamaForCausalLM, config)
        self.config = config
        self.neuron_config =  neuron_config

        self.context_length_estimate = context_length_estimate
        if context_unroll is None:
            context_unroll = config.num_hidden_layers
        self.context_unroll = context_unroll

        if unroll is None:
            unroll = config.num_hidden_layers
        self.n_positions_list = utils.power_of_two_bucket_sizes(32, n_positions)

        self.decoder_lm_head = decoder.DecoderLmHeadForSamplingNoEmbedding(
            tp_degree, self.n_positions_list, 1, batch_size, config.attention_head_size, amp,
            config.num_hidden_layers, unroll, neuron_config=neuron_config, allow_pad=True,
        )
        hlo_builder = LlamaForSamplingNoEmbeddingHlo(config, neuron_config=neuron_config)
        self.decoder_lm_head.add_inputs_builder(hlo_builder.inputs)
        self.decoder_lm_head.add_layer_builder(hlo_builder.layer)
        self.decoder_lm_head.add_ln_lm_head_builder(hlo_builder.ln_lm_head)
        self.decoder_lm_head_for_context = None

    def to_neuron(self):

        # Materialize the embedding to CPU
        self.chkpt_model.model.embed_tokens.materialize()

        ops.init()

        for layer in self.chkpt_model.model.layers:
            layer.materialize()
            attn = layer.self_attn
            mlp = layer.mlp
            new_layer = self.decoder_lm_head.new_layer()
            new_layer.add_pre_attention_layer_norm(layer.input_layernorm.weight.detach(), None)
            new_layer.add_attention_query(attn.q_proj.weight.detach().T, None)
            new_layer.add_attention_key(attn.k_proj.weight.detach().T, None)
            new_layer.add_attention_value(attn.v_proj.weight.detach().T, None)
            new_layer.add_attention_output(attn.o_proj.weight.detach().T, None)
            new_layer.add_pre_mlp_layer_norm(layer.post_attention_layernorm.weight.detach(), None)

            # Note: Automatic MLP padding is safe since zeros are *only* introduced to intermediary state
            new_layer.add_parameter(mlp.gate_proj.weight.T, sharding=1, allow_pad=True)
            new_layer.add_parameter(mlp.up_proj.weight.T, sharding=1, allow_pad=True)
            new_layer.add_parameter(mlp.down_proj.weight.T, sharding=0, allow_pad=True)

            new_layer.to_neuron()
            layer.nullify()

        ln_f = self.chkpt_model.model.norm
        ln_f.materialize()
        self.decoder_lm_head.add_final_layer_norm(ln_f.weight.detach(), None)

        lm_head = self.chkpt_model.lm_head
        lm_head.materialize()
        self.decoder_lm_head.add_lm_head(lm_head.weight.detach().T)
        lm_head.nullify()
        self.decoder_lm_head.to_neuron()

        if self.context_length_estimate is not None:
            self.decoder_lm_head_for_context = self.decoder_lm_head.build_weight_shared(
                n_positions_list=[self.context_length_estimate],
                n_active_tokens=self.context_length_estimate,
                unroll=self.context_unroll,
                share_caches=True,
            )

    def reset(self):
        self.decoder_lm_head.reset()

    def _forward(self, decoder_lm_head, input_ids, cache_ids, start_ids=None):

        # TODO: Move embedding and rotary embedding to Neuron
        hidden = self.chkpt_model.model.embed_tokens(input_ids)
        head_dim = self.config.hidden_size // self.config.num_attention_heads
        pos_embd = rotary_embedding(head_dim, head_dim, cache_ids)

        start_ids = torch.zeros([self.config.batch_size], dtype=torch.int32)
        hidden = hidden.transpose(0, -1)
        logits = decoder_lm_head(hidden, pos_embd, cache_ids, start_ids)
        logits = logits.to(torch.float32)
        logits = logits[:self.config.vocab_size]
        logits = logits.transpose(0, -1)
        logits = logits[:, -1, :]
        return logits

    def forward_for_context(self, input_ids, cache_ids, start_ids=None):
        return self._forward(self.decoder_lm_head_for_context, input_ids, cache_ids, start_ids)

    def forward(self, input_ids, cache_ids, start_ids=None):
        return self._forward(self.decoder_lm_head, input_ids, cache_ids, start_ids)

    def sample(self, input_ids, sequence_length, start_ids=None, top_k=50):
        if self.context_length_estimate is None:
            return sampling.simple_sample(self, input_ids, start_ids, sequence_length,
                                          eos_token_id=self.config.eos_token_id, top_k=top_k)

        _, start = input_ids.shape
        context_length = self.context_length_estimate
        cache_ids = torch.arange(context_length, dtype=torch.int32)
        input_context = input_ids[:, :context_length]
        if start < context_length:
            input_pad = context_length - start
            input_context = torch.nn.functional.pad(input_context, (0, input_pad, 0, 0))
        next_token_scores = self.forward_for_context(input_context, cache_ids, start_ids)
        for cur_len in range(context_length, start):
            cache_ids = torch.as_tensor([cur_len], dtype=torch.int32)
            next_token_scores = self(input_ids[:, cur_len:cur_len+1], cache_ids, start_ids)
        return sampling.sample_loop(
            self, input_ids, start_ids, next_token_scores, sequence_length,
            eos_token_id=self.config.eos_token_id, top_k=top_k)



def rotary_embedding(dim, head_dim, cache_ids, base=10000):

    # TODO: Support start_id masking for left padded sequences

    embs = []

    for offset in cache_ids:
        seq_len = offset + 1
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2) / dim))
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
        embs.append(pos_embd)

    return torch.stack(embs, dim=0)
