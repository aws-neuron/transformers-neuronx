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
from transformers_neuronx.layers.rotary import rotary_embedding


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
        # Bucket sizes below 128 do not provide significant latency benefit and add bucket switching overhead
        self.n_positions_list = utils.power_of_two_bucket_sizes(128, n_positions)

        self.decoder_lm_head = decoder.DecoderLmHeadForSamplingNoEmbedding(
            tp_degree, self.n_positions_list, 1, batch_size, config.attention_head_size, amp,
            config.num_hidden_layers, unroll, neuron_config=neuron_config, allow_pad=True,
        )
        hlo_builder = LlamaForSamplingNoEmbeddingHlo(config, neuron_config=neuron_config)
        self.decoder_lm_head.add_inputs_builder(hlo_builder.inputs)
        self.decoder_lm_head.add_layer_builder(hlo_builder.layer)
        self.decoder_lm_head.add_ln_lm_head_builder(hlo_builder.ln_lm_head)
        self.decoder_lm_head_for_context = None
        head_dim = config.hidden_size // config.num_attention_heads
        position_ids = torch.arange(n_positions)
        positional_embedding = rotary_embedding(head_dim, position_ids)
        self.head_dim = head_dim
        self.positional_embedding = positional_embedding.reshape([-1, head_dim])

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

        """
        Parallel context w/ multiple buckets:

            1. [Default] we choose context length estimates to be half of bucket sizes for output token model
            2. Provided a list of context_length_estimate, a separate KVcache is generated for each bucket
            3. If context_length_estimate <= 0, then parallel context encoding is not used at all
        """
        if self.context_length_estimate is None:
            self.context_length_estimate = [x//2 for x in self.n_positions_list]
            self.context_length_estimate.append(self.n_positions_list[-1])
        elif isinstance(self.context_length_estimate, (list, tuple)):
            self.context_length_estimate = list(self.context_length_estimate)
        elif self.context_length_estimate > 0:
            self.context_length_estimate = [self.context_length_estimate]
        else:
            self.context_length_estimate = None

        if self.context_length_estimate is not None:
            self.decoder_lm_head_for_context = {
                context_length_estimate: self.decoder_lm_head.build_weight_shared(
                    n_positions_list=[context_length_estimate],
                    n_active_tokens=context_length_estimate,
                    unroll=self.context_unroll,
                    share_caches=True,
                )
                for context_length_estimate in self.context_length_estimate}

    def reset(self):
        self.decoder_lm_head.reset()

    def find_context_length_estimate(self, context_length):
        if isinstance(self.context_length_estimate, list):
            for context_length_estimate in self.context_length_estimate:
                best_context_length_estimate = context_length_estimate
                if context_length_estimate >= context_length:
                    break
            else:
                best_context_length_estimate = self.context_length_estimate[-1]
        else:
            best_context_length_estimate = self.context_length_estimate
        return best_context_length_estimate

    def context(self, hidden, cache_ids, start_ids):
        context_length = hidden.shape[1]
        current = 0
        estimate = self.find_context_length_estimate(context_length)

        if estimate is not None:
            hidden_context = hidden
            cache_context = cache_ids

            # Slice context that when it is too large
            if context_length > estimate:
                current = estimate
                hidden_context = hidden[:, :estimate]
                cache_context = cache_ids[:estimate]

            # Cannot use context encoding for a context that is too small. This
            # is because the caller must be aware of the cache-ids/start-ids
            # used.
            elif context_length < estimate:
                current = 0

            # Directly pass input to the context network when exactly sized
            else:
                current = estimate

            if current == estimate:
                decoder_lm_head_for_context = self.decoder_lm_head_for_context[estimate]
                logits = decoder_lm_head_for_context(hidden_context, cache_context, start_ids)

        for i in range(current, context_length):
            cache_ids = torch.as_tensor([i], dtype=torch.int32)
            logits = self.decoder_lm_head(hidden[:, i:i+1], cache_ids, start_ids)

        return logits

    def forward(self, input_ids, cache_ids=None, start_ids=None):

        batch_size, context_length = input_ids.shape
        if start_ids is None:
            start_ids = torch.zeros(batch_size, dtype=torch.int32)
        if cache_ids is None:
            cache_ids = torch.arange(context_length, dtype=torch.int32)

        hidden = self.chkpt_model.model.embed_tokens(input_ids)
        hidden = hidden.transpose(0, -1)

        if context_length > 1:
            logits = self.context(hidden, cache_ids, start_ids)
        else:
            logits = self.decoder_lm_head(hidden, cache_ids, start_ids)

        logits = logits.to(torch.float32)
        logits = logits[:self.config.vocab_size]
        logits = logits.transpose(0, -1)
        logits = logits[:, -1, :]
        return logits

    def sample(self, input_ids, sequence_length, start_ids=None, top_k=50):

        # To enable optimized context encoding network, we must pad
        # up to the context length estimate or we will not correctly
        # select the final context logits (See: layers/transformer.py).
        # This also means we need to shift the start_ids over to correct
        # for padding.
        offset = 0
        batch_size, context_length = input_ids.shape
        estimate = self.find_context_length_estimate(context_length)
        if estimate:
            if context_length < estimate:
                input_ids = utils.pad(input_ids, 1, estimate, left=True)
                offset = estimate - context_length
                if start_ids is None:
                    start_ids = torch.zeros(batch_size, dtype=torch.int32)
                start_ids += offset
                sequence_length += offset
                # Sequence length cannot be greater than n_positions
                sequence_length = min(sequence_length, self.config.n_positions)

        result = sampling.simple_sample(self, input_ids, start_ids, sequence_length,
                                          eos_token_id=self.config.eos_token_id, top_k=top_k)

        if offset != 0:
            result = result[:, offset:]
        return result
