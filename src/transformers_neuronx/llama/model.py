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
import os
from transformers_neuronx import decoder
from transformers_neuronx import module
from transformers_neuronx import ops
from transformers_neuronx import sampling
from transformers_neuronx import utils
from transformers_neuronx import bucket
from transformers_neuronx import base
from transformers_neuronx.llama.config import LlamaConfig
from transformers_neuronx.llama.modules import LlamaForCausalLM
from transformers_neuronx.llama.hlo import LlamaForSamplingNoEmbeddingHlo


class LlamaForSampling(module.WrappingCheckpointCompatibleModel, base.NeuronModelBase):

    def __init__(self, config, *, n_positions=2048, batch_size=1, amp='f32', tp_degree=2,
                 context_length_estimate=None, context_unroll=None, unroll=None,
                 neuron_config=None, prefixed_length=0, **kwargs):
        config = LlamaConfig(config, n_positions, batch_size, amp, tp_degree)
        super().__init__(LlamaForCausalLM, config)
        self.config = config
        self.neuron_config = neuron_config
        self.prefixed_length = prefixed_length
        if context_unroll is None:
            context_unroll = config.num_hidden_layers
        self.context_unroll = context_unroll

        if unroll is None:
            unroll = config.num_hidden_layers

        self.token_buckets = bucket.token_sizes(n_positions)
        self.context_buckets = bucket.context_sizes(context_length_estimate, self.token_buckets)
        if prefixed_length:
            if prefixed_length not in self.context_buckets:
                self.context_buckets.append(prefixed_length)
                self.context_buckets = sorted(self.context_buckets)
                
        self.max_positions = self.token_buckets[-1]

        self.decoder_lm_head = decoder.DecoderLmHeadForSamplingNoEmbedding(
            tp_degree=tp_degree, n_positions_list=self.token_buckets, n_active_tokens=1, batch_size=batch_size,
            attention_head_size=config.attention_head_size, amp=amp,
            num_layers=config.num_hidden_layers, n_head=config.num_attention_heads, n_kv_head=config.num_key_value_heads,
            unroll=unroll, neuron_config=neuron_config, allow_pad=True, shard_over_batch=config.shard_over_batch
        )
        self.register_for_serialization(self.decoder_lm_head)
        hlo_builder = LlamaForSamplingNoEmbeddingHlo(config, neuron_config=neuron_config)
        self.decoder_lm_head.add_inputs_builder(hlo_builder.inputs)
        self.decoder_lm_head.add_layer_builder(hlo_builder.layer)
        self.decoder_lm_head.add_ln_lm_head_builder(hlo_builder.ln_lm_head)
        if self.context_buckets:
            self.decoder_lm_head_for_context = {}
            for context_length_estimate in self.context_buckets:
                self.decoder_lm_head_for_context[context_length_estimate] = decoder.DecoderLmHeadForSamplingNoEmbedding(
                    tp_degree, 
                    [context_length_estimate], 
                    context_length_estimate, 
                    batch_size, 
                    config.attention_head_size, 
                    amp=amp,
                    num_layers=config.num_hidden_layers,
                    n_head=config.num_attention_heads,
                    n_kv_head=config.num_key_value_heads,
                    unroll=context_unroll,
                    neuron_config=neuron_config, 
                    allow_pad=self.decoder_lm_head.allow_pad
                )
                self.register_for_serialization(self.decoder_lm_head_for_context[context_length_estimate])

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
            new_layer.add_attention_output(attn.o_proj.weight.detach(), None, sharding=1, transposed=False)
            new_layer.add_pre_mlp_layer_norm(layer.post_attention_layernorm.weight.detach(), None)

            # Note: Automatic MLP padding is safe since zeros are *only* introduced to intermediary state
            new_layer.add_parameter(mlp.gate_proj.weight.T, sharding=1, allow_pad=True, allow_quantize=True)
            new_layer.add_parameter(mlp.up_proj.weight.T, sharding=1, allow_pad=True, allow_quantize=True)
            new_layer.add_parameter(mlp.down_proj.weight, sharding=1, allow_pad=True, allow_quantize=True, out_feature_dim=0)

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
        self.decoder_lm_head.enable_executor()

        if self.context_buckets:
            for context_length_estimate in self.context_buckets:
                model = self.decoder_lm_head.build_weight_shared(share_caches=True, new=self.decoder_lm_head_for_context[context_length_estimate])
                # PERF: No latency improvement seen in multi-layer models from executor
                if self.context_unroll == self.config.num_hidden_layers:
                    model.enable_executor()
                self.decoder_lm_head_for_context[context_length_estimate] = model

    def reset(self):
        self.decoder_lm_head.reset()

    def context(self, hidden, cache_ids, start_ids):
        context_length = hidden.shape[1]
        current = 0
        estimate = bucket.find(self.context_buckets, context_length)

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
                model = self.decoder_lm_head_for_context[estimate]
                logits = model(hidden_context, cache_context, start_ids)

        for i in range(current, context_length):
            cache_ids = torch.as_tensor([i], dtype=torch.int32)
            hidden_slice = hidden[:, i:i+1].contiguous()
            logits = self.decoder_lm_head(hidden_slice, cache_ids, start_ids)

        return logits

    def set_prefixed(self, input_ids):
        self.prefixed_input_ids = input_ids[:, :self.prefixed_length]
        prefixed_length = self.prefixed_length
        self.prefixed_length = 0
        self.forward(self.prefixed_input_ids)
        self.prefixed_length = prefixed_length

    def forward(self, input_ids, cache_ids=None, start_ids=None):

        batch_size, context_length = input_ids.shape
        if start_ids is None:
            start_ids = torch.zeros(batch_size, dtype=torch.int32)
        if cache_ids is None:
            cache_ids = torch.arange(context_length, dtype=torch.int32)
        if self.prefixed_length:
            cache_ids += self.prefixed_length

        hidden = self.chkpt_model.model.embed_tokens(input_ids)
        hidden = hidden.transpose(0, -1).contiguous()

        if context_length > 1:
            logits = self.context(hidden, cache_ids, start_ids)
        else:
            logits = self.decoder_lm_head(hidden, cache_ids, start_ids)

        logits = logits.to(torch.float32)
        logits = logits[:self.config.vocab_size, -1, :]
        logits = logits.transpose(0, 1)
        return logits

    def sample(self, input_ids, sequence_length, start_ids=None,
               top_k=50, top_p=1.0, eos_token_override=None, temperature=1.0, streamer=None):

        # To enable optimized context encoding network, we must pad
        # up to the context length estimate or we will not correctly
        # select the final context logits (See: layers/transformer.py).
        # This also means we need to shift the start_ids over to correct
        # for padding.
        offset = 0
        batch_size, context_length = input_ids.shape
        prefixed_length = self.prefixed_length
        if context_length < prefixed_length:
            self.prefixed_length = 0
        else:
            input_ids = input_ids[:, prefixed_length:]
            context_length -= prefixed_length
            sequence_length -= prefixed_length
        estimate = bucket.find(self.context_buckets, context_length)
        if estimate:
            if context_length < estimate:
                input_ids = utils.pad(input_ids, 1, estimate, left=True)
                offset = estimate - context_length
                if not prefixed_length:
                    if start_ids is None:
                        start_ids = torch.zeros(batch_size, dtype=torch.int32)
                    start_ids += offset
                sequence_length += offset
                # Sequence length cannot be greater than n_positions
                sequence_length = min(sequence_length, self.max_positions)

        result = sampling.sample_llama(
            self, input_ids, start_ids, sequence_length,
            eos_token_id=self.config.eos_token_id if eos_token_override is None else eos_token_override,
            top_k=top_k, top_p=top_p, temperature=temperature, streamer=streamer
        )

        if offset != 0:
            result = result[:, offset:]
        return result

class FIDLlamaForSampling(LlamaForSampling):

    def __init__(self, config, *, n_positions=2048, batch_size=1, amp='f32', tp_degree=2,
                 context_length_estimate=None, context_unroll=None, unroll=None,
                 neuron_config=None, reorder_cache=False, **kwargs):
        # Force batch_size=1 in NEFF
        super().__init__(config, n_positions=n_positions, batch_size=1, amp=amp,
                        tp_degree=tp_degree, context_length_estimate=context_length_estimate,
                        context_unroll=context_unroll, unroll=unroll, neuron_config=neuron_config,
                        reorder_cache=False, **kwargs)
        self.batch_size = batch_size
        self.bos_token_id = self.config.bos_token_id

    def context(self, hidden, cache_ids, start_ids):
        # Fusion-In-Decoder context encoding
        fused_context_length = hidden.shape[1]
        context_length = fused_context_length // self.batch_size

        current = 0
        estimate = bucket.find(self.context_buckets, context_length)

        if estimate is not None:
            hidden_context = hidden
            cache_context = cache_ids

            # Slice context when it is too large
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
                model = self.decoder_lm_head_for_context[estimate]

                # Run each context separately in-place
                for j in range(self.batch_size):
                    single_context_slice = slice(j * context_length, (j+1) * context_length)
                    logits = model(hidden_context[:, single_context_slice, :], cache_context[single_context_slice], start_ids)


        for i in range(current, context_length):
            cache_ids = torch.as_tensor([i], dtype=torch.int32)
            hidden_slice = hidden[:, i:i+1].contiguous()
            logits = self.decoder_lm_head(hidden_slice, cache_ids, start_ids)

        logits[:] = float('-inf')
        logits[self.bos_token_id] = 1.0
        return logits

    def sample(self, input_ids, sequence_length, start_ids=None, top_k=50, streamer=None):
        """ Sample function
        input_ids: shape [batch_size, context_length]

        input_ids of different batch index represent single (context + query).
        They will be mixed and generate a single output sequence.
        """

        # In FID-Llama, first, context encoding is done w/ generating any output token for context
        # Here batch-size are different context+queries of single run

        offset = 0
        fused_batch_size = 1
        batch_size, context_length = input_ids.shape

        # The context length estimate is chosen based on single (context+query)
        estimate = bucket.find(self.context_buckets, context_length)

        if batch_size * context_length >= sequence_length:
            raise ValueError(f"sequence_length [{sequence_length}] should be larger than fused input contexts [{context_length} x {batch_size}]")
        if batch_size * estimate >= sequence_length:
            raise ValueError(f"sequence_length [{sequence_length}] should be larger than fused input context estimates [{estimate} x {batch_size}]")

        if estimate:
            if context_length < estimate:
                input_ids = utils.pad(input_ids, 1, estimate, left=True)
                offset = estimate - context_length
                if start_ids is None:
                    start_ids = torch.zeros(fused_batch_size, dtype=torch.int32)
                start_ids += offset
                sequence_length += offset
                # Sequence length cannot be greater than n_positions
                sequence_length = min(sequence_length, self.max_positions)
                context_length = estimate

        # Flatten input_ids
        context_length = batch_size * context_length
        input_ids = input_ids.reshape(fused_batch_size, context_length)

        # Run the model
        result = sampling.sample_llama(self, input_ids, start_ids, sequence_length,
                                          eos_token_id=self.config.eos_token_id, top_k=top_k, streamer=streamer)

        if offset != 0:
            # Offset by offset * batch_size
            result = result[:, context_length:]
        return result
