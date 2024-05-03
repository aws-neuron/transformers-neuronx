# Copyright Amazon Web Services and its Affiliates. All Rights Reserved.

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
import os
import warnings
import torch
from transformers import PreTrainedModel
from transformers.utils import ModelOutput
from transformers_neuronx import decoder
from transformers_neuronx import dtypes
from transformers_neuronx import module
from transformers_neuronx import ops
from transformers_neuronx import parallel
from transformers_neuronx import sampling
from transformers_neuronx import utils
from transformers_neuronx import bucket
from transformers_neuronx import tensor_pool
from transformers_neuronx import base
from transformers_neuronx.constants import LAYOUT_BSH, LAYOUT_HSB
from transformers_neuronx.config import NeuronConfig
from transformers_neuronx.gpt2.config import GPT2Config, GPT2HuggingFaceConfig
from transformers_neuronx.opt.model import OPTForSamplingNoEmbeddingHlo
from transformers_neuronx.generation_utils import HuggingFaceGenerationModelAdapter

class GPT2ForSampling(base.NeuronModelBase):

    def __init__(self, config, batch_size=1, amp='f32', tp_degree=2,
                 unroll=None, init_n_active_tokens=None, neuron_config=None,
                 tag=None, **kwargs):
        config = GPT2Config(config, batch_size, amp, tp_degree, **kwargs)
        super().__init__(GPT2CheckpointCompatible, config)
        self.config = config
        self.neuron_config = neuron_config if neuron_config else NeuronConfig()
        if self.neuron_config.on_device_generation:
            self.neuron_config.on_device_generation.vocab_size = self.config.vocab_size
        # Check if input sequence length is allowed given position embedding dimensions
        sequence_length = kwargs.get("n_positions", None)
        if sequence_length:
            max_allowed_sequence_length = config.max_position_embeddings
            if sequence_length > max_allowed_sequence_length:
                raise ValueError(f"Sequence length ({sequence_length}) cannot be larger than position embedding's context size ({max_allowed_sequence_length})!")
        if unroll is None:
            unroll = config.n_layer
        self.token_buckets = bucket.token_sizes(config.n_positions)
        attention_head_size = config.n_embd // config.n_head

        start_mask = os.environ.get('NEURON_INTERNAL_ASSUME_ALL_PROMPT_LENGTHS_ARE_EQUAL', None) != '1'
        hlo_builder = OPTForSamplingNoEmbeddingHlo(tp_degree, config.n_embd, 'gelu_new',
                                                   self.config.eos_token_id, amp, start_mask,
                                                   neuron_config=self.neuron_config)

        self.decoder_param_set = decoder.DecoderLmHeadForSamplingNoEmbedding(
            tp_degree, self.token_buckets, 1, batch_size, attention_head_size, amp=amp,
            num_layers=config.n_layer, n_head=config.n_head, n_kv_head=config.n_kv_head,
            unroll=unroll, neuron_config=self.neuron_config, tag=tag, builder=hlo_builder,
        )
        self.decoder_lm_head = self.decoder_param_set.init_token_decoder(unroll=unroll, buckets=self.token_buckets, model_obj=self)

        # Token counter for sliding window attention
        self.num_processed_tokens = 0
        self.tag = tag

    def load_weights(self):
        ops.init()
        self.materialize_embeddings()
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
            _, n_embd_qkv = c_attn_weight.shape
            if n_embd_qkv < n_embd * 3:
                n_group = (n_embd_qkv - n_embd) // 2
                new_layer.add_attention_key(c_attn_weight[:, n_embd:n_embd+n_group],
                                            c_attn_bias[n_embd:n_embd+n_group])
                new_layer.add_attention_value(c_attn_weight[:, n_embd+n_group:],
                                              c_attn_bias[n_embd+n_group:])
            else:
                new_layer.add_attention_key(c_attn_weight[:, n_embd:n_embd*2],
                                            c_attn_bias[n_embd:n_embd*2])
                new_layer.add_attention_value(c_attn_weight[:, n_embd*2:n_embd*3],
                                              c_attn_bias[n_embd*2:n_embd*3])
            is_bsh = self.neuron_config and self.neuron_config.attention_layout == LAYOUT_BSH
            new_layer.add_attention_output(attn.c_proj.weight.detach().T, attn.c_proj.bias.detach(), sharding=1, transposed=False)

            new_layer.add_pre_mlp_layer_norm(layer.ln_2.weight.detach(), layer.ln_2.bias.detach())
            new_layer.add_mlp_input(mlp.c_fc.weight.detach(), mlp.c_fc.bias.detach())
            if self.neuron_config.weight_tiling:
                new_layer.add_mlp_output(mlp.c_proj.weight.detach(), mlp.c_proj.bias.detach())
            else:
                if is_bsh:
                    new_layer.add_mlp_output(
                        mlp.c_proj.weight.detach(),
                        mlp.c_proj.bias.detach(),
                        sharding=0,
                        transposed=True,
                    )
                else:
                    new_layer.add_mlp_output(
                        mlp.c_proj.weight.detach().T,
                        mlp.c_proj.bias.detach(),
                        sharding=1,
                        transposed=False,
                    )
            new_layer.to_neuron()
            layer.nullify()
        ln_f = self.chkpt_model.transformer.ln_f
        ln_f.materialize()
        self.decoder_lm_head.add_final_layer_norm(ln_f.weight.detach(), ln_f.bias.detach())
        lm_head = self.chkpt_model.lm_head
        lm_head.materialize()
        if self.neuron_config.on_device_embedding:
            self.decoder_lm_head.add_pre_layer_parameter(self.chkpt_model.transformer.wte.weight, sharding=1, allow_pad=True)
            self.decoder_lm_head.add_pre_layer_parameter(self.chkpt_model.transformer.wpe.weight, sharding=1, allow_pad=True)
        self.decoder_lm_head.add_lm_head(lm_head.weight.detach().T)
        lm_head.nullify()
        self.decoder_lm_head.to_neuron()
        self.init_rest_of_model()

    def materialize_embeddings(self):
        self.chkpt_model.transformer.wte.materialize()
        self.chkpt_model.transformer.wpe.materialize()
    
    def init_rest_of_model(self):
        # We need to reset once, since there might be NaN initially in KVcache.
        # This is done right after weight loading which is shared for different generation methods.
        self.reset()

    def reset(self):
        self.decoder_lm_head.reset()
        self.num_processed_tokens = 0

    def forward(self, input_ids, cache_ids, start_ids=None):
        batch_size, context_length = input_ids.shape
        if cache_ids is None:
            cache_ids = torch.arange(context_length, dtype=torch.int32)
        # Check if sliding window attention is applied
        sliding_window_attn_enabled = self.neuron_config and \
                                      self.neuron_config.sparse_attn and \
                                      self.neuron_config.sparse_attn.skip_masking_decode
        # Compute the window starting index for specific mask patterns
        # For other patterns we pass in a default value of 0, it won't be used
        curr_window_start = \
            self.num_processed_tokens - self.neuron_config.sparse_attn.sparse_attn_config.window_size \
            if sliding_window_attn_enabled else 0
        curr_window_start = torch.as_tensor([curr_window_start], dtype=torch.int32)
        last_token_id = torch.as_tensor([0], dtype=torch.int32)

        if not self.neuron_config.on_device_embedding:
            input_ids = self.chkpt_model.transformer.wte(input_ids)
            position_ids, start_ids = self.decoder_lm_head.embed_positions_ids(cache_ids, start_ids)
            position_embeds = self.chkpt_model.transformer.wpe(position_ids)
            input_ids = input_ids + position_embeds
            if self.neuron_config.attention_layout == LAYOUT_HSB:
                input_ids = input_ids.transpose(0, 2).contiguous()
        logits = self.decoder_lm_head(input_ids, cache_ids, start_ids, last_token_id, curr_window_start)

        if self.neuron_config.on_device_generation:
            return logits

        logits = self._cast_logits(logits)
        logits = logits[:self.config.vocab_size, -1, :]
        logits = logits.transpose(0, 1)
        # The model always runs in decode mode
        self.num_processed_tokens += 1
        return logits

    def sample(self, input_ids, sequence_length, start_ids=None, top_k=50, streamer=None):
        if self.neuron_config.on_device_generation:
            return sampling.sample_tokens(self, input_ids, start_ids, sequence_length=sequence_length,
                                          config=self.neuron_config.on_device_generation, streamer=streamer)
        else:
            return sampling.simple_sample(self, input_ids, start_ids, sequence_length,
                                      eos_token_id=self.config.eos_token_id, top_k=top_k, streamer=streamer)

    def beam_search(self, input_ids, num_beams, sequence_length, start_ids=None):
        batch_size, start = input_ids.shape
        bn = batch_size * num_beams
        b_n_input_ids = input_ids.unsqueeze(1).repeat(1, num_beams, 1)
        bn_input_ids = b_n_input_ids.reshape([bn, start])
        if start_ids is not None:
            start_ids = start_ids.unsqueeze(1).repeat(1, num_beams).reshape([bn])
        cache_ids = torch.arange(start, dtype=torch.int32)
        next_token_scores = self(bn_input_ids, cache_ids, start_ids)  # [bn, v]
        next_token_scores = next_token_scores[::num_beams, ...]  # [b, v]
        total_scores, topk_indices = torch.topk(next_token_scores, num_beams)  # [b, n]
        total_scores = total_scores.reshape([batch_size, num_beams, 1])
        inputs = topk_indices.reshape([batch_size, num_beams, 1])
        tokens = [b_n_input_ids, inputs]
        for cur_len in range(start, sequence_length):
            next_len = cur_len + 1
            inputs = inputs.reshape([bn, 1])
            cache_ids = torch.as_tensor([cur_len], dtype=torch.int32)
            next_token_scores = self(inputs, cache_ids, start_ids)
            total_scores = total_scores + next_token_scores.reshape([batch_size, num_beams, -1])
            scores = total_scores.reshape([batch_size, -1])  # [b, n*v]
            topk_scores, topk_indices_in_nv = torch.topk(scores, num_beams)  # [b, n]
            topk_indices_in_nv = topk_indices_in_nv.reshape([batch_size, num_beams, 1])
            inputs = topk_indices_in_nv % self.config.vocab_size
            total_scores = total_scores.gather(dim=2, index=inputs)
            tokens.append(inputs)
        return torch.cat(tokens, dim=-1)


class GPT2ForHuggingFaceSampling(GPT2ForSampling):

    def __init__(self, config, *args, **kwargs):
        warnings.warn("GPT2ForHuggingFaceSampling class is deprecated. It now falls back to use GPT2ForSampling. "
            "Please use HuggingFaceGenerationModelAdapter for generation API.")
        super().__init__(config, *args, **kwargs)
        self.wrapper = HuggingFaceGenerationModelAdapter(config, self)

    def generate(self, *args, **kwargs):
        return self.wrapper.generate(*args, **kwargs)

    def reset_generation(self):
        self.wrapper.reset_generation()


class GPT2ForSamplingWithContextBroadcasting(base.NeuronModelBase):

    def __init__(self, config, batch_size=1, prompt_batch_size=1, amp='f32', tp_degree=2,
                 unroll=None, context_length_estimate=None, context_unroll=1, neuron_config=NeuronConfig(), reorder_cache=False, **kwargs):
        config = GPT2Config(config, batch_size, amp, tp_degree, **kwargs)
        super().__init__(GPT2CheckpointCompatible, config)
        self.config = config
        self.neuron_config = neuron_config if neuron_config else NeuronConfig()
        if self.neuron_config.on_device_generation:
            self.neuron_config.on_device_generation.vocab_size = self.config.vocab_size
        if unroll is None:
            unroll = config.n_layer
        self.unroll=unroll
        self.prompt_batch_size = prompt_batch_size
        attention_head_size = config.n_embd // config.n_head
        assert context_length_estimate != 0, (
            "context_length_estimate cannot be set to 0 in the "
            "GPT2ForSamplingWithContextBroadcasting class."
        )
        if context_length_estimate is None:
            context_length_estimate = config.n_positions
        self.context_unroll = context_unroll
        self.token_buckets = bucket.token_sizes(config.n_positions)
        self.context_buckets = bucket.context_sizes(
            context_length_estimate, self.token_buckets
        )
        self.max_positions=self.token_buckets[-1]
        self.return_ranks = -1 if not self.neuron_config.on_device_generation else 1

        # TODO: the start_mask needs to be True with left padding for context estimate,
        # need to fix this after having right padding
        start_mask = True

        hlo_builder = OPTForSamplingNoEmbeddingHlo(tp_degree, config.n_embd, 'gelu_new',
                                                   eos_token_id=self.config.eos_token_id,
                                                   amp=amp,
                                                   start_mask=start_mask,
                                                   neuron_config=self.neuron_config)

        self.decoder_param_set = decoder.DecoderLmHeadForSamplingNoEmbedding(
            tp_degree=tp_degree, n_positions_list=self.token_buckets, n_active_tokens=1, batch_size=batch_size,
            attention_head_size=attention_head_size, amp=amp,
            num_layers=config.n_layer, n_head=config.n_head, n_kv_head=config.n_kv_head,
            unroll=unroll, neuron_config=self.neuron_config, builder=hlo_builder,
            prompt_batch_size=self.prompt_batch_size
        )

        self.decoder_lm_head_for_context= self.decoder_param_set.init_context_decoder(unroll=self.context_unroll, buckets=self.context_buckets,
                                                                                      model_obj=self)
        self.decoder_lm_head=self.decoder_param_set.init_token_decoder(unroll=self.unroll, buckets=self.token_buckets, model_obj=self)

        self.decoder_lm_head.need_reorder_cache = reorder_cache
        if self.neuron_config.log_softmax_scores:
            self.decoder_lm_head.add_post_layer_builder(hlo_builder.post_layer)

        n_embd = self.config.n_embd
        d_head = n_embd // config.n_head
        n_heads_tp = config.n_head // config.tp_degree
        # Since GPT2 does not support compilation for multiple batch sizes yet,
        # assert the invariant
        bsizes = self.decoder_lm_head.batch_size
        assert len(bsizes) == 1, "GPT2 does not support compilation for multiple batch sizes"
        bs_idx = 0
        batch_size = self.decoder_lm_head.batch_size[bs_idx]
        self.decoder_batch_size = batch_size
        # Only share caches when we don't generate multiple suggestions
        self.share_caches = self.prompt_batch_size == self.decoder_lm_head.batch_size[bs_idx]
        if self.context_buckets:
            self.broadcaster={}
            for context_length_estimate in self.context_buckets:
                if not self.share_caches:
                    self.broadcaster[context_length_estimate] = decoder.FastCacheBroadcaster(
                        context_length_estimate,
                        self.prompt_batch_size,
                        self.decoder_lm_head.batch_size[bs_idx],
                        n_heads_tp,
                        d_head,
                        config.amp,
                        config.tp_degree,
                        config.n_layer,
                    )
                    self.register_for_serialization(self.broadcaster[context_length_estimate])
        self.decoder_lm_head_for_speculation={}
        self.context_pre_hook = None
        self.context_hook = None
        # Token counter for sliding window attention
        self.num_processed_tokens = 0

    def load_weights(self):
        ops.init()
        self.materialize_embeddings()
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
            is_bsh = self.neuron_config and self.neuron_config.attention_layout == LAYOUT_BSH
            new_layer.add_attention_output(attn.c_proj.weight.detach().T, attn.c_proj.bias.detach(), sharding=1, transposed=False)
            new_layer.add_pre_mlp_layer_norm(layer.ln_2.weight.detach(), layer.ln_2.bias.detach())
            new_layer.add_mlp_input(mlp.c_fc.weight.detach(), mlp.c_fc.bias.detach())
            if self.neuron_config.weight_tiling:
                new_layer.add_mlp_output(mlp.c_proj.weight.detach(), mlp.c_proj.bias.detach())
            else:
                if is_bsh:
                    new_layer.add_mlp_output(
                        mlp.c_proj.weight.detach(),
                        mlp.c_proj.bias.detach(),
                        sharding=0,
                        transposed=True,
                    )
                else:
                    new_layer.add_mlp_output(
                        mlp.c_proj.weight.detach().T,
                        mlp.c_proj.bias.detach(),
                        sharding=1,
                        transposed=False,
                    )
            new_layer.to_neuron()
            layer.nullify()
        ln_f = self.chkpt_model.transformer.ln_f
        ln_f.materialize()
        self.decoder_lm_head.add_final_layer_norm(ln_f.weight.detach(), ln_f.bias.detach())
        lm_head = self.chkpt_model.lm_head
        lm_head.materialize()
        if self.neuron_config.on_device_embedding:
            self.decoder_lm_head.add_pre_layer_parameter(self.chkpt_model.transformer.wte.weight, sharding=1, allow_pad=True)
            self.decoder_lm_head.add_pre_layer_parameter(self.chkpt_model.transformer.wpe.weight, sharding=1, allow_pad=True)
        self.decoder_lm_head.add_lm_head(lm_head.weight.detach().T)
        lm_head.nullify()
        self.decoder_lm_head.to_neuron()
        self.init_rest_of_model()
    
    def materialize_embeddings(self):
        self.chkpt_model.transformer.wte.materialize()
        self.chkpt_model.transformer.wpe.materialize()
    
    def init_rest_of_model(self):
        config = self.config
        # Since GPT2 does not support compilation for multiple batch sizes yet,
        # assert the invariant
        bsizes = self.decoder_lm_head.batch_size
        assert len(bsizes) == 1, "GPT2 does not support compilation for multiple batch sizes"
        bs_idx = 0
        batch_size = self.decoder_lm_head.batch_size[bs_idx]
        if self.context_buckets:
            for i, context_length_estimate in enumerate(self.context_buckets):
                model = self.decoder_lm_head.build_weight_shared(share_caches=self.share_caches,
                                                                    new=self.decoder_lm_head_for_context[context_length_estimate, self.prompt_batch_size])
                if not self.share_caches:
                    source_caches = []
                    for layer in model.layers:
                        source_caches.append(layer.attn_k_cache[model.batch_size[bs_idx]])
                        source_caches.append(layer.attn_v_cache[model.batch_size[bs_idx]])
                    manipulator = parallel.ParallelTensorManipulator(config.tp_degree)
                    target_caches = []
                    for layer in self.decoder_lm_head.layers:
                        attn_k_cache = manipulator.slice_on_nc(
                            layer.attn_k_cache[batch_size],
                            0,
                            start=0,
                            end=context_length_estimate,
                            step=1,
                        )
                        attn_v_cache = manipulator.slice_on_nc(
                            layer.attn_v_cache[batch_size],
                            0,
                            start=0,
                            end=context_length_estimate,
                            step=1,
                        )
                        target_caches.append(attn_k_cache)
                        target_caches.append(attn_v_cache)
                    self.broadcaster[context_length_estimate].set_source_caches(source_caches)
                    self.broadcaster[context_length_estimate].set_target_caches(target_caches)
                self.tensor_pool = tensor_pool.TensorPool()
                # We need to reset once, since there might be NaN initially in KVcache.
                # This is done right after weight loading which is shared for different generation methods.
                self.decoder_lm_head_for_context[context_length_estimate, batch_size] = model
                self.reset(context_length_estimate=context_length_estimate)
        if self.decoder_lm_head_for_speculation:
            for i,k in enumerate(self.decoder_lm_head_for_speculation):
                model= self.decoder_lm_head.build_weight_shared(share_caches=True,
                                                                      new=self.decoder_lm_head_for_speculation[k])
                self.decoder_lm_head_for_speculation[k]=model

    def reset(self,context_length_estimate):
        self.decoder_lm_head.reset()
        self.decoder_lm_head_for_context[context_length_estimate, self.decoder_batch_size].reset()
        self.num_processed_tokens = 0

    def _embedding(self, decoder_lm_head, input_ids, cache_ids, start_ids, is_context_encode):
        inputs_embeds = self.chkpt_model.transformer.wte(input_ids)
        position_ids, start_ids = decoder_lm_head.embed_positions_ids(cache_ids, start_ids)
        position_embeds = self.chkpt_model.transformer.wpe(position_ids)
        hidden = inputs_embeds + position_embeds
        if is_context_encode:
            self.tensor_pool.push([inputs_embeds, position_embeds])
        return hidden, start_ids

    def forward(self, input_ids, cache_ids=None, start_ids=None):
        # Check if sliding window attention is applied
        sliding_window_attn_enabled = self.neuron_config and \
                                      self.neuron_config.sparse_attn and \
                                      self.neuron_config.sparse_attn.skip_masking_decode
        batch_size, context_length = input_ids.shape
        is_context_encode = context_length > 1
        estimate = bucket.find(self.context_buckets, context_length)

        inputs, cache_ids, last_token_id = self._prepare_for_par_ctx_rhs_padding(input_ids, cache_ids)
        batch_size, context_length = inputs.shape

        model = self.decoder_lm_head

        if is_context_encode:
            model = self.decoder_lm_head_for_context[estimate, batch_size]

        if start_ids is None:
            start_ids = torch.zeros(batch_size, dtype=torch.int32)

        if cache_ids is None:
            cache_ids = torch.arange(context_length, dtype=torch.int32)

        # Compute the window starting index for specific mask patterns
        # For other patterns we pass in a default value of 0, it won't be used
        curr_window_start = \
            self.num_processed_tokens - self.neuron_config.sparse_attn.sparse_attn_config.window_size \
            if sliding_window_attn_enabled else 0
        curr_window_start = torch.as_tensor([curr_window_start], dtype=torch.int32)

        if start_ids.shape[0] != batch_size:
            start_ids = start_ids.repeat(batch_size)

        if is_context_encode:
            # The big tensor destruction is slow in CPU. Use asynchronous clear
            # to parallel the tensor free with the context encoding execution.
            task = self.tensor_pool.async_clear()
        if start_ids.shape[0] != batch_size:
            start_ids = start_ids.repeat(batch_size)

        if not self.neuron_config.on_device_embedding:
            inputs, start_ids = self._embedding(model, inputs, cache_ids, start_ids, is_context_encode)
            if self.neuron_config.attention_layout == LAYOUT_HSB:
                inputs = inputs.transpose(0,2).contiguous()

        if context_length > 1:
            if self.neuron_config.log_softmax_scores:
                logits, scores = logits, scores = self.context(inputs, cache_ids, start_ids, last_token_id, curr_window_start)
            else:
                logits = self.context(inputs, cache_ids, start_ids, last_token_id, curr_window_start)
        else:
            if self.neuron_config.log_softmax_scores:
                logits, scores = self.decoder_lm_head(inputs, cache_ids, start_ids, last_token_id, curr_window_start)
            else:
                logits = self.decoder_lm_head(inputs, cache_ids, start_ids, last_token_id, curr_window_start)

        if self.neuron_config.on_device_generation is not None:
            return logits

        # Increment the token counter
        # If running decode mode then last_token_id = 0
        self.num_processed_tokens += (last_token_id + 1)
        logits = self._cast_logits(logits)
        if self.neuron_config.output_all_logits and context_length > 1:
            logits = logits.permute(2, 1, 0)
        else:
            logits = logits[:self.config.vocab_size, -1, :]
            logits = logits.transpose(0, 1)
        if self.neuron_config.log_softmax_scores:
            scores = scores[:self.config.vocab_size, -1, :]
            scores = scores.transpose(0, 1)
        if is_context_encode:
            task.wait()
            self.tensor_pool.push(inputs)
        if is_context_encode and not self.share_caches:
            self.broadcaster[estimate].run_broadcast()
        if self.neuron_config.log_softmax_scores:
            return logits, scores
        return logits

    def speculative_forward(self, input_ids, cache_ids=None, start_ids=None, speculation_length=None):
        # Check if sliding window attention is applied
        sliding_window_attn_enabled = self.neuron_config and \
                                      self.neuron_config.sparse_attn and \
                                      self.neuron_config.sparse_attn.skip_masking_decode

        last_token_id = torch.as_tensor([0], dtype=torch.int32)
        batch_size, context_length = input_ids.shape

        if start_ids is None:
            start_ids = torch.zeros(batch_size, dtype=torch.int32)

        if cache_ids is None:
            cache_ids = torch.arange(context_length, dtype=torch.int32)


        if speculation_length is None:
            model=self.decoder_lm_head
        else:
            model=self.decoder_lm_head_for_speculation[speculation_length, batch_size]

        # Compute the window starting index for specific mask patterns
        # For other patterns we pass in a default value of 0, it won't be used
        curr_window_start = \
            self.num_processed_tokens - self.neuron_config.sparse_attn.sparse_attn_config.window_size \
            if sliding_window_attn_enabled else 0
        curr_window_start = torch.as_tensor([curr_window_start], dtype=torch.int32)
        if start_ids.shape[0] != batch_size:
            start_ids = start_ids.repeat(batch_size)

        if not self.neuron_config.on_device_embedding:
            input_ids, start_ids = self._embedding(model, input_ids, cache_ids, start_ids, is_context_encode=False)
            if self.neuron_config.attention_layout == LAYOUT_HSB:
                input_ids = input_ids.transpose(0,2).contiguous()
        logits = model(input_ids, cache_ids, start_ids, last_token_id, curr_window_start)
        self.num_processed_tokens += (last_token_id + 1)
        logits = self._cast_logits(logits)
        _, n_active_tokens, _ = logits.shape
        logits = logits[:self.config.vocab_size, -n_active_tokens:, :]
        logits = logits.transpose(0, 1)
        return logits

    @torch.no_grad()
    def sample(self, input_ids, sequence_length, cache_ids=None, start_ids=None, top_k=50, streamer=None, output_scores=False, **kwargs):

        if self.neuron_config.on_device_generation:
            return sampling.sample_tokens(self, input_ids, start_ids=start_ids,
                                          sequence_length=sequence_length, config=self.neuron_config.on_device_generation)

        if self.context_pre_hook is not None:
            self.context_pre_hook()
        runtime_batch_size, context_length = input_ids.shape
        assert len(self.decoder_lm_head.batch_size) == 1, "GPT 2 does not support compilation of multiple batch sizes"
        batch_size = self.decoder_lm_head.batch_size[0]
        if batch_size % runtime_batch_size:
            raise ValueError(
                f"model batch_size must be multiples of runtime_batch_size; got "
                f"batch_size={batch_size} and runtime_batch_size={runtime_batch_size}"
            )

        log_softmax = self.neuron_config and self.neuron_config.log_softmax_scores
        if log_softmax:
            next_token_scores, log_softmax_scores = self.forward(input_ids, cache_ids=cache_ids,start_ids=start_ids)
        else:
            next_token_scores = self.forward(input_ids, cache_ids=cache_ids, start_ids=start_ids)
            log_softmax_scores = None
        repeat_factor = batch_size // runtime_batch_size
        input_ids = input_ids.repeat([repeat_factor, 1])
        next_token_scores = next_token_scores.repeat([repeat_factor, 1])
        if log_softmax:
            log_softmax_scores = log_softmax_scores.repeat([repeat_factor, 1])
        if self.context_hook is not None:
            self.context_hook()
        interleaved = sampling.sample_loop(
            self,
            input_ids,
            start_ids,
            next_token_scores,
            sequence_length,
            eos_token_id=self.config.eos_token_id,
            top_k=top_k,
            streamer=streamer,
            output_scores=output_scores,
            neuron_config=self.neuron_config,
            log_softmax_scores=log_softmax_scores,
            cache_ids=cache_ids,
        )
        if output_scores:
            if log_softmax:
                interleaved, scores, log_softmax_scores = interleaved
            else:
                interleaved, scores = interleaved

        # When we pass cache_ids explicitly to sample, we are not using sample to generate
        # all tokens but just a subset of the tokens (For example: To generate leftover tokens
        # autoregressively in speculative sampling). Assuming batch_size = runtime_batch_size = 1
        if not cache_ids:
            interleaved = interleaved.reshape([-1, runtime_batch_size, sequence_length])
            interleaved= interleaved.permute([1, 0, 2]).reshape([-1, sequence_length])

        if output_scores:
            if log_softmax:
                return interleaved, scores, log_softmax_scores
            return interleaved, scores
        return interleaved


class GPT2CheckpointCompatible(module.PretrainedModel):

    def __init__(self, config):
        super().__init__()
        dtype, _, _ = utils.parse_amp(config.amp)
        dtype = dtypes.to_torch_dtype(dtype)
        self.transformer = GPT2Transformer(config)
        self.lm_head = module.LowMemoryLazyLinear(config.vocab_size, dtype=dtype, bias=False)

    def get_tied_parameters(self):
        return [(self.transformer.wte.weight, self.lm_head.weight)]

    def get_base_model(self):
        return self.transformer


class GPT2Transformer(module.LowMemoryModule):

    def __init__(self, config):
        super().__init__()
        self.wte = module.LowMemoryEmbedding(config.vocab_size, config.n_embd)
        self.wpe = module.LowMemoryEmbedding(config.max_position_embeddings, config.n_embd)
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
