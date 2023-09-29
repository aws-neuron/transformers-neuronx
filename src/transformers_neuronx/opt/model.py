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
import os
import warnings
import torch
from transformers_neuronx import decoder
from transformers_neuronx import dtypes
from transformers_neuronx import hlo
from transformers_neuronx import module
from transformers_neuronx import ops
from transformers_neuronx import sampling
from transformers_neuronx import utils
from transformers_neuronx import base
from transformers_neuronx.config import NeuronConfig
from transformers_neuronx.opt.config import OPTConfig
from transformers_neuronx.layers import attention_hsb as attention, transformer


class OPTForSampling(base.NeuronModelBase):

    def __init__(self, config, batch_size=1, amp=None, tp_degree=2, n_positions=2048,
                 unroll=None, context_length_estimate=None, context_unroll=1, neuron_config=None, **kwargs):
        if amp is None:
            amp = dtypes.to_amp(config.torch_dtype)
        else:
            warnings.warn(f'torch_dtype={config.torch_dtype} ignored in favor of amp={amp}')
        config = OPTConfig(config, n_positions, batch_size, amp, tp_degree, **kwargs)
        # Build model in Python, result in self.chkpt_model
        super().__init__(OPTCheckpointCompatible, config)
        self.config = config

        # Check if input sequence length is allowed given position embedding dimensions
        sequence_length = n_positions
        max_allowed_sequence_length = config.max_position_embeddings
        if sequence_length > max_allowed_sequence_length:
            raise ValueError(f"Sequence length ({sequence_length}) cannot be larger than position embedding's context size ({max_allowed_sequence_length})!")

        if unroll is None:
            unroll = config.num_hidden_layers
        n_positions_list = utils.power_of_two_bucket_sizes(128, n_positions)
        if isinstance(batch_size,int):
            self.batch_sizes = [batch_size]
        elif isinstance(batch_size,list):
            self.batch_sizes = sorted(batch_size)
        else:
            raise TypeError("batch_size must be list of ints or int type")

        attention_head_size = config.hidden_size // config.num_attention_heads
        self.decoder_lm_head = decoder.DecoderLmHeadForSamplingNoEmbedding(
            tp_degree, n_positions_list, 1, # n_active_tokens
            self.batch_sizes, attention_head_size, amp=amp,
            num_layers=config.num_hidden_layers, n_head=config.num_attention_heads,
            unroll=unroll, neuron_config=neuron_config
        )
        self.register_for_serialization(self.decoder_lm_head)
        start_mask = os.environ.get('NEURON_INTERNAL_ASSUME_ALL_PROMPT_LENGTHS_ARE_EQUAL', None) != '1'
        hlo_builder = OPTForSamplingNoEmbeddingHlo(tp_degree, config.hidden_size,
                                                   config.activation_function, start_mask,
                                                   neuron_config=neuron_config)
        self.decoder_lm_head.add_inputs_builder(hlo_builder.inputs)
        self.decoder_lm_head.add_layer_builder(hlo_builder.layer)
        self.decoder_lm_head.add_ln_lm_head_builder(hlo_builder.ln_lm_head)
        self.decoder_lm_head_for_context = dict()
        self.context_length_estimate = context_length_estimate
        self.context_unroll = context_unroll
        self.neuron_config = NeuronConfig() if neuron_config is None else neuron_config
        if self.context_length_estimate is not None:
            for batch_size in self.batch_sizes:
                self.decoder_lm_head_for_context[batch_size] = decoder.DecoderLmHeadForSamplingNoEmbedding(
                                                    tp_degree, 
                                                    [context_length_estimate], 
                                                    context_length_estimate, 
                                                    batch_size, 
                                                    attention_head_size, 
                                                    amp=amp, 
                                                    num_layers=config.num_hidden_layers, 
                                                    n_head=config.num_attention_heads,
                                                    unroll=context_unroll,
                                                    neuron_config=neuron_config,
                                                    allow_pad=self.decoder_lm_head.allow_pad
                                                )
                self.register_for_serialization(self.decoder_lm_head_for_context[batch_size])

    def to_neuron(self):
        ops.init()
        self.chkpt_model.model.decoder.embed_tokens.materialize()
        self.chkpt_model.model.decoder.embed_positions.materialize()
        for layer in self.chkpt_model.model.decoder.layers:
            layer.materialize()
            attn = layer.self_attn
            new_layer = self.decoder_lm_head.new_layer()
            new_layer.add_pre_attention_layer_norm(layer.self_attn_layer_norm.weight.detach(),
                                                   layer.self_attn_layer_norm.bias.detach())
            new_layer.add_attention_query(attn.q_proj.weight.detach().T, attn.q_proj.bias.detach())
            new_layer.add_attention_key(attn.k_proj.weight.detach().T, attn.k_proj.bias.detach())
            new_layer.add_attention_value(attn.v_proj.weight.detach().T, attn.v_proj.bias.detach())
            new_layer.add_attention_output(attn.out_proj.weight.detach(), attn.out_proj.bias.detach(), sharding=1, transposed=False)
            new_layer.add_pre_mlp_layer_norm(layer.final_layer_norm.weight.detach(),
                                             layer.final_layer_norm.bias.detach())
            new_layer.add_mlp_input(layer.fc1.weight.detach().T, layer.fc1.bias.detach())
            if os.environ.get("NEURON_INTERNAL_TRANSFORM_WEIGHT_LAYOUT", None):
                new_layer.add_mlp_output(layer.fc2.weight.detach().T, layer.fc2.bias.detach())
            else:
                new_layer.add_mlp_output(
                    layer.fc2.weight.detach(),
                    layer.fc2.bias.detach(),
                    sharding=1,
                    transposed=False,
                )
            new_layer.to_neuron()
            layer.nullify()
        ln_f = self.chkpt_model.model.decoder.final_layer_norm
        ln_f.materialize()
        self.decoder_lm_head.add_final_layer_norm(ln_f.weight.detach(), ln_f.bias.detach())
        lm_head = self.chkpt_model.lm_head
        lm_head.materialize()
        self.decoder_lm_head.add_lm_head(lm_head.weight.detach().T)
        lm_head.nullify()
        self.decoder_lm_head.to_neuron()
        if self.context_length_estimate is not None:
            for batch_size in self.batch_sizes:
                self.decoder_lm_head_for_context[batch_size] = self.decoder_lm_head.build_weight_shared(new=self.decoder_lm_head_for_context[batch_size])

    def reset(self):
        self.decoder_lm_head.reset()

    def forward(self, input_ids, cache_ids, start_ids=None):
        return self._forward(self.decoder_lm_head, input_ids, cache_ids, start_ids)

    def forward_for_context(self, input_ids, cache_ids, start_ids=None):
        batch_size , _ = input_ids.shape
        context_decoder = self.decoder_lm_head_for_context[batch_size]
        return self._forward(context_decoder, input_ids, cache_ids, start_ids)

    def _forward(self, decoder_lm_head, input_ids, cache_ids, start_ids):
        input_ids, last_token_id = self._prepare_for_par_ctx_rhs_padding(input_ids)
        batch_size, context_length = input_ids.shape
        if cache_ids is None:
            cache_ids = torch.arange(context_length, dtype=torch.int32)
        inputs_embeds = self.chkpt_model.model.decoder.embed_tokens(input_ids)
        batch_size, _ = input_ids.shape
        position_ids, start_ids = decoder_lm_head.embed_positions_ids(cache_ids, start_ids, batch_size)
        position_embeds = self.chkpt_model.model.decoder.embed_positions(position_ids)
        hidden = inputs_embeds + position_embeds
        hidden = hidden.transpose(0, 2).contiguous()
        logits = decoder_lm_head(hidden, cache_ids, start_ids, last_token_id)
        logits = logits.to(torch.float32)
        logits = logits[:self.config.vocab_size, -1, :]
        logits = logits.transpose(0, 1)
        return logits

    def sample(self, input_ids, sequence_length, start_ids=None, top_k=50):
        if self.context_length_estimate is None:
            return sampling.simple_sample(self, input_ids, start_ids, sequence_length,
                                          eos_token_id=self.config.eos_token_id, top_k=top_k)
        _, start = input_ids.shape
        context_length = self.context_length_estimate
        next_token_scores = self.forward_for_context(input_ids, None, start_ids)
        for cur_len in range(context_length, start):
            cache_ids = torch.as_tensor([cur_len], dtype=torch.int32)
            next_token_scores = self(input_ids[:, cur_len:cur_len+1], cache_ids, start_ids)
        return sampling.sample_loop(
            self, input_ids, start_ids, next_token_scores, sequence_length,
            eos_token_id=self.config.eos_token_id, top_k=top_k)


class OPTCheckpointCompatible(module.PretrainedModel):

    def __init__(self, config):
        super().__init__()
        dtype, _, _ = utils.parse_amp(config.amp)
        dtype = dtypes.to_torch_dtype(dtype)
        self.model = OPTModel(config)
        self.lm_head = module.LowMemoryLazyLinear(config.vocab_size, dtype=dtype, bias=False)


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
        dtype, _, _ = utils.parse_amp(config.amp)
        dtype = dtypes.to_torch_dtype(dtype)
        self.fc1 = module.LowMemoryLazyLinear(config.hidden_size, dtype=dtype)
        self.fc2 = module.LowMemoryLazyLinear(config.ffn_dim, dtype=dtype)
        self.final_layer_norm = module.LowMemoryLayerNorm(config.hidden_size)


class OPTAttention(module.LowMemoryModule):

    def __init__(self, config):
        super().__init__()
        hidden_size = config.hidden_size
        dtype, _, _ = utils.parse_amp(config.amp)
        dtype = dtypes.to_torch_dtype(dtype)
        self.q_proj = module.LowMemoryLazyLinear(hidden_size, dtype=dtype)
        self.k_proj = module.LowMemoryLazyLinear(hidden_size, dtype=dtype)
        self.v_proj = module.LowMemoryLazyLinear(hidden_size, dtype=dtype)
        self.out_proj = module.LowMemoryLazyLinear(hidden_size, dtype=dtype)


class OPTForSamplingNoEmbeddingHlo:

    def __init__(self, tp_degree, hidden_size, activation_function, start_mask=True, neuron_config=None, shard_over_batch=False):
        self.tp_degree = tp_degree
        self.hidden_size = hidden_size
        self.activation_function = activation_function
        self.start_mask = start_mask
        self.allow_kv_dot_prefetch = os.environ.get('NEURON_INTERNAL_THOMAS_PREFETCH', None) == '1'
        self.neuron_config = NeuronConfig() if neuron_config is None else neuron_config
        self.shard_over_batch = shard_over_batch

    def inputs(self, scribe, hidden_dtype, n_positions, n_active_tokens, batch_size):
        hidden_sizes = self.hidden_size, n_active_tokens, batch_size
        hidden = hidden_dtype[hidden_sizes].Parameter(parameter_number=0)
        cache_ids = scribe.u32[n_active_tokens].Parameter(parameter_number=1)
        start_ids = scribe.u32[batch_size].Parameter(parameter_number=2)
        last_token_id = scribe.u32.Parameter(parameter_number=3)
        # For the best perf, we only use kv prefetch in the token generation stage
        token_generation = n_active_tokens == 1
        triu_comparison = 'LT' if token_generation else 'LE'
        mask, active_mask = hlo.decoder_attention_mask(
            start_ids,
            cache_ids,
            n_positions,
            triu_comparison=triu_comparison,
            allow_kv_dot_prefetch=token_generation,
            start_mask=True
        )
        return (hidden, last_token_id, cache_ids, mask, active_mask), (1, 0, None, None)

    def layer(self, hidden, last_token_id, cache_ids, mask, active_mask, attn_k_cache, attn_v_cache,
              pre_attn_ln_weight, pre_attn_ln_bias,
              attn_q_weight, attn_q_scales, attn_q_bias,
              attn_k_weight, attn_k_scales, attn_k_bias,
              attn_v_weight, attn_v_scales, attn_v_bias,
              attn_out_weight, attn_out_scales, attn_out_bias,
              post_attn_ln_weight, post_attn_ln_bias,
              pre_mlp_ln_weight, pre_mlp_ln_bias,
              mlp_in_weight, mlp_in_scales, mlp_in_bias,
              mlp_out_weight, mlp_out_scales, mlp_out_bias,
              sparse_mask, active_sparse_mask,
              post_mlp_ln_weight, post_mlp_ln_bias):
        dtype = hidden.dtype
        ln_hidden = hlo.layer_norm(hidden, pre_attn_ln_weight, pre_attn_ln_bias)
        assert attn_k_cache.sizes[-2] * attn_k_cache.sizes[-1] == attn_k_weight.sizes[-1], \
            f"kv cache shape ({attn_k_cache.sizes}) doesn't match kv weight shape ({attn_k_weight.sizes})"
        attn_output, out_attn_k_cache, out_attn_v_cache = self.attention(
            ln_hidden, cache_ids, mask, active_mask, attn_k_cache, attn_v_cache,
            attn_q_weight, attn_q_scales, attn_q_bias,
            attn_k_weight, attn_k_scales, attn_k_bias,
            attn_v_weight, attn_v_scales, attn_v_bias,
            attn_out_weight, attn_out_scales, attn_out_bias,
            neuron_config=self.neuron_config
        )
        hidden = hlo.add(attn_output, hidden)
        ln_hidden = hlo.layer_norm(hidden, pre_mlp_ln_weight, pre_mlp_ln_bias)
        mlp_hidden = hlo.mlp(
            ln_hidden, mlp_in_weight, mlp_in_bias, mlp_out_weight, mlp_out_bias,
            activation_function=self.activation_function, tp_degree=self.tp_degree,
            in_scales=mlp_in_scales, out_scales=mlp_out_scales, neuron_config=self.neuron_config,
            transposed=True,
        )
        hidden = hlo.add(mlp_hidden, hidden)
        return hidden, out_attn_k_cache, out_attn_v_cache

    def ln_lm_head(self, hidden, last_token_id, ln_f_weight, ln_f_bias, lm_head_weight, lm_head_bias, n_parallel_output_tokens=1):
        return transformer.ln_lm_head(hidden, last_token_id, ln_f_weight, ln_f_bias, lm_head_weight, lm_head_bias, n_parallel_output_tokens)

    def attention(self, hidden, cache_ids, mask, active_mask,
                  cached_keys, cached_values,
                  q_weight, q_scales, q_bias,
                  k_weight, k_scales, k_bias,
                  v_weight, v_scales, v_bias,
                  out_weight, out_scales, out_bias,
                  neuron_config=None):

        dtype = hidden.dtype
        scribe = hidden.scribe
        f32 = scribe.f32

        hidden_size, n_active_tokens, n_seqs = hidden.sizes

        max_ctx_plus_n_active_tokens, _, n_kv_heads_tp, d_head = cached_keys.sizes
        _, hidden_size_tp = q_weight.sizes
        _, kv_hidden_size_tp = k_weight.sizes
        n_head = hidden_size // d_head
        tp_degree = hidden_size // hidden_size_tp
        n_kv_heads = n_kv_heads_tp * tp_degree

        # Q = (hidden @ wQ) + bQ
        # K = (hidden @ wK) + bK
        # V = (hidden @ wV) + bV
        query, key, value = attention.query_key_value(
            hidden,
            q_weight, q_scales, q_bias,
            k_weight, k_scales, k_bias,
            v_weight, v_scales, v_bias,
            d_head,
            neuron_config=neuron_config,
            tp_degree=tp_degree,
            shard_over_batch=self.shard_over_batch,
        )

        # Q = Q / sqrt(d_head)
        query = attention.scale(query, d_head)

        # Single Token Generation ("Prefetch"-style)
        if active_mask is not None:

            # Sp = Q @ Kp
            prior_scores = attention.score(query, cached_keys, n_kv_heads=n_kv_heads, tp_degree=tp_degree, shard_over_batch=self.shard_over_batch)
            prior_scores = attention.mask(prior_scores, mask, tp_degree=tp_degree, shard_over_batch=self.shard_over_batch)
            prior_scores = hlo.cast(prior_scores, f32)

            # Sa = Q @ Ka
            active_score = attention.score(query, key, n_kv_heads=n_kv_heads, tp_degree=tp_degree, shard_over_batch=self.shard_over_batch)
            active_score = attention.mask(active_score, active_mask, tp_degree=tp_degree, shard_over_batch=self.shard_over_batch)
            active_score = hlo.cast(active_score, f32)

            # C = softmax(Sa, Sp) @ (Va, Vp)
            context = attention.context(prior_scores, active_score, cached_values, value, n_kv_heads=n_kv_heads, dtype=dtype, \
                                        tp_degree=tp_degree, shard_over_batch=self.shard_over_batch)

            # KCache[I] = K
            # VCache[I] = V
            cached_keys = attention.update_cache(cached_keys, cache_ids, key)
            cached_values = attention.update_cache(cached_values, cache_ids, value)

        # Multi-Token Context Encoding
        else:

            # S = Q @ K
            score = attention.score(query, key, n_kv_heads=n_kv_heads, tp_degree=tp_degree, shard_over_batch=self.shard_over_batch)
            score = attention.mask(score, mask, tp_degree=tp_degree, shard_over_batch=self.shard_over_batch)
            score = hlo.cast(score, f32)
            context = attention.context_combined(score, value, n_kv_heads=n_kv_heads, dtype=dtype, \
                                                 tp_degree=tp_degree, shard_over_batch=self.shard_over_batch)

            # KCache = K
            # VCache = V
            cached_keys = key
            cached_values = value

        output = attention.output(context, out_weight, out_scales, out_bias, self.tp_degree, neuron_config)
        return output, cached_keys, cached_values


class OPTForGreedySearchNoEmbeddingHlo(OPTForSamplingNoEmbeddingHlo):

    def ln_lm_head(self, *args, **kwargs):
        logits = super().ln_lm_head(*args, **kwargs)
        return hlo.argmax(logits, dim=0, keepdim=True, tp_degree=self.tp_degree)


class OPTForGreedySearch(OPTForSampling):
    """
    An OPT model variant which performs greedy token selection on device.

    This variant reduces the generation flexibility by compiling the token
    selection into the model binary. This may improve performance for large
    batch sizes when compared to CPU token selection since this avoids
    large data copies from the Neuron device to CPU.

    In contrast, when using CPU token selection, different generation
    strategies can be used with the same model since the token selection
    is not compiled into the model graph.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hlo_builder = OPTForGreedySearchNoEmbeddingHlo(
            self.config.tp_degree,
            self.config.hidden_size,
            self.config.activation_function
        )
        self.decoder_lm_head.add_ln_lm_head_builder(hlo_builder.ln_lm_head)

    def forward(self, input_ids, cache_ids, start_ids=None):
        inputs_embeds = self.chkpt_model.model.decoder.embed_tokens(input_ids)
        batch_size, _ = input_ids.shape
        position_ids, start_ids = self.decoder_lm_head.embed_positions_ids(cache_ids, start_ids, batch_size)
        position_embeds = self.chkpt_model.model.decoder.embed_positions(position_ids)
        hidden = inputs_embeds + position_embeds
        hidden = hidden.transpose(0, 2).contiguous()
        tokens = self.decoder_lm_head(hidden, cache_ids, start_ids)
        tokens = tokens[:, -1, :]
        tokens = tokens.transpose(0, 1)
        return tokens

    def sample(self, input_ids, sequence_length, start_ids=None):
        return sampling.sample_tokens(self, input_ids, start_ids, sequence_length)
