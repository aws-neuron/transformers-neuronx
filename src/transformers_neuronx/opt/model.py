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
from transformers_neuronx import bucket
from transformers_neuronx.constants import FUSED_QKV_TP_FACTOR, LAYOUT_BSH, LAYOUT_HSB
from transformers_neuronx import constants
from transformers_neuronx.config import NeuronConfig
from transformers_neuronx.opt.config import OPTConfig
from transformers_neuronx.layers import transformer, generation, attention, attention_utils


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
        self.neuron_config = neuron_config if neuron_config else NeuronConfig()
        if self.neuron_config.on_device_generation:
            self.neuron_config.on_device_generation.vocab_size = self.config.vocab_size

        # Check if input sequence length is allowed given position embedding dimensions
        if isinstance(n_positions, list) or isinstance(n_positions, tuple):
            max_bucket = max(n_positions)
        else:
            max_bucket = n_positions
        max_allowed_sequence_length = config.max_position_embeddings
        if (max_bucket) > max_allowed_sequence_length:
            raise ValueError(f"Sequence length ({max_bucket}) cannot be larger than position embedding's context size ({max_allowed_sequence_length})!")

        if unroll is None:
            unroll = config.num_hidden_layers
        self.unroll = unroll

        if context_unroll is None:
            context_unroll = config.num_hidden_layers
        self.context_unroll = context_unroll

        self.token_buckets = bucket.token_sizes(n_positions)
        self.context_buckets = bucket.context_sizes(context_length_estimate, self.token_buckets)
        self.batch_sizes = bucket.batch_sizes(batch_size)

        attention_head_size = config.hidden_size // config.num_attention_heads

        start_mask = os.environ.get('NEURON_INTERNAL_ASSUME_ALL_PROMPT_LENGTHS_ARE_EQUAL', None) != '1'
        hlo_builder = OPTForSamplingNoEmbeddingHlo(tp_degree, config.hidden_size,
                                                   config.activation_function,
                                                   eos_token_id=config.eos_token_id,
                                                   amp=amp, start_mask=start_mask,
                                                   neuron_config=self.neuron_config, position_offset=2)

        self.decoder_param_set = decoder.DecoderLmHeadForSamplingNoEmbedding(
            tp_degree, n_positions_list=self.token_buckets, n_active_tokens=1,
            batch_size=self.batch_sizes, attention_head_size=attention_head_size, amp=amp,
            num_layers=config.num_hidden_layers, n_head=config.num_attention_heads,
            unroll=unroll, neuron_config=self.neuron_config, builder=hlo_builder,
        )
        self.decoder_lm_head = self.decoder_param_set.init_token_decoder(unroll=self.unroll, buckets=self.token_buckets, model_obj=self)
        self.decoder_lm_head_for_context = self.decoder_param_set.init_context_decoder(unroll=self.context_unroll, buckets=self.context_buckets, model_obj=self)

        # Track number of processed tokens for sliding window attention
        self.num_processed_tokens = 0

    def load_weights(self):
        ops.init()
        self.materialize_embeddings()
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
            if self.neuron_config.weight_tiling:
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
        if self.neuron_config.on_device_embedding:
            self.decoder_lm_head.add_pre_layer_parameter(self.chkpt_model.model.decoder.embed_tokens.weight.detach(), sharding=1, allow_pad=True)
            self.decoder_lm_head.add_pre_layer_parameter(self.chkpt_model.model.decoder.embed_positions.weight.detach(), sharding=1, allow_pad=True)
        lm_head.nullify()
        self.decoder_lm_head.to_neuron()
        self.init_rest_of_model()
    
    def materialize_embeddings(self):
        self.chkpt_model.model.decoder.embed_tokens.materialize()
        self.chkpt_model.model.decoder.embed_positions.materialize()
    
    def init_rest_of_model(self):
        if self.context_buckets:
            for context_length_estimate in self.context_buckets:
                for batch_size in self.batch_sizes:
                    model = self.decoder_lm_head.build_weight_shared(new=self.decoder_lm_head_for_context[context_length_estimate, batch_size], share_caches=True)
                    self.decoder_lm_head_for_context[context_length_estimate, batch_size] = model


    def reset(self):
        self.decoder_lm_head.reset()
        # Reset the token counter
        self.num_processed_tokens = 0

    def forward(self, input_ids, cache_ids=None, start_ids=None):

        inputs, cache_ids, start_ids, last_token_id = self._preprocess(input_ids, start_ids=start_ids, cache_ids=cache_ids)

        if not self.neuron_config.on_device_embedding:
            inputs = self.chkpt_model.model.decoder.embed_tokens(inputs)
            position_ids, start_ids = self.decoder_lm_head.embed_positions_ids(cache_ids, start_ids)
            position_embeds = self.chkpt_model.model.decoder.embed_positions(position_ids)
            inputs = inputs + position_embeds
            if self.neuron_config.attention_layout == LAYOUT_HSB:
                inputs = inputs.transpose(0, 2).contiguous()

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

        result = self._forward(inputs, cache_ids, start_ids, last_token_id, curr_window_start)
        self.num_processed_tokens += (last_token_id+1)
        return result

    def sample(self, input_ids, sequence_length, start_ids=None, top_k=50, streamer=None):
        if self.neuron_config.on_device_generation:
            return sampling.sample_tokens(self, input_ids, start_ids=start_ids,
                                          sequence_length=sequence_length, config=self.neuron_config.on_device_generation, streamer=streamer)
        else:
            return sampling.simple_sample(self, input_ids, start_ids, sequence_length,
                                          eos_token_id=self.config.eos_token_id, top_k=top_k, streamer=streamer)


class OPTCheckpointCompatible(module.PretrainedModel):

    def __init__(self, config):
        super().__init__()
        dtype, _, _ = utils.parse_amp(config.amp)
        dtype = dtypes.to_torch_dtype(dtype)
        self.model = OPTModel(config)
        self.lm_head = module.LowMemoryLazyLinear(config.vocab_size, dtype=dtype, bias=False)

    def get_tied_parameters(self):
        return [(self.model.decoder.embed_tokens.weight, self.lm_head.weight)]

    def get_base_model(self):
        return self.model


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

    def __init__(self, tp_degree, hidden_size, activation_function, eos_token_id=None, amp=None, start_mask=True, neuron_config=None, position_offset=0):
        self.tp_degree = tp_degree
        self.hidden_size = hidden_size
        self.activation_function = activation_function
        self.eos_token_id = eos_token_id
        self.amp = amp
        self.start_mask = start_mask
        self.allow_kv_dot_prefetch = os.environ.get('NEURON_INTERNAL_THOMAS_PREFETCH', None) == '1'
        self.neuron_config = NeuronConfig() if neuron_config is None else neuron_config
        self.n_positions = None

        # Offset for for architecture specific on-device embedding
        # OPT Reference: https://github.com/huggingface/transformers/blob/v4.37.2/src/transformers/models/opt/modeling_opt.py#L94-L96
        self.position_offset = position_offset

    @property
    def shard_over_batch(self):
        # Property access allows fallback configuration to be enabled after construction
        return self.neuron_config.group_query_attention == constants.GQA.SHARD_OVER_BATCH

    def inputs(self, scribe, dtype, n_active_tokens, batch_size):
        tensors, dims = transformer.inputs(
            scribe, dtype, batch_size, n_active_tokens, self.hidden_size, self.neuron_config
        )
        curr_window_start = scribe.s32[1].Parameter(parameter_number=4)
        return (*tensors, curr_window_start), (*dims, None)

    def embed_positions_ids(self, position_ids, start_ids):
        batch_size, = start_ids.sizes
        position_ids = hlo.unsqueeze(position_ids, dim=0)
        position_ids = hlo.broadcast(position_ids, out_dim_size=(batch_size, position_ids.sizes[-1]), broadcast_dimensions=(0,1))
        start_ids = hlo.unsqueeze(start_ids, dim=1)
        start_ids = hlo.broadcast(start_ids, out_dim_size=(batch_size, position_ids.sizes[-1]), broadcast_dimensions=(0,1))
        position_ids = hlo.subtract(position_ids, start_ids)
        zero = hlo.full(value=0, dtype=position_ids.dtype, sizes=(position_ids.sizes))
        mask = hlo.less(position_ids, zero)
        return hlo.masked_select(mask, zero, position_ids)

    def embedding(self, input_ids, cache_ids, start_ids, last_token_id, curr_window_start, wte, wpe):
        dtype = getattr(input_ids.scribe, self.amp)
        inputs_embeds = hlo.embedding(wte, input_ids, tp_degree=self.tp_degree, dtype=dtype)
        position_ids = self.embed_positions_ids(cache_ids, start_ids)

        if self.position_offset != 0:
            offset = hlo.full_like(position_ids, self.position_offset)
            position_ids = hlo.add(position_ids, offset)

        position_embeds = hlo.embedding(wpe, position_ids, tp_degree=self.tp_degree, dtype=dtype)
        hidden = hlo.add(inputs_embeds, position_embeds)

        if self.hidden_size % self.tp_degree != 0:
            hidden = hlo.slice_along(hidden, dim=-1, limit=self.hidden_size, start=0)
        if self.neuron_config.attention_layout == LAYOUT_HSB:
            hidden = hlo.transpose210(hidden)
        return hidden

    def pre_layer(self, hidden, cache_ids, start_ids, last_token_id, curr_window_start, *weights):
        mask, active_mask = hlo.attention_mask(cache_ids, start_ids, self.n_positions,
                                               last_token_id=last_token_id, neuron_config=self.neuron_config)
        return hidden, last_token_id, curr_window_start, cache_ids, start_ids, mask, active_mask

    def layer(self, hidden, last_token_id, curr_window_start, cache_ids, start_ids, mask, active_mask, attn_k_cache, attn_v_cache,
              pre_attn_ln_weight, pre_attn_ln_bias,
              fused_pre_attn_ln_qkv_weight,
              attn_q_weight, attn_q_scales, attn_q_bias,
              attn_k_weight, attn_k_scales, attn_k_bias,
              attn_v_weight, attn_v_scales, attn_v_bias,
              attn_out_weight, attn_out_scales, attn_out_bias,
              post_attn_ln_weight, post_attn_ln_bias,
              pre_mlp_ln_weight, pre_mlp_ln_bias,
              fused_pre_mlp_ln_in_weight,
              mlp_in_weight, mlp_in_scales, mlp_in_bias,
              mlp_out_weight, mlp_out_scales, mlp_out_bias,
              post_mlp_ln_weight, post_mlp_ln_bias,
              ):
        dtype = hidden.dtype
        is_bsh = self.neuron_config and self.neuron_config.attention_layout == LAYOUT_BSH
        layer_norm = hlo.layer_norm_bsh if is_bsh else hlo.layer_norm
        ln_hidden = layer_norm(hidden, pre_attn_ln_weight, pre_attn_ln_bias)
        if self.neuron_config and self.neuron_config.fuse_qkv:
            k_weight_shape = attn_q_weight.sizes
            k_weight_dim = k_weight_shape[-1] // FUSED_QKV_TP_FACTOR
        else:
            k_weight_shape = attn_k_weight.sizes
            k_weight_dim = k_weight_shape[-1]
        attn_output, out_attn_k_cache, out_attn_v_cache = self.attention(
            ln_hidden, curr_window_start, cache_ids, mask, active_mask, attn_k_cache, attn_v_cache,
            attn_q_weight, attn_q_scales, attn_q_bias,
            attn_k_weight, attn_k_scales, attn_k_bias,
            attn_v_weight, attn_v_scales, attn_v_bias,
            attn_out_weight, attn_out_scales, attn_out_bias,
            neuron_config=self.neuron_config
        )
        hidden = hlo.add(attn_output, hidden)
        ln_hidden = layer_norm(hidden, pre_mlp_ln_weight, pre_mlp_ln_bias)
        mlp = hlo.mlp_bsh if is_bsh else hlo.mlp
        mlp_hidden = mlp(
            ln_hidden, mlp_in_weight, mlp_in_bias, mlp_out_weight, mlp_out_bias,
            activation_function=self.activation_function, tp_degree=self.tp_degree,
            in_scales=mlp_in_scales, out_scales=mlp_out_scales, neuron_config=self.neuron_config,
            transposed=True,
        )
        hidden = hlo.add(mlp_hidden, hidden)
        return hidden, out_attn_k_cache, out_attn_v_cache

    def ln_lm_head(self, hidden, last_token_id, ln_f_weight, ln_f_bias, lm_head_weight, lm_head_bias, return_all_outputs=True):
        logits = transformer.ln_lm_head(self.tp_degree, hidden, last_token_id, ln_f_weight, ln_f_bias, lm_head_weight, lm_head_bias, return_all_outputs, neuron_config=self.neuron_config)
        return logits

    def post_layer(self, logits):
        if self.neuron_config.log_softmax_scores:
            scores = hlo.log_softmax(logits, tp_degree=self.tp_degree, dim=0)
            return logits, scores
        return logits

    def attention(self, hidden, curr_window_start, cache_ids, mask, active_mask,
                  cached_keys, cached_values,
                  q_weight, q_scales, q_bias,
                  k_weight, k_scales, k_bias,
                  v_weight, v_scales, v_bias,
                  out_weight, out_scales, out_bias,
                  neuron_config=None):
        enable_sparse_attn = neuron_config and neuron_config.sparse_attn
        sparse_mask = None
        window_attn_decode = False
        if enable_sparse_attn:
            # Generate the sparse masks if necessary. Our behavior here:
            # - If in prefill mode (n_active_tokens > 1), sparse mask is always
            #   generated or provided by user.
            # - If in decode mode (n_active_tokens = 1), sparse mask is not
            #   generated if 1) window attention is used, or 2) block-sparse
            #   attention with only local blocks is used. Otherwise it's generated.
            # - Active sparse mask is only generated in decode mode. Similarly,
            #   it's only generated if user is not using the two patterns mentioned
            #   above.

            # Determine sparse mask shapes according to the attention mask shapes
            n_active_tokens, n_positions = mask.sizes[-2:]
            sparse_mask = neuron_config.sparse_attn.create_sparse_mask(n_active_tokens, n_positions)
            # Directly convert the masks to HLO constants
            if sparse_mask is not None:
                sparse_mask = hlo.literal(mask.scribe.pred, sparse_mask)
            window_attn_decode = neuron_config.sparse_attn.skip_masking_decode

        dtype = hidden.dtype
        scribe = hidden.scribe
        f32 = scribe.f32

        hidden_size, n_active_tokens, n_seqs = hidden.sizes
        if len(q_weight.sizes) == 4:
            weight_tiling = True
            tile_size, hidden_size_tile, _, _ = q_weight.sizes
            hidden_size_tp = tile_size * hidden_size_tile
        else:
            _, hidden_size_tp = q_weight.sizes
        fuse_qkv = neuron_config and neuron_config.fuse_qkv
        if fuse_qkv:
            hidden_size_tp //= FUSED_QKV_TP_FACTOR
            kv_hidden_size_tp = hidden_size_tp
        else:
            _, kv_hidden_size_tp = k_weight.sizes
        _, _, _, d_head = cached_keys.sizes
        n_head = hidden_size // d_head
        tp_degree = hidden_size // hidden_size_tp

        if self.shard_over_batch:
            max_ctx_plus_n_active_tokens, n_seqs_per_nc, n_kv_heads, d_head = cached_keys.sizes
        else:
            max_ctx_plus_n_active_tokens, _, n_kv_heads_tp, d_head = cached_keys.sizes
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

        # In BSH cache layout, the output of QKV linear projection is still kept as SBH for all QKV.
        bsh_cache_layout = False
        if self.neuron_config is not None:
            bsh_cache_layout = self.neuron_config.cache_layout == constants.LAYOUT_BSH
        if bsh_cache_layout:
            query, key, value = attention_utils.transpose_qkv(query, key, value)

        # Single Token Generation ("Prefetch"-style)
        if active_mask is not None:
            # When using window attention, directly slice the KV cache
            # Since GPT uses left padding, we always pick the right-most side of the KV cache
            # KV cache layout: (n_positions, bs, n_heads, head_size)
            # Mask layout: (bs, n_active_tokens, n_positions)
            useful_cached_keys = cached_keys
            useful_cached_values = cached_values
            useful_mask = mask
            if window_attn_decode:
                window_size = neuron_config.sparse_attn.sparse_attn_config.window_size
                useful_cached_keys = hlo.dynamic_slice_along(cached_keys, dim=0, start=curr_window_start, size=window_size)
                useful_cached_values = hlo.dynamic_slice_along(cached_values, dim=0, start=curr_window_start, size=window_size)
                useful_mask = hlo.dynamic_slice_along(mask, dim=2, start=curr_window_start, size=window_size)

            # Sp = Q @ Kp
            prior_scores = attention.score(query, useful_cached_keys, n_kv_heads=n_kv_heads, tp_degree=tp_degree, neuron_config=self.neuron_config)
            prior_scores = attention.mask(prior_scores, useful_mask, tp_degree=tp_degree, shard_over_batch=self.shard_over_batch)
            # Apply the sparse mask to the raw scores, notice that we don't have the mask for window attention
            if enable_sparse_attn and not window_attn_decode:
                assert sparse_mask is not None, "Require a valid sparse mask when not using sliding window attention during decode!"
                # Not sharding over batch because the sparse mask is not supposed to have a batch dimension
                prior_scores = attention.mask(prior_scores, sparse_mask, tp_degree=None, shard_over_batch=False)
            prior_scores = hlo.cast(prior_scores, f32)

            # Sa = Q @ Ka
            active_score = attention.score(query, key, n_kv_heads=n_kv_heads, tp_degree=tp_degree, neuron_config=self.neuron_config)
            active_score = attention.mask(active_score, active_mask, tp_degree=tp_degree, shard_over_batch=self.shard_over_batch)
            active_score = hlo.cast(active_score, f32)

            # C = softmax(Sa, Sp) @ (Va, Vp)
            context = attention.context(prior_scores, active_score, useful_cached_values, value, sparse_mask=sparse_mask,
                                        n_kv_heads=n_kv_heads, dtype=dtype, tp_degree=tp_degree, neuron_config=self.neuron_config)

            # KCache[I] = K
            # VCache[I] = V
            cached_keys = attention.update_cache(cached_keys, cache_ids, key)
            cached_values = attention.update_cache(cached_values, cache_ids, value)

        # Multi-Token Context Encoding
        else:
            # S = Q @ K
            score = attention.score(query, key, n_kv_heads=n_kv_heads, tp_degree=tp_degree, neuron_config=self.neuron_config)
            score = attention.mask(score, mask, tp_degree=tp_degree, shard_over_batch=self.shard_over_batch)
            if enable_sparse_attn:
                score = attention.mask(score, sparse_mask, tp_degree=None, shard_over_batch=False)
            score = hlo.cast(score, f32)
            context = attention.context_combined(score, value, sparse_mask=sparse_mask, n_kv_heads=n_kv_heads, dtype=dtype, \
                                                 tp_degree=tp_degree, neuron_config=self.neuron_config)

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
