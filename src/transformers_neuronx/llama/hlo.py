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
from typing import Optional

from transformers_neuronx import hlo
from transformers_neuronx import constants
from transformers_neuronx import utils
from transformers_neuronx.layers import transformer, rotary, attention, attention_utils
from transformers_neuronx.llama.config import LlamaConfig
from transformers_neuronx.config import NeuronConfig
from transformers_neuronx.constants import LAYOUT_BSH, LAYOUT_HSB


class LlamaForSamplingNoEmbeddingHlo:

    def __init__(self,
        config: LlamaConfig,
        neuron_config: Optional[NeuronConfig] = None
    ):
        self.config = config
        self.neuron_config = neuron_config
        self.n_positions = None

    @property
    def shard_over_batch(self):
        # Property access allows fallback configuration to be enabled after construction
        return (
            self.neuron_config is not None
            and self.neuron_config.group_query_attention == constants.GQA.SHARD_OVER_BATCH
        )

    def inputs(self, scribe, dtype, n_active_tokens, batch_size):
        tensors, dims = transformer.inputs(
            scribe, dtype, batch_size, n_active_tokens, self.config.hidden_size, self.neuron_config)

        return tensors, dims

    def embedding(self, input_ids, cache_ids, start_ids, last_token_id, embed_weight):
        dtype = getattr(input_ids.scribe, self.config.amp)
        hidden = hlo.embedding(embed_weight, input_ids, tp_degree=self.config.tp_degree, dtype=dtype)
        if self.config.hidden_size % self.config.tp_degree != 0:
            hidden = hlo.slice_along(hidden, dim=-1, limit=self.config.hidden_size, start=0)
        if self.neuron_config.attention_layout == LAYOUT_HSB:
            hidden = hlo.transpose210(hidden)
        return hidden

    def pre_layer(self, hidden, cache_ids, start_ids, last_token_id, *weights):
        head_dim = self.config.attention_head_size
        pos_embed = rotary.hlo_rotary_embedding(
            hidden.dtype, int(head_dim * self.config.rotary_percentage), cache_ids,
            base=self.config.rope_theta,
            interpolation_factor=self.config.position_interpolation_factor
        )
        mask, active_mask = hlo.attention_mask(cache_ids, start_ids, self.n_positions)
        return hidden, last_token_id, pos_embed, cache_ids, start_ids, mask, active_mask

    def layer(
            self, hidden, last_token_id, pos_embed, cache_ids, start_ids, mask, active_mask,
            attn_k_cache, attn_v_cache,
            pre_attn_ln_weight, pre_attn_ln_bias,
            attn_q_weight, attn_q_scales, attn_q_bias,
            attn_k_weight, attn_k_scales, attn_k_bias,
            attn_v_weight, attn_v_scales, attn_v_bias,
            attn_out_weight, attn_out_scales, attn_out_bias,
            post_attn_ln_weight, post_attn_ln_bias,
            pre_mlp_ln_weight, pre_mlp_ln_bias,
            mlp_in_weight, mlp_in_scales, mlp_in_bias,
            mlp_out_weight, mlp_out_scales, mlp_out_bias,
            post_mlp_ln_weight, post_mlp_ln_bias,
            in0_weight, in0_scales,
            in1_weight, in1_scales,
            out_weight, out_scales,
        ):
        eps = self.config.rms_norm_eps
        is_bsh = self.neuron_config and self.neuron_config.attention_layout == LAYOUT_BSH
        ln_hidden = hlo.rms_norm(hidden, pre_attn_ln_weight, eps) if is_bsh else hlo.rms_norm(hidden, pre_attn_ln_weight, eps, dim=0)
        attn_output, out_attn_k_cache, out_attn_v_cache = self.attention(
            ln_hidden, cache_ids, start_ids, pos_embed, mask, active_mask,
            attn_k_cache, attn_v_cache,
            attn_q_weight, attn_q_scales, attn_q_bias,
            attn_k_weight, attn_k_scales, attn_k_bias,
            attn_v_weight, attn_v_scales, attn_v_bias,
            attn_out_weight, attn_out_scales, attn_out_bias
        )
        hidden = hlo.add(attn_output, hidden)
        gated_mlp = hlo.gated_mlp_bsh if is_bsh else hlo.gated_mlp
        rms_norm_dim = 2 if is_bsh else 0
        norm_hidden = hlo.rms_norm(hidden, pre_mlp_ln_weight, eps, dim=rms_norm_dim)
        mlp_hidden = gated_mlp(
            norm_hidden,
            in0_weight, in1_weight, out_weight,
            in0_scales=in0_scales,
            in1_scales=in1_scales,
            out_scales=out_scales,
            activation_function='silu',
            tp_degree=self.config.tp_degree,
            neuron_config=self.neuron_config
        )
        res_hidden = hlo.add(mlp_hidden, hidden)
        return res_hidden, out_attn_k_cache, out_attn_v_cache

    def ln_lm_head(self, hidden, last_token_id, rms_weight, unused_bias, lm_head_weight, lm_head_bias, return_all_outputs=True):
        logits = transformer.rms_lm_head(self.config.tp_degree, hidden, last_token_id, rms_weight, lm_head_weight, lm_head_bias, return_all_outputs, eps=self.config.rms_norm_eps, neuron_config=self.neuron_config)
        return logits


    def attention(
        self,
        hidden, cache_ids, start_ids, pos_embed, mask, active_mask,
        cached_keys, cached_values,
        q_weight, q_scales, q_bias,
        k_weight, k_scales, k_bias,
        v_weight, v_scales, v_bias,
        out_weight, out_scales, out_bias,
    ):
        d_head = self.config.attention_head_size
        tp_degree = self.config.tp_degree

        # Compute the expected number of KV heads (Used in case fused QKV is used)
        n_kv_heads_tp = None
        if self.config.num_key_value_heads is not None:
            n_head = self.config.num_attention_heads
            n_kv_head = self.config.num_key_value_heads
            _, n_kv_head_padded = utils.get_qkv_padding(n_head, n_kv_head, tp_degree, self.neuron_config)
            n_kv_heads_tp = n_kv_head_padded // tp_degree

        # Q = (hidden @ wQ) + bQ
        # K = (hidden @ wK) + bK
        # V = (hidden @ wV) + bV
        query, key, value = attention.query_key_value(
            hidden,
            q_weight, q_scales, q_bias,
            k_weight, k_scales, k_bias,
            v_weight, v_scales, v_bias,
            d_head,
            neuron_config=self.neuron_config,
            tp_degree=tp_degree,  # TODO: include tp_degree into neuron_config
            shard_over_batch=self.shard_over_batch,
            n_kv_heads_tp=n_kv_heads_tp,
        )

        # Q = Rotate(Q)
        # K = Rotate(K)
        query, key = rotary.rotate_half(query, key, pos_embed, self.config.rotary_percentage,
                                        tp_degree=tp_degree, shard_over_batch=self.shard_over_batch)

        # Q = Q / sqrt(d_head)
        query = attention.scale(query, d_head)

        # In BSH cache layout, the output of QKV linear projection is still kept as SBH for all QKV.
        bsh_cache_layout = False
        batch_dim = 1
        if self.neuron_config is not None:
            bsh_cache_layout = self.neuron_config.cache_layout == constants.LAYOUT_BSH
        if bsh_cache_layout:
            query, key, value = attention_utils.transpose_qkv(query, key, value)
            batch_dim = 0

        # Single Token Generation ("Prefetch"-style) ans speculative forward
        if active_mask is not None:

            n_active_tokens = key.sizes[1] if bsh_cache_layout else key.sizes[0]
            if n_active_tokens > 1 and self.neuron_config and self.neuron_config.continuous_batching:
                # For speculative forward + continuous batching, slice out samples in the batch size
                # corresponding to the batch size of the speculative head
                slice_sizes = [1] * len(cached_keys.sizes)
                if cached_keys.sizes[batch_dim] == 1:
                    # Use hlo.select for batch size 1 as index select is prohibitively slow
                    # TODO: revert to hlo.index_select once its faster https://t.corp.amazon.com/P126527643
                    cached_keys_s = hlo.select(cached_keys, batch_dim, hlo.reshape(start_ids, slice_sizes), keepdim=True)
                    cached_values_s = hlo.select(cached_values, batch_dim, hlo.reshape(start_ids, slice_sizes), keepdim=True)
                else:
                    cached_keys_s = hlo.index_select(cached_keys, batch_dim, start_ids)
                    cached_values_s = hlo.index_select(cached_values, batch_dim, start_ids)
            else:
                cached_keys_s = cached_keys
                cached_values_s = cached_values

            # Sp = Q @ Kp
            prior_scores = attention.score(query, cached_keys_s, n_kv_heads=self.config.num_key_value_heads,
                                           tp_degree=tp_degree, neuron_config=self.neuron_config)
            prior_scores = attention.mask(prior_scores, mask, tp_degree=tp_degree, shard_over_batch=self.shard_over_batch)

            # Sa = Q @ Ka
            active_score = attention.score(query, key, n_kv_heads=self.config.num_key_value_heads,
                                           tp_degree=tp_degree, neuron_config=self.neuron_config)
            active_score = attention.mask(active_score, active_mask, tp_degree=tp_degree, shard_over_batch=self.shard_over_batch)

            # C = softmax(Sa, Sp) @ (Va, Vp)
            context = attention.context(prior_scores, active_score, cached_values_s, value,
                                        n_kv_heads=self.config.num_key_value_heads, tp_degree=tp_degree,
                                        neuron_config=self.neuron_config)

            # KCache[I], VCache[I] = K, V
            updated_keys, updated_values = attention.fused_kv_update_cache(cached_keys, cached_values, cache_ids,
                                                                           key, value, start_ids, neuron_config=self.neuron_config)

        # Multi-Token Context Encoding
        else:

            context = attention.flash_attention(query, key, value)

            if context is None:
                # S = Q @ K

                score = attention.score(query, key, n_kv_heads=self.config.num_key_value_heads,
                                        tp_degree=tp_degree, neuron_config=self.neuron_config)
                score = attention.mask(score, mask, tp_degree=tp_degree, shard_over_batch=self.shard_over_batch)
                context = attention.context_combined(score, value, n_kv_heads=self.config.num_key_value_heads,
                                                    tp_degree=tp_degree, neuron_config=self.neuron_config)

            # KCache, VCache = K, V
            if cached_keys.sizes == key.sizes:
                updated_keys, updated_values = key, value
            else:
                updated_keys, updated_values = attention.fused_kv_update_cache(cached_keys, cached_values, cache_ids,
                                                                               key, value, start_ids, neuron_config=self.neuron_config)

        # O = (C @ wO) + bO
        output = attention.output(context, out_weight, out_scales, out_bias, tp_degree, self.neuron_config)
        return output, updated_keys, updated_values




    def convert_attn_mask_and_cache_id(self, cache_ids, core_id, n_heads_tp,
                                    batch_size=1, is_context_encoding=False):
        """
        Convert normal cache IDs to the format suitable for sharded KV cache, and create proper attention
        masks. Since each Q/KV head can be distributed to multiple cores, each core will have a
        different mask and cache ID.

        In this version, the KV cache of all KV heads is evenly split across all cores. Tokens are
        written to the KV caches in a strided way: token 0 goes to core 0's cache, token 1 goes to core
        1's cache, etc. When computing active tokens, each core is in charge of tokens that are written
        to it's cache.

        For tokens that should not be written to the current core's KV cache, the cache ID for this token
        is set to cache_size. We need 1 or more garbage entries in the KV cache for this purpose.
        """
        assert len(cache_ids.sizes) == 1, "Assuming 1D cache IDs!"
        assert n_heads_tp == 1, "Assuming each core only process 1 Q head!"

        n_active_tokens = cache_ids.sizes[0]
        cores_per_kv_head = self.config.tp_degree // self.config.num_kv_heads
        cache_size = self.n_positions // cores_per_kv_head
        pred = cache_ids.scribe.pred
        dtype = cache_ids.dtype

        # Real cache ID = raw cache ID // the number of cores that hold a single head's KV cache
        num_cache_splits = cores_per_kv_head
        real_cache_ids = hlo.divide(cache_ids, num_cache_splits)
        # Default cache ID = cache_size
        default_cache_ids = hlo.full(cache_size, dtype, real_cache_ids.sizes)
        # Now mask out the entries that should not go to this core's cache
        target_core_ids = hlo.remainder(cache_ids, num_cache_splits)
        core_id_cast = hlo.cast(core_id, dtype)
        curr_core_id_in_head = hlo.remainder(core_id_cast, num_cache_splits)
        curr_core_id_in_head = hlo.broadcast(curr_core_id_in_head, target_core_ids.sizes, [0])
        mask = hlo.compare(target_core_ids, curr_core_id_in_head, "EQ")
        converted_cache_ids = dtype[cache_ids.sizes].Select(mask, real_cache_ids, default_cache_ids)

        # Generate masks
        if is_context_encoding:
            # We don't need active mask for context encoding
            converted_active_mask = None
            # Prior mask is simpler for context encoding
            converted_mask = hlo.tril_mask(pred, (n_active_tokens, n_active_tokens))
            converted_mask = hlo.broadcast(converted_mask, (batch_size, n_active_tokens, n_active_tokens), [1, 2])
        else:
            converted_mask_size = batch_size, n_active_tokens, cache_size

            # For prior mask, we compute how many tokens are there in this core's KV cache
            num_processed_tokens = hlo.reduce_min(cache_ids, dim=0, keepdim=True)
            core_id_in_head = hlo.remainder(core_id_cast, num_cache_splits)
            num_tokens_on_core = hlo.divide(hlo.subtract(hlo.add(num_processed_tokens, num_cache_splits-1), core_id_in_head), num_cache_splits)
            # Use Iota to generate the mask
            iota = dtype[converted_mask_size].Iota(dimensions=[2])
            num_tokens_on_core_br = hlo.broadcast(num_tokens_on_core, converted_mask_size, [2])
            converted_mask = hlo.less(iota, num_tokens_on_core_br)

            # Construct the active mask based on the rule above, each core is in charge of tokens
            # that are written to its own cache
            converted_active_mask = hlo.tril_mask(pred, (n_active_tokens, n_active_tokens))
            converted_active_mask = hlo.broadcast(converted_active_mask, (batch_size, n_active_tokens, n_active_tokens), broadcast_dimensions=[1, 2])
            mask_br = hlo.broadcast(mask, converted_active_mask.sizes, [2])
            converted_active_mask = hlo.logical_and(converted_active_mask, mask_br)

        return converted_cache_ids, converted_mask, converted_active_mask
