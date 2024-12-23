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
from transformers_neuronx.layers import attention, transformer, rotary, attention_utils
from transformers_neuronx.mistral.config import MistralConfig
from transformers_neuronx.config import NeuronConfig
from transformers_neuronx.constants import LAYOUT_BSH, LAYOUT_HSB
from transformers_neuronx.sparse_attn_utils import build_sliding_window_mask

class MistralForSamplingNoEmbeddingHlo:

    def __init__(self,
        config: MistralConfig,
        neuron_config: Optional[NeuronConfig] = None
    ):
        self.config = config
        self.neuron_config = neuron_config
        self.n_positions = None

    def inputs(self, scribe, dtype, n_active_tokens, batch_size):
        tensors, dims = transformer.inputs(
            scribe, dtype, batch_size, n_active_tokens, self.config.hidden_size, self.neuron_config
        )
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
        mask, active_mask = hlo.attention_mask(cache_ids, start_ids, self.n_positions,
                                               last_token_id=last_token_id, neuron_config=self.neuron_config)
        return hidden, last_token_id, pos_embed, cache_ids, start_ids, mask, active_mask

    def layer(
            self, hidden, last_token_id, pos_embed, cache_ids, start_ids, mask, active_mask,
            attn_k_cache, attn_v_cache,
            pre_attn_ln_weight, pre_attn_ln_bias,
            fused_pre_attn_ln_qkv_weight,
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
        # Generate the sparse masks. Our behavior here:
        # - If in prefill mode (n_active_tokens > 1), sparse mask is always
        #   generated or provided by user.
        # - If in decode mode (n_active_tokens = 1), sparse mask is not
        #   generated if 1) window attention is used, or 2) block-sparse
        #   attention with only local blocks is used. Otherwise it's generated.
        # - Active sparse mask is only generated in decode mode. Similarly,
        #   it's only generated if user is not using the two patterns mentioned
        #   above.
        # - The sparse maks here is sliding-window mask

        n_active_tokens, n_positions = mask.sizes[-2:]
        assert (n_active_tokens == 1) or (n_active_tokens == n_positions), \
            "Only supporting decode mode (q_seq_len=1) or self-attention mode (q_seq_len=k_seq_len)!"

        shard_over_batch = (
            self.neuron_config is not None
            and self.neuron_config.group_query_attention == constants.GQA.SHARD_OVER_BATCH
        )

        # Don't generate mask if q_seq_len = 1 (decode)
        if n_active_tokens == 1:
            sparse_mask = None
        else:
            if self.config.sliding_window:
                # Generate sliding-window mask
                sparse_mask = build_sliding_window_mask(n_active_tokens, n_positions, self.config.sliding_window, causal=True)
                sparse_mask = hlo.literal(mask.scribe.pred, sparse_mask)
            else:
                sparse_mask = None

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
            shard_over_batch=shard_over_batch,
            n_kv_heads_tp=n_kv_heads_tp,
        )

        # Q = Rotate(Q)
        # K = Rotate(K)
        query, key = rotary.rotate_half(query, key, pos_embed, self.config.rotary_percentage,
                                        tp_degree=tp_degree, shard_over_batch=shard_over_batch)

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

            if self.config.sliding_window:
                useful_cached_keys, useful_cached_values, useful_mask = hlo.sliding_window_slice(cached_keys, cached_values, mask, cache_ids, self.config.sliding_window, n_positions, self.neuron_config.cache_layout, self.neuron_config.lhs_aligned)
            else:
                useful_cached_keys, useful_cached_values, useful_mask = cached_keys, cached_values, mask

            # Sp = Q @ Kp
            prior_scores = attention.score(query, useful_cached_keys, n_kv_heads=self.config.num_key_value_heads,
                                           tp_degree=tp_degree, neuron_config=self.neuron_config)
            prior_scores = attention.mask(prior_scores, useful_mask, tp_degree=tp_degree, shard_over_batch=shard_over_batch)

            # Sa = Q @ Ka
            active_score = attention.score(query, key, n_kv_heads=self.config.num_key_value_heads,
                                           tp_degree=tp_degree, neuron_config=self.neuron_config)
            active_score = attention.mask(active_score, active_mask, tp_degree=tp_degree, shard_over_batch=shard_over_batch)

            # C = softmax(Sa, Sp) @ (Va, Vp)
            context = attention.context(prior_scores, active_score, useful_cached_values, value, sparse_mask=sparse_mask,
                                        n_kv_heads=self.config.num_key_value_heads, tp_degree=tp_degree,
                                        neuron_config=self.neuron_config)

            # KCache[I], VCache[I] = K, V
            updated_keys, updated_values = attention.fused_kv_update_cache(cached_keys, cached_values, cache_ids,
                                                                           key, value, start_ids, neuron_config=self.neuron_config)

        # Multi-Token Context Encoding
        else:

            # S = Q @ K
            score = attention.score(query, key, n_kv_heads=self.config.num_key_value_heads,
                                    tp_degree=tp_degree, neuron_config=self.neuron_config)
            score = attention.mask(score, mask, tp_degree=tp_degree, shard_over_batch=shard_over_batch)
            if self.config.sliding_window:
                score = attention.sparse_attn_mask(score, sparse_mask)
            context = attention.context_combined(score, value, sparse_mask=sparse_mask, n_kv_heads=self.config.num_key_value_heads,
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
