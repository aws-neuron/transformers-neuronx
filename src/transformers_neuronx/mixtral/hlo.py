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
from transformers_neuronx.layers import attention_hsb as attention, transformer, rotary
from transformers_neuronx.mistral.hlo import MistralForSamplingNoEmbeddingHlo
from transformers_neuronx.mixtral.config import MixtralConfig
from transformers_neuronx.config import NeuronConfig
from transformers_neuronx.constants import LAYOUT_BSH, LAYOUT_HSB
from transformers_neuronx.sparse_attn_utils import build_sliding_window_mask

class MixtralForSamplingNoEmbeddingHlo(MistralForSamplingNoEmbeddingHlo):

    def __init__(self,
        config: MixtralConfig,
        neuron_config: Optional[NeuronConfig] = None
    ):
        self.config = config
        self.neuron_config = neuron_config

        is_bsh = self.neuron_config and self.neuron_config.attention_layout == LAYOUT_BSH
        assert not is_bsh, "BSH layout is currently not supported for moe_layer"
        assert str(MistralForSamplingNoEmbeddingHlo.attention) == str(self.attention.__func__), \
            "The self.attention() function should be derived from MistralForSamplingNoEmbeddingHlo.attention()"

    def inputs(self, scribe, dtype, n_positions, n_active_tokens, batch_size):

        hidden, cache_ids, start_ids, last_token_id, dims = transformer.inputs(
            scribe, dtype, batch_size, n_active_tokens, self.config.hidden_size, self.neuron_config
        )
        curr_window_start = scribe.s32.Parameter(parameter_number=4)

        head_dim = self.config.attention_head_size
        pos_embed = rotary.hlo_rotary_embedding(dtype, int(head_dim * self.config.rotary_percentage), cache_ids,
                                                base=self.config.rope_theta,
                                                interpolation_factor=self.config.position_interpolation_factor)
        mask, active_mask = hlo.attention_mask(cache_ids, start_ids, n_positions)

        return (hidden, last_token_id, curr_window_start, pos_embed, cache_ids, start_ids, mask, active_mask), (*dims, None)

    def embedding(self, input_ids, embed_weight):
        dtype = getattr(input_ids.scribe, self.config.amp)
        hidden = hlo.embedding(embed_weight, input_ids, tp_degree=self.config.tp_degree, dtype=dtype)
        if self.config.hidden_size % self.config.tp_degree != 0:
            hidden = hlo.slice_along(hidden, dim=-1, limit=self.config.hidden_size, start=0)
        if self.neuron_config.attention_layout == LAYOUT_HSB:
            hidden = hlo.transpose210(hidden)
        return hidden

    def pre_layer(self, hidden, last_token_id, curr_window_start, pos_embed, cache_ids, start_ids, mask, active_mask, *pre_layer_weights):
        if self.neuron_config and self.neuron_config.on_device_embedding:
            hidden = self.embedding(hidden, *pre_layer_weights)
        return hidden, last_token_id, curr_window_start, pos_embed, cache_ids, start_ids, mask, active_mask

    def layer(
            self,
            # hidden in layer_builder (from decoder.py)
            hidden,
            # tensors in layer_builder (from decoder.py)
            last_token_id, curr_window_start, pos_embed, cache_ids, start_ids, mask, active_mask,
            # in_caches in layer_builder (from decoder.py)
            attn_k_cache, attn_v_cache,
            # weights in layer_builder (from decoder.py)
            pre_attn_ln_weight, pre_attn_ln_bias,
            # attention
            attn_q_weight, attn_q_scales, attn_q_bias,
            attn_k_weight, attn_k_scales, attn_k_bias,
            attn_v_weight, attn_v_scales, attn_v_bias,
            attn_out_weight, attn_out_scales, attn_out_bias,
            # rms_norm
            post_attn_ln_weight, post_attn_ln_bias,
            pre_mlp_ln_weight, pre_mlp_ln_bias,
            # placeholder
            mlp_in_weight, mlp_in_scales, mlp_in_bias,
            mlp_out_weight, mlp_out_scales, mlp_out_bias,
            post_mlp_ln_weight, post_mlp_ln_bias,
            # gating network and experts for MoE
            expert_indices, gate_weight,
            w1_weight_tp, w1_scales, w2_weight_tp, w2_scales, w3_weight_tp, w3_scales
        ):

        eps = self.config.rms_norm_eps
        is_bsh = self.neuron_config and self.neuron_config.attention_layout == LAYOUT_BSH
        ln_hidden = hlo.rms_norm(hidden, pre_attn_ln_weight, eps) if is_bsh else hlo.rms_norm(hidden, pre_attn_ln_weight, eps, dim=0)
        attn_output, out_attn_k_cache, out_attn_v_cache = self.attention(
            ln_hidden, curr_window_start, cache_ids, start_ids, pos_embed, mask, active_mask,
            attn_k_cache, attn_v_cache,
            attn_q_weight, attn_q_scales, attn_q_bias,
            attn_k_weight, attn_k_scales, attn_k_bias,
            attn_v_weight, attn_v_scales, attn_v_bias,
            attn_out_weight, attn_out_scales, attn_out_bias
        )
        hidden = hlo.add(attn_output, hidden)
        rms_norm_dim = 2 if is_bsh else 0
        norm_hidden = hlo.rms_norm(hidden, pre_mlp_ln_weight, eps, dim=rms_norm_dim)

        final_hidden_states = self.moe_layer(norm_hidden, expert_indices, gate_weight, \
                                             w1_weight_tp, w1_scales, w2_weight_tp, w2_scales, w3_weight_tp, w3_scales)

        res_hidden = hlo.add(final_hidden_states, hidden)
        return res_hidden, out_attn_k_cache, out_attn_v_cache

    def ln_lm_head(self, hidden, last_token_id, rms_weight, unused_bias, lm_head_weight, lm_head_bias, return_all_outputs=True):
        return transformer.rms_lm_head(self.config.tp_degree, hidden, last_token_id, rms_weight, lm_head_weight, lm_head_bias, return_all_outputs, eps=self.config.rms_norm_eps, neuron_config=self.neuron_config)

    def moe_layer(
        self,
        norm_hidden, expert_indices, gate_weight,
        w1_weight_tp, w1_scales, w2_weight_tp, w2_scales, w3_weight_tp, w3_scales
    ):
        is_bsh = self.neuron_config and self.neuron_config.attention_layout == LAYOUT_BSH
        assert not is_bsh, "BSH layout is currently not supported for moe_layer"
        gated_mlp = hlo.gated_mlp_bsh if is_bsh else hlo.gated_mlp

        # Gating network
        dot_dims = dict(lhs_contracting_dimensions=[0], rhs_contracting_dimensions=[0])
        router_logits = hlo.dot_general(gate_weight, norm_hidden, dot_dims)
        routing_weights = hlo.softmax(router_logits, dim=0)
        routing_weights, selected_experts = hlo.topk(routing_weights, k=self.config.num_experts_per_tok, dim=0)

        # Normalize weights of activated experts
        routing_weights_sum = hlo.reduce_sum(routing_weights, dim=0, keepdim=True)
        routing_weights_sum_br = hlo.broadcast(routing_weights_sum, routing_weights.sizes, [0, 1, 2])
        routing_weights = hlo.divide(routing_weights, routing_weights_sum_br)

        # Following expert parallelism implement in https://github.com/vllm-project/vllm/pull/2090
        num_experts_per_core = expert_indices.sizes[0]
        _, intermediate_size = w1_weight_tp.sizes
        slice_size = intermediate_size // num_experts_per_core 
        slice_size_const = hlo.full(slice_size, dtype=expert_indices.dtype, sizes=[])

        local_hidden_states = None
        for idx in range(num_experts_per_core):
            idx_const = hlo.full(idx, dtype=expert_indices.dtype, sizes=[])

            # Slice weight for tp < num_local_experts
            slice_idx_const = hlo.multiply(idx_const, slice_size_const)
            w1_weight = hlo.dynamic_slice_along(w1_weight_tp, dim=1, start=slice_idx_const, size=slice_size)
            w3_weight = hlo.dynamic_slice_along(w3_weight_tp, dim=1, start=slice_idx_const, size=slice_size)
            w2_weight = hlo.dynamic_slice_along(w2_weight_tp, dim=1, start=slice_idx_const, size=slice_size)
            
            # Build expert mask
            expert_idx = hlo.dynamic_slice_along(expert_indices, dim=0, start=idx_const, size=1)
            expert_idx_br = hlo.broadcast(expert_idx, selected_experts.sizes, [0])
            expert_idx_br = hlo.cast(expert_idx_br, selected_experts.dtype)
            expert_mask = hlo.equal(selected_experts, expert_idx_br)
            expert_mask = hlo.cast(expert_mask, routing_weights.dtype)
            expert_weights = hlo.multiply(routing_weights, expert_mask)
            expert_weights = hlo.reduce_sum(expert_weights, dim=0, keepdim=True) # all-reduce across selected experts

            mlp_hidden = gated_mlp(
                norm_hidden,
                w1_weight, w3_weight, w2_weight,
                in0_scales=w1_scales,
                in1_scales=w3_scales,
                out_scales=w2_scales,
                activation_function='silu',
                tp_degree=self.config.tp_degree,
                neuron_config=self.neuron_config,
                return_partial=True,
            )
            # Apply expert weighting
            expert_weights_br = hlo.broadcast(expert_weights, mlp_hidden.sizes, [0, 1, 2])
            current_hidden_states = hlo.multiply(mlp_hidden, expert_weights_br)

            if local_hidden_states is None:
                local_hidden_states = current_hidden_states
            else:
                local_hidden_states = hlo.add(local_hidden_states, current_hidden_states)

        dtype, replica_groups = utils.parse_dtype_replica_groups(self.neuron_config, self.config.tp_degree)
        final_hidden_states = hlo.all_reduce_sum(local_hidden_states, self.config.tp_degree, dtype=dtype, replica_groups=replica_groups)

        return final_hidden_states
