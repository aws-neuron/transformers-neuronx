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

from transformers_neuronx import hlo, utils
from transformers_neuronx import constants
from transformers_neuronx import utils
from transformers_neuronx.layers import transformer, rotary, attention, attention_utils, flash_decoding
from transformers_neuronx.llama.config import LlamaConfig
from transformers_neuronx.config import NeuronConfig
from transformers_neuronx.constants import LAYOUT_BSH, LAYOUT_HSB
from transformers_neuronx.hlo import quantize_kv_cache_direct_cast, dequantize_kv_cache_direct_cast

from transformers_neuronx.nki.compile import nki_call

import logging


class LlamaForSamplingNoEmbeddingHlo:

    def __init__(self,
        config: LlamaConfig,
        neuron_config: Optional[NeuronConfig] = None
    ):
        self.config = config
        self.neuron_config = neuron_config
        self.n_positions = None
        self.num_active_blocks = None

    @property
    def shard_over_batch(self):
        # Property access allows fallback configuration to be enabled after construction
        return (
            self.neuron_config is not None
            and self.neuron_config.group_query_attention == constants.GQA.SHARD_OVER_BATCH
        )

    def inputs(self, scribe, dtype, n_active_tokens, batch_size):
        tensors, dims = transformer.inputs(
            scribe, dtype, batch_size, n_active_tokens, self.config.hidden_size, self.neuron_config, self.config.tp_degree)

        return tensors, dims

    def token_tree_inputs(self, scribe, dtype, n_active_tokens, batch_size):
        tensors, dims = self.inputs(scribe, dtype, n_active_tokens, batch_size)
        s32 = scribe.s32
        cache_2d = self.neuron_config and self.neuron_config.use_2d_cache_ids
        # Allow tree based speculation inputs
        if cache_2d:
            position_sizes = batch_size, n_active_tokens
            previous_cache_ids = s32[position_sizes].Parameter(parameter_number=4)
            reorder_mapping = s32[position_sizes].Parameter(parameter_number=5)
        else:
            previous_cache_ids = s32[n_active_tokens].Parameter(parameter_number=4)
            reorder_mapping = s32[n_active_tokens].Parameter(parameter_number=5)
        seq_slice_dim = 1 if cache_2d else 0

        return (*tensors, previous_cache_ids, reorder_mapping), (*dims, seq_slice_dim, seq_slice_dim)

    def eagle_draft_inputs(self, scribe, dtype, n_active_tokens, batch_size, token_tree=False, k=0, n_leaves=0, depth=0, n_entrees=0, width=0):
        # Should use token_tree_inputs
        tensors, dims = self.inputs(scribe, dtype, n_active_tokens, batch_size)
        hidden_sizes = batch_size, n_active_tokens, self.config.hidden_size
        prev_hidden = dtype[hidden_sizes].Parameter(parameter_number=4)
        if not token_tree:
            return (*tensors, prev_hidden), (*dims, 1)
        s32 = scribe.s32
        tree_mask_sizes = k, k
        tree_mask = s32[tree_mask_sizes].Parameter(parameter_number=5)
        indices_sizes = batch_size, k-1
        update_indices = s32[indices_sizes].Parameter(parameter_number=6)
        hidden_update_sizes = batch_size, k-1
        hidden_update_indices = s32[hidden_update_sizes].Parameter(parameter_number=7)
        cache_update_sizes = batch_size, depth
        cache_gather_indices = s32[cache_update_sizes].Parameter(parameter_number=8)
        cache_scatter_indices = s32[cache_update_sizes].Parameter(parameter_number=9)
        pos_sizes = batch_size, k
        position_ids = s32[pos_sizes].Parameter(parameter_number=10)
        path_sizes = n_leaves, depth
        all_paths = s32[path_sizes].Parameter(parameter_number=11)
        mask_sizes = n_entrees, width
        path_mask = s32[mask_sizes].Parameter(parameter_number=12)
        return (*tensors, 
                prev_hidden, 
                tree_mask, 
                update_indices, 
                hidden_update_indices, 
                cache_gather_indices, 
                cache_scatter_indices, 
                position_ids, 
                all_paths, 
                path_mask), (*dims, 1, 1, 1, 1, 1, 1, 1, 1, 1)

    def embedding(self, input_ids, cache_ids, start_ids, last_token_id, *weights):
        core_id = None
        if ((self.neuron_config.shard_over_sequence or self.neuron_config.sequence_parallel_norm)
                and self.neuron_config.on_device_embedding):
            core_id, embed_weight = weights
        else:
            embed_weight, *rst = weights
        dtype = getattr(input_ids.scribe, self.config.amp)
        if self.neuron_config.on_device_embedding and self.neuron_config.sequence_parallel_norm:
            hidden = hlo.embedding(embed_weight, input_ids,
                                   tp_degree=self.config.tp_degree, dim=0,
                                   dtype=dtype, core_id=core_id,
                                   sequence_parallel=self.neuron_config.is_sequence_parallel)
        else:
            hidden = hlo.embedding(embed_weight, input_ids, tp_degree=self.config.tp_degree, dtype=dtype)
        if self.config.hidden_size % self.config.tp_degree != 0:
            hidden = hlo.slice_along(hidden, dim=-1, limit=self.config.hidden_size, start=0)
        if self.neuron_config.attention_layout == LAYOUT_HSB:
            hidden = hlo.transpose210(hidden)
        return hidden

    def token_tree_embedding(self, input_ids, cache_ids, start_ids, last_token_id, previous_cache_ids, reorder_mapping,
                             *weights):
        return self.embedding(input_ids, cache_ids, start_ids, last_token_id, *weights)

    def pre_layer(self, hidden, cache_ids, start_ids, last_token_id, *weights, position_ids=None):
        # TODO: move this fallback calculation to decoder.py
        if self.num_active_blocks is None and self.neuron_config.optimized_paged_attention:
            max_model_len = self.neuron_config.continuous_batching.max_model_len
            max_num_seqs = self.neuron_config.continuous_batching.max_num_seqs
            block_size = self.neuron_config.continuous_batching.block_size
            self.num_active_blocks = (max_model_len * max_num_seqs // block_size) - 2

        if self.neuron_config.optimized_paged_attention and len(last_token_id.sizes) == 2:
            # For decoding with multiple KV cache blocks:
            # - cache_ids are used as context_lens
            # - start_ids are used as slot_mapping
            # - last_token_id is used as block_tables
            # The function below transforms 2D block_tables into 1D active block table
            last_token_id = attention_utils.active_block_tables(
                block_tables=last_token_id, context_lens=cache_ids,
                num_active_blocks=self.num_active_blocks, neuron_config=self.neuron_config)
            max_num_seqs = self.neuron_config.continuous_batching.max_num_seqs
            block_size = self.neuron_config.continuous_batching.block_size
            block_to_seq = attention_utils.block_to_seq_indexing(
                context_lens=cache_ids, num_seqs=max_num_seqs, num_blocks=self.num_active_blocks, block_size=block_size)
        else:
            block_to_seq = None

        head_dim = self.config.attention_head_size
        position_ids = cache_ids if position_ids is None else position_ids
        pos_embed = rotary.hlo_rotary_embedding(
            hidden.dtype, int(head_dim * self.config.rotary_percentage), position_ids,
            base=self.config.rope_theta,
            interpolation_factor=self.config.position_interpolation_factor,
            rope_scaling=self.config.rope_scaling
        )
        core_id = None
        if (self.neuron_config.shard_over_sequence or
                (self.neuron_config.sequence_parallel_norm and self.neuron_config.on_device_embedding)):
            core_id, *rst = weights
        # flash decoding 
        if self.neuron_config.shard_over_sequence:
            n_kv_heads = self.config.num_key_value_heads if hasattr(self.config, "num_key_value_heads") else self.config.num_attention_heads
            cores_per_kv_head = self.config.tp_degree // n_kv_heads
            self.cores_per_kv_head  = cores_per_kv_head if cores_per_kv_head > 1 else self.config.tp_degree 
            cache_ids, mask, active_mask = flash_decoding.convert_attn_mask_and_cache_id(cache_ids, start_ids,
                                                                        core_id, self.n_positions,
                                                                        cores_per_kv_head=self.cores_per_kv_head)
        else:
            mask, active_mask = hlo.attention_mask(cache_ids, start_ids, self.n_positions, 
                                                   last_token_id=last_token_id, num_active_blocks=self.num_active_blocks, neuron_config=self.neuron_config)

        return hidden, last_token_id, pos_embed, cache_ids, start_ids, block_to_seq, mask, active_mask, core_id

    def token_tree_pre_layer(self, hidden, cache_ids, start_ids, last_token_id, previous_cache_ids, reorder_mapping, *weights):
        hidden, last_token_id, pos_embed, cache_ids, start_ids, block_to_seq, mask, active_mask, core_id = self.pre_layer(hidden, cache_ids, start_ids, last_token_id, *weights)
        if self.neuron_config.on_device_embedding:
            embed_weight, token_tree_mask = weights
        else:
            token_tree_mask, *rst = weights
        active_mask = hlo.token_tree_attention_mask(token_tree_mask, active_mask)
        return hidden, last_token_id, pos_embed, cache_ids, start_ids, block_to_seq, previous_cache_ids, reorder_mapping, mask, active_mask, core_id

    def eagle_draft_pre_layer(self, hidden, cache_ids, start_ids, last_token_id, *weights, position_ids=None):
        if self.config.bias:
            embed_weight, fc_weight, fc_bias, *rst = weights
        else:
            embed_weight, fc_weight, *rst = weights
            fc_bias = None
        hidden = hlo.dot_add(fc_weight, hidden, fc_bias, 0, 2, 0)
        hidden = hlo.permute(hidden, [1, 2, 0])
        hidden = hlo.all_gather(hidden, 2, self.config.tp_degree) 
        #hidden = hlo.dot_add(hidden, fc_weight, fc_bias, 2, 0, 2)
        return self.pre_layer(hidden, cache_ids, start_ids, last_token_id, *weights, position_ids=position_ids)
    
    def layer(self, hidden, last_token_id, pos_embed, cache_ids, start_ids, block_to_seq, mask, active_mask, core_id,
            attn_k_cache, attn_v_cache,
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
            in0_weight=None, in0_scales=None,
            in1_weight=None, in1_scales=None,
            out_weight=None, out_scales=None,
            is_first_last_layer=False,
        ):
        local_args = {**locals()}
        local_args.pop('self')

        # Initialize with kernels
        enable_qkv_kernel, enable_mlp_kernel = False, False
        if self.neuron_config and self.neuron_config.fused_rmsnorm_qkv:
            try:
                from neuronxcc.nki._private_kernels.qkv import rmsnorm_qkv_isa_fused_add_kernel
                enable_qkv_kernel = True
            except:
                logging.warning("No QKV kernel found")
        if self.neuron_config and self.neuron_config.fused_rmsnorm_mlp:
            try:
                from neuronxcc.nki._private_kernels.mlp import mlp_isa_kernel
                enable_mlp_kernel = True
            except:
                logging.warning("No MLP kernel found")
            enable_mlp_kernel = True

        if (not enable_qkv_kernel and not enable_mlp_kernel) or active_mask != None:
            return self.flat_compiler_layer(**local_args)

        local_args['enable_qkv_kernel'] = enable_qkv_kernel
        local_args['enable_mlp_kernel'] = enable_mlp_kernel
        return self.native_kernel_layer(**local_args)


    def flat_compiler_layer(
            self, hidden, last_token_id, pos_embed, cache_ids, start_ids, block_to_seq, mask, active_mask, core_id,
            attn_k_cache, attn_v_cache,
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
            in0_weight=None, in0_scales=None,
            in1_weight=None, in1_scales=None,
            out_weight=None, out_scales=None,
            is_first_last_layer=False,
        ):
        eps = self.config.rms_norm_eps
        is_bsh = self.neuron_config and self.neuron_config.attention_layout == LAYOUT_BSH
        if not self.neuron_config.is_eagle_draft:
            ln_hidden = hlo.rms_norm(hidden, pre_attn_ln_weight, eps, neuron_config=self.neuron_config, tp_degree=self.config.tp_degree) if is_bsh else hlo.rms_norm(hidden, pre_attn_ln_weight, eps, dim=0, neuron_config=self.neuron_config, tp_degree=self.config.tp_degree)
        else:
            ln_hidden = hidden
        attn_output, out_attn_k_cache, out_attn_v_cache = self.attention(
            ln_hidden, cache_ids, start_ids, last_token_id, block_to_seq, pos_embed, mask, active_mask, core_id,
            attn_k_cache, attn_v_cache,
            attn_q_weight, attn_q_scales, attn_q_bias,
            attn_k_weight, attn_k_scales, attn_k_bias,
            attn_v_weight, attn_v_scales, attn_v_bias,
            attn_out_weight, attn_out_scales, attn_out_bias
        )
        hidden = hlo.add(attn_output, hidden)
        gated_mlp = hlo.gated_mlp_bsh if is_bsh else hlo.gated_mlp
        rms_norm_dim = 2 if is_bsh else 0
        norm_hidden = hlo.rms_norm(hidden, pre_mlp_ln_weight, eps, dim=rms_norm_dim, neuron_config=self.neuron_config, tp_degree=self.config.tp_degree)
        if self.neuron_config.fuse_mlp:
            assert all(map(lambda x: not(x), [in0_weight, in1_weight, out_weight, in0_scales, in1_scales, out_scales])) ,\
                f"in0, in1 and out weights have to be None"
            in0_weight, in0_scales = mlp_in_weight, mlp_in_scales
            out_weight, out_scales = mlp_out_weight, mlp_out_scales

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

    def native_kernel_layer(
            self, hidden, last_token_id, pos_embed, cache_ids, start_ids, block_to_seq, mask, active_mask, core_id,
            attn_k_cache, attn_v_cache,
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
            in0_weight=None, in0_scales=None,
            in1_weight=None, in1_scales=None,
            out_weight=None, out_scales=None,
            is_first_last_layer=False,
            enable_qkv_kernel=False,
            enable_mlp_kernel=False
    ):
        eps = self.config.rms_norm_eps
        is_bsh = self.neuron_config and self.neuron_config.attention_layout == LAYOUT_BSH
        assert is_bsh
        rms_norm_dim = 2 if is_bsh else 0

        from neuronxcc.nki._private_kernels.mlp import mlp_isa_kernel, mlp_fused_add_isa_kernel
        # lambda functions for calling kernels
        def _mlp_fused_add_kernel(attn_output, hidden, ln_w, gate_w, up_w, down_w, out, fused_rmsnorm=True):
            mlp_fused_add_isa_kernel(attn_output, hidden, ln_w, gate_w, up_w, down_w, out, "MLP", fused_rmsnorm=fused_rmsnorm)
            
        def _mlp_kernel(hidden, ln_w, gate_w, up_w, down_w, out, fused_rmsnorm=False):
            mlp_isa_kernel(hidden, ln_w, gate_w, up_w, down_w, out, "MLP", fused_rmsnorm=fused_rmsnorm)

        if enable_qkv_kernel:
            assert fused_pre_attn_ln_qkv_weight is not None
            fused_out = self.fused_rmsnorm_qkv(
                hidden, None, eps,
                cache_ids, start_ids, last_token_id, block_to_seq, pos_embed, mask, active_mask, core_id,
                attn_k_cache, attn_v_cache,
                fused_pre_attn_ln_qkv_weight, attn_q_scales, attn_q_bias,
                attn_k_weight, attn_k_scales, attn_k_bias, # should be none
                attn_v_weight, attn_v_scales, attn_v_bias, # should be none
                attn_out_weight, attn_out_scales, attn_out_bias
            )
            if len(fused_out) == 3:
                attn_output, out_attn_k_cache, out_attn_v_cache = fused_out
            else:
                attn_output, out_attn_k_cache, out_attn_v_cache, fused_added_hidden = fused_out
        else:
            ln_hidden = hlo.rms_norm(hidden, pre_attn_ln_weight, eps, neuron_config=self.neuron_config, tp_degree=self.config.tp_degree) if is_bsh else hlo.rms_norm(hidden, pre_attn_ln_weight, eps, dim=0, neuron_config=self.neuron_config, tp_degree=self.config.tp_degree)
            attn_output, out_attn_k_cache, out_attn_v_cache = self.attention(
                ln_hidden, cache_ids, start_ids, last_token_id, block_to_seq, pos_embed, mask, active_mask, core_id,
                attn_k_cache, attn_v_cache,
                attn_q_weight, attn_q_scales, attn_q_bias,
                attn_k_weight, attn_k_scales, attn_k_bias,
                attn_v_weight, attn_v_scales, attn_v_bias,
                attn_out_weight, attn_out_scales, attn_out_bias
            )

        if isinstance(hidden, tuple):
            hidden = hidden[0]

        if enable_mlp_kernel:
            if self.neuron_config.is_sequence_parallel:
                # In sequence parallel, we cannot fuse residual add and rms norm into the kernel
                hidden = hlo.add(attn_output, hidden)
                norm_hidden = hlo.rms_norm(hidden, pre_mlp_ln_weight, eps, dim=rms_norm_dim, neuron_config=self.neuron_config, tp_degree=self.config.tp_degree)
                mlp_result = nki_call(_mlp_kernel, norm_hidden, pre_mlp_ln_weight, in0_weight, in1_weight, out_weight,
                                    output_HloShapes=[norm_hidden.dtype[norm_hidden.sizes[0], norm_hidden.sizes[1], norm_hidden.sizes[2]]])
                dtype, replica_groups = utils.parse_dtype_replica_groups(self.neuron_config, self.config.tp_degree)
                mlp_hidden = hlo.reduce_scatter_sum(mlp_result, tp_degree=self.config.tp_degree, dim=1, replica_groups=replica_groups, dtype=dtype)
                return hlo.add(mlp_hidden, hidden), out_attn_k_cache, out_attn_v_cache

            # In TP, we can fuse residual add and rms norm into the kernel
            if is_first_last_layer or not enable_qkv_kernel:
                hidden_add = hlo.add(attn_output, hidden)
            mlp_result = nki_call(_mlp_fused_add_kernel, attn_output, hidden, pre_mlp_ln_weight, in0_weight, in1_weight, out_weight, 
                                 output_HloShapes=[hidden.dtype[hidden.sizes[0], hidden.sizes[1], hidden.sizes[2]]])
            dtype, replica_groups = utils.parse_dtype_replica_groups(self.neuron_config, self.config.tp_degree)
            mlp_hidden = hlo.all_reduce_sum(mlp_result, self.config.tp_degree, dtype=dtype, replica_groups=replica_groups)
            if is_first_last_layer or not enable_qkv_kernel:
                return hlo.add(mlp_hidden, hidden_add), out_attn_k_cache, out_attn_v_cache

            return (hidden, mlp_hidden, attn_output), out_attn_k_cache, out_attn_v_cache
        else:
            hidden = hlo.add(attn_output, hidden)
            gated_mlp = hlo.gated_mlp_bsh if is_bsh else hlo.gated_mlp
            norm_hidden = hlo.rms_norm(hidden, pre_mlp_ln_weight, eps, dim=rms_norm_dim, neuron_config=self.neuron_config, tp_degree=self.config.tp_degree)
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
            if is_first_last_layer or not enable_qkv_kernel:
                return hlo.add(mlp_hidden, hidden), out_attn_k_cache, out_attn_v_cache
            return (hidden, mlp_hidden, attn_output), out_attn_k_cache, out_attn_v_cache

    def token_tree_layer(
            self, hidden, last_token_id, pos_embed, cache_ids, start_ids, block_to_seq,
            previous_cache_ids, reorder_mapping,
            mask, active_mask, core_id,
            attn_k_cache, attn_v_cache,
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
            in0_weight, in0_scales,
            in1_weight, in1_scales,
            out_weight, out_scales,
            is_first_last_layer=False,
    ):
        eps = self.config.rms_norm_eps
        is_bsh = self.neuron_config and self.neuron_config.attention_layout == LAYOUT_BSH
        ln_hidden = hlo.rms_norm(hidden, pre_attn_ln_weight, eps, neuron_config=self.neuron_config, tp_degree=self.config.tp_degree) if is_bsh else hlo.rms_norm(hidden, pre_attn_ln_weight, eps, dim=0, neuron_config=self.neuron_config, tp_degree=self.config.tp_degree)
        reordered_attn_k_cache, reordered_attn_v_cache = attention.reorder_kv_cache(attn_k_cache, attn_v_cache, previous_cache_ids, reorder_mapping, neuron_config=self.neuron_config)
        attn_output, out_attn_k_cache, out_attn_v_cache = self.attention(
            ln_hidden, cache_ids, start_ids, last_token_id, block_to_seq, pos_embed, mask, active_mask, core_id,
            reordered_attn_k_cache, reordered_attn_v_cache,
            attn_q_weight, attn_q_scales, attn_q_bias,
            attn_k_weight, attn_k_scales, attn_k_bias,
            attn_v_weight, attn_v_scales, attn_v_bias,
            attn_out_weight, attn_out_scales, attn_out_bias
        )
        hidden = hlo.add(attn_output, hidden)
        gated_mlp = hlo.gated_mlp_bsh if is_bsh else hlo.gated_mlp
        rms_norm_dim = 2 if is_bsh else 0
        norm_hidden = hlo.rms_norm(hidden, pre_mlp_ln_weight, eps, dim=rms_norm_dim, neuron_config=self.neuron_config, tp_degree=self.config.tp_degree)
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

    def fused_rmsnorm_qkv(
        self, hidden, pre_attn_ln_weight, eps,
        cache_ids, start_ids, last_token_id, block_to_seq, pos_embed, mask, active_mask, core_id,
        attn_k_cache, attn_v_cache,
        attn_q_weight, attn_q_scales, attn_q_bias,
        attn_k_weight, attn_k_scales, attn_k_bias, # should be none
        attn_v_weight, attn_v_scales, attn_v_bias, # should be none
        attn_out_weight, attn_out_scales, attn_out_bias
    ):
        from neuronxcc.nki._private_kernels.qkv import rmsnorm_qkv_isa_kernel, rmsnorm_qkv_isa_fused_add_kernel
        def _kernel(h, w, output):
            return rmsnorm_qkv_isa_kernel(h, w, output, "QKV")

        def _fused_out_kernel(h0, h1, h2, w, output):
            # This kernel will perform h0 = h0 + h1 + h2 (writing results in-place to an input buffer
            # FIXME: allow for multiple outputs
            return rmsnorm_qkv_isa_fused_add_kernel(h0, h1, h2, w, output, "QKV")

        fused_add = False
        if isinstance(hidden, tuple):
            fused_add = True
            hidden, mlp_out, attn_out = hidden

        n_seqs, n_active_tokens, _ = hidden.sizes
        d_head = self.config.attention_head_size
        tp_degree = self.config.tp_degree

        # Compute the expected number of KV heads (Used in case fused QKV is used)
        n_kv_heads_tp = None
        if self.config.num_key_value_heads is not None:
            n_head = self.config.num_attention_heads
            n_kv_head = self.config.num_key_value_heads
            n_head, n_kv_head_padded = utils.get_qkv_padding(n_head, n_kv_head, tp_degree, self.neuron_config)
            n_kv_heads_tp = n_kv_head_padded // tp_degree

        _, hidden_size_tp = attn_q_weight.sizes

        n_total_heads_tp = hidden_size_tp // d_head
        n_heads_tp = n_total_heads_tp - 2 * n_kv_heads_tp

        if fused_add:
            nki_output = nki_call(_fused_out_kernel,
                                hidden, mlp_out, attn_out, attn_q_weight,
                                output_HloShapes=[hidden.dtype[n_seqs, n_active_tokens, hidden_size_tp]])
        else:
            nki_output = nki_call(_kernel,
                                hidden, attn_q_weight,
                                output_HloShapes=[hidden.dtype[n_seqs, n_active_tokens, hidden_size_tp]])
        slice_lim = nki_output.sizes[-1] // (n_heads_tp + 2 * n_kv_heads_tp)
        query = hlo.slice_along(nki_output, -1, n_heads_tp*slice_lim, start=0)
        key = hlo.slice_along(nki_output, -1, (n_heads_tp+n_kv_heads_tp)*slice_lim, start=n_heads_tp*slice_lim)
        value = hlo.slice_along(nki_output, -1, (n_heads_tp+2*n_kv_heads_tp)*slice_lim, start=(n_heads_tp+n_kv_heads_tp)*slice_lim)

        # shard over head (llama/hlo.py)
        active_q_sizes = n_active_tokens, n_seqs, n_heads_tp, d_head
        active_kv_sizes = n_active_tokens, n_seqs, n_kv_heads_tp, d_head
        query = hlo.reshape(query, active_q_sizes)
        key = hlo.reshape(key, active_kv_sizes)
        value = hlo.reshape(value, active_kv_sizes)
        assert all([attn_q_scales is None,
                    attn_q_bias is None,
                    attn_k_weight is None,
                    attn_k_scales is None,
                    attn_k_bias is None,
                    attn_v_weight is None,
                    attn_v_scales is None,
                    attn_v_bias is None])

        # Pass QKV tuple since it will not be computed in the attention block
        attn_output, out_attn_k_cache, out_attn_v_cache = self.attention(
            nki_output, cache_ids, start_ids, last_token_id, block_to_seq, pos_embed, mask, active_mask, core_id,
            attn_k_cache, attn_v_cache,
            attn_q_weight, None, None,
            None, None, None,
            None, None, None,
            attn_out_weight, attn_out_scales, attn_out_bias,
            qkv_tuple=(query, key, value),
        )
        if fused_add:
            return attn_output, out_attn_k_cache, out_attn_v_cache, hidden
        return attn_output, out_attn_k_cache, out_attn_v_cache


    def attention(
        self,
        hidden, cache_ids, start_ids, last_token_id, block_to_seq, pos_embed, mask, active_mask, core_id,
        cached_keys, cached_values,
        q_weight, q_scales, q_bias,
        k_weight, k_scales, k_bias,
        v_weight, v_scales, v_bias,
        out_weight, out_scales, out_bias,
        qkv_tuple: tuple = None,
    ):
        d_head = self.config.attention_head_size
        tp_degree = self.config.tp_degree

        # Compute the expected number of KV heads (Used in case fused QKV is used)
        n_kv_heads_tp = None
        if self.config.num_key_value_heads is not None:
            n_head = self.config.num_attention_heads
            n_kv_head = self.config.num_key_value_heads
            n_head, n_kv_head_padded = utils.get_qkv_padding(n_head, n_kv_head, tp_degree, self.neuron_config)
            n_kv_heads_tp = n_kv_head_padded // tp_degree

        # Q = (hidden @ wQ) + bQ
        # K = (hidden @ wK) + bK
        # V = (hidden @ wV) + bV
        if qkv_tuple: 
            # If computed already, skip computation here
            assert active_mask is None
            query, key, value = qkv_tuple
        else:
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
                    # TODO: revert to hlo.index_select once its faster P126527643
                    cached_keys_s = hlo.select(cached_keys, batch_dim, hlo.reshape(start_ids, slice_sizes), keepdim=True)
                    cached_values_s = hlo.select(cached_values, batch_dim, hlo.reshape(start_ids, slice_sizes), keepdim=True)
                elif cached_keys.sizes[batch_dim] == start_ids.sizes[0]:
                    # For batched speculative decoding, we will select kv caches for all sequences. No need to do
                    # index select, which is slow
                    cached_keys_s = cached_keys
                    cached_values_s = cached_values
                else:
                    # for multi prompt use case, cached_keys.sizes[batch_dim] can still be larger than 1, so we
                    # need to use start_ids size to determine if we want to select kv cache.
                    cached_keys_s = hlo.index_select(cached_keys, batch_dim, start_ids)
                    cached_values_s = hlo.index_select(cached_values, batch_dim, start_ids)
                if self.neuron_config and self.neuron_config.kv_cache_quant:
                    cached_keys_s = dequantize_kv_cache_direct_cast(cached_keys_s, self.neuron_config)
                    cached_values_s = dequantize_kv_cache_direct_cast(cached_values_s, self.neuron_config)
            elif self.neuron_config and self.neuron_config.paged_attention:
                # For decoding with multiple KV cache blocks, start_ids are used as block_tables
                cached_keys_s = attention_utils.gather_blocks(cached_keys, block_tables=last_token_id, neuron_config=self.neuron_config)
                cached_values_s = attention_utils.gather_blocks(cached_values, block_tables=last_token_id, neuron_config=self.neuron_config)
                if self.neuron_config and self.neuron_config.kv_cache_quant:
                    cached_keys_s = dequantize_kv_cache_direct_cast(cached_keys_s, self.neuron_config)
                    cached_values_s = dequantize_kv_cache_direct_cast(cached_values_s, self.neuron_config)
            elif self.neuron_config and self.neuron_config.kv_cache_quant:
                cached_keys_s = dequantize_kv_cache_direct_cast(cached_keys, self.neuron_config)
                cached_values_s = dequantize_kv_cache_direct_cast(cached_values, self.neuron_config)   
            else:
                cached_keys_s = cached_keys
                cached_values_s = cached_values
            # Communication 1: all-gather query from cores
            if (n_active_tokens != self.n_positions) and self.neuron_config.shard_over_sequence:
                query = flash_decoding.gather_query_group(query, self.cores_per_kv_head, 
                                                  n_head,
                                                  tp_degree)

            # Sp = Q @ Kp
            prior_scores = attention.score(query, cached_keys_s, n_kv_heads=self.config.num_key_value_heads,
                                           tp_degree=tp_degree, block_to_seq=block_to_seq, neuron_config=self.neuron_config)
            prior_scores = attention.mask(prior_scores, mask, tp_degree=tp_degree, shard_over_batch=self.shard_over_batch)

            # Sa = Q @ Ka
            active_score = attention.score(query, key, n_kv_heads=self.config.num_key_value_heads,
                                           tp_degree=tp_degree, neuron_config=self.neuron_config)
            active_score = attention.mask(active_score, active_mask, tp_degree=tp_degree, shard_over_batch=self.shard_over_batch)

            # C = softmax(Sa, Sp) @ (Va, Vp)
            if self.neuron_config.shard_over_sequence:
                dtype = query.dtype
                context = flash_decoding.context(prior_scores, active_score, cached_values_s, value, core_id, mask,
                                                 active_mask,
                                                 n_kv_heads=self.config.num_key_value_heads, n_heads=n_head,
                                                 dtype=dtype,
                                                 tp_degree=tp_degree, neuron_config=self.neuron_config,
                                                 shard_over_batch=self.shard_over_batch)
                cache_ids, value, key = flash_decoding.select_values_within_bound(cache_ids, value, key,
                                                                                  self.cores_per_kv_head,
                                                                                  core_id,
                                                                                  dim=0, n_positions=self.n_positions)

            else:
                context = attention.context(prior_scores, active_score, cached_values_s, value,
                                            n_kv_heads=self.config.num_key_value_heads, tp_degree=tp_degree,
                                            context_lens=cache_ids, num_active_blocks=self.num_active_blocks,
                                            block_to_seq=block_to_seq,
                                            neuron_config=self.neuron_config)

            # KCache[I], VCache[I] = K, V
            updated_keys, updated_values = attention.fused_kv_update_cache(cached_keys, cached_values, cache_ids,
                                                                           key, value, start_ids, neuron_config=self.neuron_config)

        # Multi-Token Context Encoding
        else:
            _, batch_size, _, _ = query.sizes
            if self.neuron_config.lhs_aligned or batch_size == 1:
                context = attention.flash_attention(query, key, value)
            else:
                # do not use flash attention for lhs padded (right aligned) batch > 1 case
                # because it does not correctly take mask into account 
                context = None

            if context is None:
                # S = Q @ K

                score = attention.score(query, key, n_kv_heads=self.config.num_key_value_heads,
                                        tp_degree=tp_degree, neuron_config=self.neuron_config)
                score = attention.mask(score, mask, tp_degree=tp_degree, shard_over_batch=self.shard_over_batch)
                context = attention.context_combined(score, value, n_kv_heads=self.config.num_key_value_heads,
                                                    tp_degree=tp_degree, neuron_config=self.neuron_config)

            if self.neuron_config.shard_over_sequence:
                cache_ids, value, key = flash_decoding.select_values_within_bound(cache_ids,
                                                                                  value,
                                                                                  key,
                                                                                  self.cores_per_kv_head,
                                                                                  core_id, dim=0,
                                                                                  n_positions=self.n_positions)
            # KCache, VCache = K, V
            if cached_keys.sizes == key.sizes:
                if self.neuron_config and self.neuron_config.kv_cache_quant:
                    updated_keys = quantize_kv_cache_direct_cast(key, self.neuron_config)
                    updated_values = quantize_kv_cache_direct_cast(value, self.neuron_config)
                else:
                    updated_keys, updated_values = key, value
            else:
                updated_keys, updated_values = attention.fused_kv_update_cache(cached_keys, cached_values, cache_ids,
                                                                               key, value, start_ids, neuron_config=self.neuron_config)

        # O = (C @ wO) + bO
        output = attention.output(context, out_weight, out_scales, out_bias, tp_degree, self.neuron_config)
        return output, updated_keys, updated_values


