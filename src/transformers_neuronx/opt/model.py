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
from transformers_neuronx import dtypes
from transformers_neuronx import hlo
from transformers_neuronx import module
from transformers_neuronx import ops
from transformers_neuronx import parallel
from transformers_neuronx import sampling
from transformers_neuronx import utils
from transformers_neuronx.decoder import DecoderLmHeadForSamplingNoEmbedding
from transformers_neuronx.opt.config import OPTConfig


class OPTForSampling(module.WrappingCheckpointCompatibleModel):

    def __init__(self, config, batch_size=1, amp=None, tp_degree=2, n_positions=2048,
                 unroll=None, context_length_estimate=None, context_unroll=1, **kwargs):
        if amp is None:
            amp = dtypes.to_amp(config.torch_dtype)
        else:
            warnings.warn(f'torch_dtype={config.torch_dtype} ignored in favor of amp={amp}')
        config = OPTConfig(config, n_positions, batch_size, amp, tp_degree, **kwargs)
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
        attention_head_size = config.hidden_size // config.num_attention_heads
        self.decoder_lm_head = DecoderLmHeadForSamplingNoEmbedding(
            tp_degree, n_positions_list, 1, batch_size, attention_head_size, amp,
            config.num_hidden_layers, unroll,
        )
        start_mask = os.environ.get('NEURON_INTERNAL_ASSUME_ALL_PROMPT_LENGTHS_ARE_EQUAL', None) != '1'
        hlo_builder = OPTForSamplingNoEmbeddingHlo(tp_degree, config.hidden_size,
                                                   config.activation_function, start_mask)
        self.decoder_lm_head.add_inputs_builder(hlo_builder.inputs)
        self.decoder_lm_head.add_layer_builder(hlo_builder.layer)
        self.decoder_lm_head.add_ln_lm_head_builder(hlo_builder.ln_lm_head)
        self.decoder_lm_head_for_context = None
        self.context_length_estimate = context_length_estimate
        self.context_unroll = context_unroll

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
            new_layer.add_attention_output(attn.out_proj.weight.detach().T, attn.out_proj.bias.detach())
            new_layer.add_pre_mlp_layer_norm(layer.final_layer_norm.weight.detach(),
                                             layer.final_layer_norm.bias.detach())
            new_layer.add_mlp_input(layer.fc1.weight.detach().T, layer.fc1.bias.detach())
            new_layer.add_mlp_output(layer.fc2.weight.detach().T, layer.fc2.bias.detach())
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
            self.decoder_lm_head_for_context = self.decoder_lm_head.build_weight_shared(
                n_positions_list=[self.context_length_estimate],
                n_active_tokens=self.context_length_estimate,
                unroll=self.context_unroll,
                share_caches=True,
            )

    def reset(self):
        self.decoder_lm_head.reset()

    def forward(self, input_ids, cache_ids, start_ids=None):
        return self._forward(self.decoder_lm_head, input_ids, cache_ids, start_ids)

    def forward_for_context(self, input_ids, cache_ids, start_ids=None):
        return self._forward(self.decoder_lm_head_for_context, input_ids, cache_ids, start_ids)

    def _forward(self, decoder_lm_head, input_ids, cache_ids, start_ids):
        inputs_embeds = self.chkpt_model.model.decoder.embed_tokens(input_ids)
        position_ids, start_ids = decoder_lm_head.embed_positions_ids(cache_ids, start_ids)
        position_embeds = self.chkpt_model.model.decoder.embed_positions(position_ids)
        hidden = inputs_embeds + position_embeds
        hidden = hidden.transpose(0, -1)
        logits = decoder_lm_head(hidden, cache_ids, start_ids)
        logits = logits.to(torch.float32)
        logits = logits[:self.config.vocab_size]
        logits = logits.transpose(0, -1)
        logits = logits[:, -1, :]
        return logits

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

    def __init__(self, tp_degree, hidden_size, activation_function, start_mask=True):
        self.tp_degree = tp_degree
        self.hidden_size = hidden_size
        self.activation_function = activation_function
        self.start_mask = start_mask
        self.allow_kv_dot_prefetch = os.environ.get('NEURON_INTERNAL_THOMAS_PREFETCH', None) == '1'

    def inputs(self, scribe, hidden_dtype, n_positions, n_active_tokens, batch_size):
        hidden_sizes = self.hidden_size, n_active_tokens, batch_size
        hidden = hidden_dtype[hidden_sizes].Parameter(parameter_number=0)
        cache_ids = scribe.s32[n_active_tokens].Parameter(parameter_number=1)
        start_ids = scribe.s32[batch_size].Parameter(parameter_number=2)
        triu_comparison = 'LT' if self.allow_kv_dot_prefetch else 'LE'
        mask, active_mask = hlo.decoder_attention_mask(start_ids, cache_ids, n_positions,
                                                       triu_comparison, self.allow_kv_dot_prefetch,
                                                       self.start_mask)
        return (hidden, cache_ids, mask, active_mask), (1, 0, None)

    def layer(self, hidden, cache_ids, mask, active_mask, attn_k_cache, attn_v_cache,
              pre_attn_ln_weight, pre_attn_ln_bias, attn_q_weight, attn_q_bias,
              attn_k_weight, attn_k_bias, attn_v_weight, attn_v_bias,
              attn_out_weight, attn_out_bias, post_attn_ln_weight, post_attn_ln_bias,
              pre_mlp_ln_weight, pre_mlp_ln_bias, mlp_in_weight, mlp_in_bias,
              mlp_out_weight, mlp_out_bias, post_mlp_ln_weight, post_mlp_ln_bias):
        dtype = hidden.dtype
        ln_hidden = hlo.layer_norm(hidden, pre_attn_ln_weight, pre_attn_ln_bias)
        attn_output, out_attn_k_cache, out_attn_v_cache = self.attention(
            ln_hidden, cache_ids, mask, active_mask, attn_k_cache, attn_v_cache,
            attn_q_weight, attn_q_bias, attn_k_weight, attn_k_bias,
            attn_v_weight, attn_v_bias, attn_out_weight, attn_out_bias,
        )
        hidden = dtype[hidden.sizes].Add(attn_output, hidden)
        ln_hidden = hlo.layer_norm(hidden, pre_mlp_ln_weight, pre_mlp_ln_bias)
        mlp_hidden = hlo.mlp(ln_hidden, mlp_in_weight, mlp_in_bias, mlp_out_weight, mlp_out_bias,
                             activation_function=self.activation_function, tp_degree=self.tp_degree)
        hidden = dtype[hidden.sizes].Add(mlp_hidden, hidden)
        return hidden, out_attn_k_cache, out_attn_v_cache

    def ln_lm_head(self, hidden, ln_f_weight, ln_f_bias, lm_head_weight, lm_head_bias):
        hidden_size, n_active_tokens, batch_size = hidden.sizes
        dtype = hidden.dtype
        slice_threshold = 2  # 1 doesn't work for now; see P86509517
        if n_active_tokens > slice_threshold:
            slice_dimensions = [
                dict(start=0, limit=hidden_size, stride=1),
                dict(start=n_active_tokens-slice_threshold, limit=n_active_tokens, stride=1),
                dict(start=0, limit=batch_size, stride=1),
            ]
            n_active_tokens = slice_threshold
            sizes = hidden_size, n_active_tokens, batch_size
            hidden = dtype[sizes].Slice(hidden, slice_dimensions=slice_dimensions)
        ln_hidden = hlo.layer_norm(hidden, ln_f_weight, ln_f_bias)
        ln_hidden = dtype[hidden_size,n_active_tokens*batch_size].Reshape(ln_hidden)
        logits = hlo.dot00(lm_head_weight, ln_hidden)
        if lm_head_bias is not None:
            lm_head_bias = dtype[logits.sizes].Broadcast(lm_head_bias, dimensions=[0])
            logits = dtype[logits.sizes].Add(logits, lm_head_bias)
        vocab_size, _ = logits.sizes
        return dtype[vocab_size,n_active_tokens,batch_size].Reshape(logits)

    def attention(self, hidden, cache_ids, mask, active_mask, cached_keys, cached_values,
                  q_weight, q_bias, k_weight, k_bias, v_weight, v_bias, out_weight, out_bias):
        dtype = hidden.dtype
        scribe = hidden.scribe
        f32 = scribe.f32
        pred = scribe.pred
        hidden_size, n_active_tokens, n_seqs = hidden_sizes = hidden.sizes
        max_ctx_plus_n_active_tokens, _, n_heads_tp, d_head = cached_keys.sizes
        attn_size = hidden_size // self.tp_degree
        hidden_r_sizes = hidden_size, n_active_tokens * n_seqs
        active_sizes = n_active_tokens, n_seqs, n_heads_tp, d_head

        hidden_r = dtype[hidden_r_sizes].Reshape(hidden)
        active_q = hlo.dot00_add1(hidden_r, q_weight, q_bias)
        active_q = dtype[active_sizes].Reshape(active_q)

        active_k = hlo.dot00_add1(hidden_r, k_weight, k_bias)
        active_k = dtype[active_sizes].Reshape(active_k)

        active_v = hlo.dot00_add1(hidden_r, v_weight, v_bias)
        active_v = dtype[active_sizes].Reshape(active_v)

        can_skip_scatter = n_active_tokens == max_ctx_plus_n_active_tokens

        # compute self-attention: V x Softmax(QK^T)
        # Keep the attention weights computation in fp32 to avoid overflow issues
        scale_attn = d_head ** 0.5
        scale = dtype.Constant(constant_value=scale_attn)
        scale_attn_br = dtype[active_sizes].Broadcast(scale, dimensions=[])
        active_q = dtype[active_sizes].Divide(active_q, scale_attn_br)
        dot_dims = dict(lhs_contracting_dimensions=[3],
                        lhs_batch_dimensions=[1, 2],
                        rhs_contracting_dimensions=[3],
                        rhs_batch_dimensions=[1, 2])

        scatter_dims = dict(update_window_dims=[1,2,3],
                            inserted_window_dims=[0],
                            scatter_dims_to_operand_dims=[0],
                            index_vector_dim=1)
        assign_func = hlo.gen_assign_func(dtype)
        if self.allow_kv_dot_prefetch:
            active_score_sizes = n_seqs, n_heads_tp, n_active_tokens, n_active_tokens
            active_score = dtype[active_score_sizes].Dot(active_q, active_k, dot_dimension_numbers=dot_dims)
            if active_mask is not None:
                large_neg = dtype.Constant(constant_value=-30000)
                large_neg_br = dtype[active_score_sizes].Broadcast(large_neg, dimensions=[])
                active_mask_br = pred[active_score_sizes].Broadcast(active_mask, dimensions=[0, 3])
                active_score = dtype[active_score_sizes].Select(active_mask_br, active_score, large_neg_br)
            active_score = f32[active_score_sizes].Convert(active_score)
        else:
            if can_skip_scatter:
                cached_keys = active_k
            else:
                cached_keys = dtype[cached_keys.sizes].Scatter(
                    cached_keys, cache_ids, active_k, scatter_dimension_numbers=scatter_dims, to_apply=assign_func)

        score_sizes = n_seqs, n_heads_tp, n_active_tokens, max_ctx_plus_n_active_tokens
        score = dtype[score_sizes].Dot(active_q, cached_keys, dot_dimension_numbers=dot_dims)
        large_neg = dtype.Constant(constant_value=-30000)
        large_neg_br = dtype[score_sizes].Broadcast(large_neg, dimensions=[])
        if len(mask.sizes) == 2:
            mask_br = pred[score_sizes].Broadcast(mask, dimensions=[2, 3])
        else:
            mask_br = pred[score_sizes].Broadcast(mask, dimensions=[0, 2, 3])
        score = dtype[score_sizes].Select(mask_br, score, large_neg_br)
        score = f32[score_sizes].Convert(score)

        dot_dims = dict(lhs_contracting_dimensions=[3],
                        lhs_batch_dimensions=[0, 1],
                        rhs_contracting_dimensions=[0],
                        rhs_batch_dimensions=[1, 2])
        sizes = n_seqs, n_heads_tp, n_active_tokens, d_head

        if self.allow_kv_dot_prefetch:
            # Main logic:
            # 1. Split softmax into exp / sum(exp) where exp is independent along axis 3
            #   probs_post_scatter = softmax(score_post_scatter)
            #   output = probs_post_scatter @ cached_values_post_scatter
            #          = exps_post_scatter / sum(exps_post_scatter) @ cached_values_post_scatter
            #          = exps_post_scatter @ cached_values_post_scatter / sum(exps_post_scatter)
            # 2. Split exps @ cached_values dot into a sum of dots
            #   dot = exps_post_scatter @ cached_values_post_scatter
            #       = exps_pre_scatter @ cached_values_pre_scatter + exp_active @ cached_values_active
            #   Note that exps_pre_scatter @ cached_values_active and exp_active @ cached_values_pre_scatter
            #   are assumed to be 0 due to the use of attention mask.
            reduce_sizes = n_seqs, n_heads_tp, n_active_tokens
            minus_inf = f32.Constant(constant_value=float('-inf'))
            max_func = hlo.gen_max_func(f32)
            reduce_max = f32[reduce_sizes].Reduce(score, minus_inf, dimensions=[3], to_apply=max_func)
            active_reduce_max = f32[reduce_sizes].Reduce(active_score, minus_inf, dimensions=[3], to_apply=max_func)
            reduce_max = f32[reduce_sizes].Maximum(reduce_max, active_reduce_max)
            reduce_max_br = f32[score.sizes].Broadcast(reduce_max, dimensions=[0, 1, 2])
            score_shifted = f32[score.sizes].Subtract(score, reduce_max_br)
            exp = f32[score.sizes].Exp(score_shifted)
            zero = f32.Constant(constant_value=0)
            add_func = hlo.gen_add_func(f32)
            denom = f32[reduce_sizes].Reduce(exp, zero, dimensions=[3], to_apply=add_func)
            exp = dtype[exp.sizes].Convert(exp)
            reduce_max_bra = f32[active_score_sizes].Broadcast(reduce_max, dimensions=[0, 1, 2])
            active_score_shifted = f32[active_score_sizes].Subtract(active_score, reduce_max_bra)
            active_exp = f32[active_score_sizes].Exp(active_score_shifted)
            active_denom = f32[reduce_sizes].Reduce(active_exp, zero, dimensions=[3], to_apply=add_func)
            denom = f32[reduce_sizes].Add(denom, active_denom)
            active_exp = dtype[active_exp.sizes].Convert(active_exp)
            denom = dtype[denom.sizes].Convert(denom)
            output = dtype[sizes].Dot(exp, cached_values, dot_dimension_numbers=dot_dims)
            active_output = dtype[sizes].Dot(active_exp, active_v, dot_dimension_numbers=dot_dims)
            output = dtype[sizes].Add(output, active_output)
            denom_br = dtype[sizes].Broadcast(denom, dimensions=[0, 1, 2])
            output = dtype[sizes].Divide(output, denom_br)
            cached_keys = dtype[cached_keys.sizes].Scatter(
                cached_keys, cache_ids, active_k, scatter_dimension_numbers=scatter_dims, to_apply=assign_func)
            cached_values = dtype[cached_values.sizes].Scatter(
                cached_values, cache_ids, active_v, scatter_dimension_numbers=scatter_dims, to_apply=assign_func)
        else:
            if can_skip_scatter:
                cached_values = active_v
            else:
                cached_values = dtype[cached_values.sizes].Scatter(
                    cached_values, cache_ids, active_v, scatter_dimension_numbers=scatter_dims, to_apply=assign_func)
            probs = hlo.softmax(score)
            probs = dtype[probs.sizes].Convert(probs)
            output = dtype[sizes].Dot(probs, cached_values, dot_dimension_numbers=dot_dims)

        sizes = n_active_tokens, n_seqs, n_heads_tp, d_head
        output = dtype[sizes].Transpose(output, dimensions=[2, 0, 1, 3])

        output_sizes_2d = n_active_tokens*n_seqs, attn_size
        output = dtype[output_sizes_2d].Reshape(output)
        dot_dims = dict(lhs_contracting_dimensions=[0], rhs_contracting_dimensions=[1])
        output = dtype[hidden_r_sizes].Dot(out_weight, output, dot_dimension_numbers=dot_dims)
        out_bias = dtype[hidden_r_sizes].Broadcast(out_bias, dimensions=[0])
        output = dtype[hidden_r_sizes].Add(output, out_bias)
        output = dtype[hidden_sizes].Reshape(output)
        if self.tp_degree == 1:
            return output, cached_keys, cached_values
        replica_groups = [list(range(self.tp_degree))]
        add_func = hlo.gen_add_func(dtype)
        output = dtype[hidden_sizes].AllReduce(output, replica_groups=replica_groups, to_apply=add_func)

        return output, cached_keys, cached_values


class OPTForGreedySearchNoEmbeddingHlo(OPTForSamplingNoEmbeddingHlo):

    def ln_lm_head(self, *args, **kwargs):
        logits = super().ln_lm_head(*args, **kwargs)
        # FIXME: Update to tp_degree=self.tp_degree after AllGather fix
        return hlo.argmax(logits, dim=0, keepdim=True, tp_degree=1)


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
        position_ids, start_ids = self.decoder_lm_head.embed_positions_ids(cache_ids, start_ids)
        position_embeds = self.chkpt_model.model.decoder.embed_positions(position_ids)
        hidden = inputs_embeds + position_embeds
        hidden = hidden.transpose(0, -1)
        tokens = self.decoder_lm_head(hidden, cache_ids, start_ids)
        tokens = tokens.transpose(0, -1)
        tokens = tokens[:, -1, :]
        return tokens

    def sample(self, input_ids, sequence_length, start_ids=None):
        return sampling.sample_tokens(self, input_ids, start_ids, sequence_length)
