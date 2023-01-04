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
from transformers_neuronx import dtypes
from transformers_neuronx import hlo
from transformers_neuronx import module
from transformers_neuronx import ops
from transformers_neuronx import utils
from transformers_neuronx.decoder import DecoderLmHeadForSamplingNoEmbedding
from transformers_neuronx.opt.config import OPTConfig
from transformers_neuronx.sampling import simple_sample


class OPTForSampling(module.PretrainedModel):

    def __init__(self, config, batch_size=1, amp='f32', tp_degree=2, n_positions=2048,
                 unroll=None, init_n_active_tokens=None, **kwargs):
        super().__init__()
        config = OPTConfig(config, n_positions, batch_size, amp, tp_degree, **kwargs)
        self.chkpt_model = OPTCheckpointCompatible(config)
        self.config = config
        if unroll is None:
            unroll = config.num_hidden_layers
        n_positions_list = utils.power_of_two_bucket_sizes(128, n_positions)
        attention_head_size = config.hidden_size // config.num_attention_heads
        self.decoder_lm_head = DecoderLmHeadForSamplingNoEmbedding(
            tp_degree, n_positions_list, 1, batch_size, attention_head_size, amp,
            config.num_hidden_layers, unroll,
        )
        hlo_builder = OPTForSamplingNoEmbeddingHlo(tp_degree, config.hidden_size, config.activation_function)
        self.decoder_lm_head.add_inputs_builder(hlo_builder.inputs)
        self.decoder_lm_head.add_layer_builder(hlo_builder.layer)
        self.decoder_lm_head.add_ln_lm_head_builder(hlo_builder.ln_lm_head)

    def load_state_dict_dir(self, state_dict_dir):
        self.chkpt_model.load_state_dict_dir(state_dict_dir)

    def load_state_dict_low_memory(self, state_dict):
        self.chkpt_model.load_state_dict_low_memory(state_dict)

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

    def reset(self):
        self.decoder_lm_head.reset()

    def forward(self, input_ids, position_ids):
        inputs_embeds = self.chkpt_model.model.decoder.embed_tokens(input_ids)
        position_embeds = self.chkpt_model.model.decoder.embed_positions(position_ids)
        hidden = inputs_embeds + position_embeds
        hidden = hidden.transpose(0, -1)
        logits = self.decoder_lm_head(hidden, position_ids)
        logits = logits.to(torch.float32)
        logits = logits[:self.config.vocab_size]
        logits = logits.transpose(0, -1)
        logits = logits[:, -1, :]
        return logits

    def sample(self, input_ids, sequence_length):
        return simple_sample(self, input_ids, sequence_length, self.config.n_positions,
                             eos_token_id=self.config.eos_token_id, top_k=50)


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

    def __init__(self, tp_degree, hidden_size, activation_function):
        self.tp_degree = tp_degree
        self.hidden_size = hidden_size
        self.activation_function = activation_function

    def inputs(self, scribe, hidden_dtype, n_positions, n_active_tokens, batch_size):
        hidden_sizes = self.hidden_size, n_active_tokens, batch_size
        hidden = hidden_dtype[hidden_sizes].Parameter(parameter_number=0)
        position_ids = scribe.s32[n_active_tokens].Parameter(parameter_number=1)
        mask = hlo.decoder_attention_mask(position_ids, scribe.f32, n_positions)
        return (hidden, position_ids, mask), (1, 0)

    def layer(self, hidden, position_ids, mask, attn_k_cache, attn_v_cache,
              pre_attn_ln_weight, pre_attn_ln_bias, attn_q_weight, attn_q_bias,
              attn_k_weight, attn_k_bias, attn_v_weight, attn_v_bias,
              attn_out_weight, attn_out_bias, post_attn_ln_weight, post_attn_ln_bias,
              pre_mlp_ln_weight, pre_mlp_ln_bias, mlp_in_weight, mlp_in_bias,
              mlp_out_weight, mlp_out_bias, post_mlp_ln_weight, post_mlp_ln_bias):
        dtype = hidden.dtype
        ln_hidden = hlo.layer_norm(hidden, pre_attn_ln_weight, pre_attn_ln_bias)
        attn_output, out_attn_k_cache, out_attn_v_cache = self.attention(
            ln_hidden, position_ids, mask, attn_k_cache, attn_v_cache,
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
        ln_hidden = hlo.layer_norm(hidden, ln_f_weight, ln_f_bias)
        ln_hidden = dtype[hidden_size,n_active_tokens*batch_size].Reshape(ln_hidden)
        logits = hlo.dot00(lm_head_weight, ln_hidden)
        if lm_head_bias is not None:
            lm_head_bias = dtype[logits.sizes].Broadcast(lm_head_bias, dimensions=[0])
            logits = dtype[logits.sizes].Add(logits, lm_head_bias)
        vocab_size, _ = logits.sizes
        return dtype[vocab_size,n_active_tokens,batch_size].Reshape(logits)

    def attention(self, hidden, cache_offset, mask, cached_keys, cached_values,
                  q_weight, q_bias, k_weight, k_bias, v_weight, v_bias, out_weight, out_bias):
        dtype = hidden.dtype
        scribe = hidden.scribe
        f32 = scribe.f32
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

        scatter_dims = dict(update_window_dims=[1,2,3],
                            inserted_window_dims=[0],
                            scatter_dims_to_operand_dims=[0],
                            index_vector_dim=1)
        func = hlo.gen_assign_func(dtype)
        cached_keys = dtype[cached_keys.sizes].Scatter(
            cached_keys, cache_offset, active_k, scatter_dimension_numbers=scatter_dims, to_apply=func)
        cached_values = dtype[cached_values.sizes].Scatter(
            cached_values, cache_offset, active_v, scatter_dimension_numbers=scatter_dims, to_apply=func)

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
        score_sizes = n_seqs, n_heads_tp, n_active_tokens, max_ctx_plus_n_active_tokens
        score = dtype[score_sizes].Dot(active_q, cached_keys, dot_dimension_numbers=dot_dims)
        score = f32[score_sizes].Convert(score)

        mask_br = f32[score_sizes].Broadcast(mask, dimensions=[2, 3])
        score = f32[score_sizes].Multiply(score, mask_br)
        one = f32.Constant(constant_value=1.0)
        ones_br = f32[mask.sizes].Broadcast(one, dimensions=[])
        add_mask = f32[mask.sizes].Subtract(ones_br, mask)
        large_neg = f32.Constant(constant_value=-65536)
        large_neg_br = f32[add_mask.sizes].Broadcast(large_neg, dimensions=[])
        add_mask = f32[add_mask.sizes].Multiply(add_mask, large_neg_br)
        add_mask_br = f32[score_sizes].Broadcast(add_mask, dimensions=[2, 3])
        score = f32[score_sizes].Add(score, add_mask_br)

        probs = hlo.softmax(score)
        probs = dtype[score_sizes].Convert(probs)

        dot_dims = dict(lhs_contracting_dimensions=[3],
                        lhs_batch_dimensions=[0, 1],
                        rhs_contracting_dimensions=[0],
                        rhs_batch_dimensions=[1, 2])
        sizes = n_seqs, n_heads_tp, n_active_tokens, d_head
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
