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
from transformers_neuronx import compiler
from transformers_neuronx import hlo

def build_gptneox_hlo_module(config, n_active_tokens, n_positions, debugger=None):
    builder = Builder(debugger=debugger)
    gptneox = builder.gen_scribable_gptneox(config, n_active_tokens, n_positions)
    hlo = compiler.compile_py_func(gptneox)
    return hlo


def attention(self, hidden, q_weight, q_bias, k_weight, k_bias, v_weight, v_bias,
              out_weight, out_bias, pos_embd, cached_keys, cached_values,
              cache_offset, mask, tp_degree): 
    # TODO ADD SHAPE ANNOTATIONS
    dtype = hidden.dtype
    scribe = hidden.scribe
    f32 = scribe.f32
    hidden_size, n_active_tokens, n_seqs = hidden_sizes = hidden.sizes
    max_ctx_plus_n_active_tokens, _, n_heads_tp, d_head = cached_keys.sizes
    attn_size = hidden_size // tp_degree
    hidden_r_sizes = hidden_size, n_active_tokens * n_seqs
    active_r_sizes = n_active_tokens, n_seqs * n_heads_tp, d_head
    active_sizes = n_active_tokens, n_seqs, n_heads_tp, d_head

    hidden_r = dtype[hidden_r_sizes].Reshape(hidden)    

    active_q = hlo.dot00_add1(hidden_r, q_weight, q_bias)
<<<<<<< HEAD

    # Debug feature: 
    #       1) If debugger is instatiated by debugger = Debugger(False), debugger.add_var does nothing 
    #       2) If debugger is instatiated by debugger = Debugger(False), this adds the variable to output vars
    #          and the tensor can be obtained in cpu side by compiler.get_debug_outputs(program)
    self.debugger.add_var(active_q, "active_q")
=======
    self.debugger.add_var(active_q, "active_q_before")
>>>>>>> e8bc81d (Adding debugging feature to transformers-neuronx. Added a debugger class)
    active_q = dtype[active_r_sizes].Reshape(active_q)  
    self.debugger.add_var(active_q, "active_q_after_reshape")

    dot_dims = dict(lhs_batch_dimensions=[0],
                    lhs_contracting_dimensions=[2],
                    rhs_batch_dimensions=[0],
                    rhs_contracting_dimensions=[1])
<<<<<<< HEAD
=======
    self.debugger.add_var(active_q, "active_q_after")
>>>>>>> e8bc81d (Adding debugging feature to transformers-neuronx. Added a debugger class)

    active_q = dtype[active_r_sizes].Dot(active_q, pos_embd, dot_dimension_numbers=dot_dims)
    active_q = dtype[active_sizes].Reshape(active_q)

    active_k = hlo.dot00_add1(hidden_r, k_weight, k_bias)
    active_k = dtype[active_r_sizes].Reshape(active_k)
    dot_dims = dict(lhs_batch_dimensions=[0],
                    lhs_contracting_dimensions=[2],
                    rhs_batch_dimensions=[0],
                    rhs_contracting_dimensions=[1])


    active_k = dtype[active_r_sizes].Dot(active_k, pos_embd, dot_dimension_numbers=dot_dims)
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
    active_q = f32[active_sizes].Convert(active_q)
    scale_attn = d_head ** 0.5
    scale = f32.Constant(constant_value=scale_attn)
    scale_attn_br = f32[active_sizes].Broadcast(scale, dimensions=[])
    active_q = f32[active_sizes].Divide(active_q, scale_attn_br)
    cached_keys_f32 = f32[cached_keys.sizes].Convert(cached_keys)
    dot_dims = dict(lhs_contracting_dimensions=[3],
                    lhs_batch_dimensions=[1, 2],
                    rhs_contracting_dimensions=[3],
                    rhs_batch_dimensions=[1, 2])
    score_sizes = n_seqs, n_heads_tp, n_active_tokens, max_ctx_plus_n_active_tokens

    score = f32[score_sizes].Dot(active_q, cached_keys_f32, dot_dimension_numbers=dot_dims)
    self.debugger.add_var(score, "score")

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
    replica_groups = [list(range(tp_degree))]
    add_func = hlo.gen_add_func(dtype)
    output = dtype[hidden_sizes].AllReduce(output, replica_groups=replica_groups, to_apply=add_func)

    return output, cached_keys, cached_values


def block(self, hidden, ln_1_weight, ln_1_bias,
          attn_q_weight, attn_q_bias, attn_k_weight, attn_k_bias,
          attn_v_weight, attn_v_bias, attn_out_weight, attn_out_bias,
          pos_embd, key_cache, value_cache, cache_offset, mask, ln_2_weight, ln_2_bias,
          mlp_in_weight, mlp_in_bias, mlp_out_weight, mlp_out_bias,
          config):
    dtype = hidden.dtype
    ln_1_weight = hlo.transfer_with_static_ring(ln_1_weight)
    ln_1_bias = hlo.transfer_with_static_ring(ln_1_bias)
    attn_q_weight = hlo.transfer_with_static_ring(attn_q_weight)
    attn_q_bias = hlo.transfer_with_static_ring(attn_q_bias)
    attn_k_weight = hlo.transfer_with_static_ring(attn_k_weight)
    attn_k_bias = hlo.transfer_with_static_ring(attn_k_bias)
    attn_v_weight = hlo.transfer_with_static_ring(attn_v_weight)
    attn_v_bias = hlo.transfer_with_static_ring(attn_v_bias)
    attn_out_weight = hlo.transfer_with_static_ring(attn_out_weight)
    attn_out_bias = hlo.transfer_with_static_ring(attn_out_bias)
    in_key_cache = hlo.transfer_with_static_ring(key_cache)
    in_value_cache = hlo.transfer_with_static_ring(value_cache)
    ln_hidden = hlo.layer_norm(hidden, ln_1_weight, ln_1_bias)
    attn_output, out_key_cache, out_value_cache = attention(
        self,
        hidden=ln_hidden, q_weight=attn_q_weight, q_bias=attn_q_bias, k_weight=attn_k_weight, k_bias=attn_k_bias,
        v_weight=attn_v_weight, v_bias=attn_v_bias, out_weight=attn_out_weight, out_bias=attn_out_bias, pos_embd=pos_embd,
        cached_keys=in_key_cache, cached_values=in_value_cache, cache_offset=cache_offset, mask=mask,
        tp_degree=config.tp_degree,
    )
    out_hidden = dtype[hidden.sizes].Add(attn_output, hidden)
    ln_2_weight = hlo.transfer_with_static_ring(ln_2_weight)
    ln_2_bias = hlo.transfer_with_static_ring(ln_2_bias)
    mlp_in_weight = hlo.transfer_with_static_ring(mlp_in_weight)
    mlp_in_bias = hlo.transfer_with_static_ring(mlp_in_bias)
    mlp_out_weight = hlo.transfer_with_static_ring(mlp_out_weight)
    mlp_out_bias = hlo.transfer_with_static_ring(mlp_out_bias)
    # Parallel Attention + FF Layers pseudocode:
    #   x = x + attn(ln1(x)) + mlp(ln2(x))
    out_ln_hidden = hlo.layer_norm(hidden, ln_2_weight, ln_2_bias)
    mlp_hidden = hlo.mlp(out_ln_hidden, mlp_in_weight, mlp_in_bias, mlp_out_weight, mlp_out_bias,
                         activation_function=config.activation_function, tp_degree=config.tp_degree)
    out_hidden = dtype[hidden.sizes].Add(mlp_hidden, out_hidden)
    out_key_cache.set_alias_to(key_cache, must=True)
    out_value_cache.set_alias_to(value_cache, must=True)
    return out_hidden, out_key_cache, out_value_cache


def ln_lm_head(self, hidden, ln_f_weight, ln_f_bias, lm_head_weight):
    feature_size, n_active_tokens, batch_size = hidden.sizes
    dtype = hidden.dtype
    ln_f_weight = hlo.transfer_with_static_ring(ln_f_weight)
    ln_f_bias = hlo.transfer_with_static_ring(ln_f_bias)
    lm_head_weight = hlo.transfer_with_static_ring(lm_head_weight)
    ln_hidden = hlo.layer_norm(hidden, ln_f_weight, ln_f_bias)
    ln_hidden = dtype[feature_size,n_active_tokens*batch_size].Reshape(ln_hidden)
    logits = hlo.dot00(lm_head_weight, ln_hidden) # GPT-NeoX embed_out Linear doesn't use bias
    vocab_size, _ = logits.sizes
    return dtype[vocab_size,n_active_tokens,batch_size].Reshape(logits)


def gptneox(self, hidden, pos_embd, cache_offset, mask, blocks_caches, blocks_params, ln_lm_head_params, config):
    scribe = hidden.scribe
    dtype = hidden.dtype
    outputs = []
    for (key_cache, value_cache), block_params in zip(blocks_caches, blocks_params):
        (
            ln_1_weight, ln_1_bias, attn_q_weight, attn_q_bias,
            attn_k_weight, attn_k_bias, attn_v_weight, attn_v_bias,
            attn_out_weight, attn_out_bias, ln_2_weight, ln_2_bias,
            mlp_in_weight, mlp_in_bias, mlp_out_weight, mlp_out_bias,
        ) = block_params
        hidden, out_key_cache, out_value_cache = block(self, hidden, ln_1_weight,
          ln_1_bias, attn_q_weight, attn_q_bias, attn_k_weight, attn_k_bias,
          attn_v_weight, attn_v_bias, attn_out_weight, attn_out_bias,
          pos_embd, key_cache, value_cache, cache_offset, mask, ln_2_weight, ln_2_bias,
          mlp_in_weight, mlp_in_bias, mlp_out_weight, mlp_out_bias,
          config)
        outputs.append(out_key_cache)
        outputs.append(out_value_cache)
    outputs.extend(self.debugger.get_tensors())

    ln_f_weight, ln_f_bias, lm_head_weight = ln_lm_head_params
    logits = ln_lm_head(self, hidden, ln_f_weight, ln_f_bias, lm_head_weight)
    outputs.insert(0, logits)
    root_shapes = [shape.dtype[shape.sizes] for shape in outputs]
    return scribe.tuple(*root_shapes).Tuple(*outputs)

class Builder:
    def __init__(self, debugger=None):
        self.debugger = debugger

    def gen_scribable_gptneox(self, config, n_active_tokens, n_positions):
        embed_dim = config.n_embd
        n_heads = config.n_head
        batch_size = config.batch_size
        intermediate_dim = config.intermediate_dim
        amp = config.amp
        tp_degree = config.tp_degree
        n_layer = config.n_layer
        vocab_size = config.vocab_size
        head_dim = embed_dim // n_heads
        attn_dim_tp = embed_dim // tp_degree
        n_heads_tp = n_heads // tp_degree
        intermediate_dim_tp = intermediate_dim // tp_degree
        if vocab_size % tp_degree:
            vocab_size = (vocab_size // tp_degree + 1) * tp_degree
        vocab_size_tp = vocab_size // tp_degree

        def scribable(scribe):
            pbuilder = hlo.ParameterBuilder(getattr(scribe, amp))
            hidden = pbuilder([embed_dim, n_active_tokens, batch_size])
            pos_embd = pbuilder([n_active_tokens, head_dim, head_dim])
            cache_offset = pbuilder([n_active_tokens], dtype=scribe.s32)

            def gen_block_caches():
                cache_shape = [n_positions, batch_size, n_heads_tp, head_dim]
                key_cache = pbuilder(cache_shape)
                value_cache = pbuilder(cache_shape)
                return key_cache, value_cache

            def gen_block_params():
                ln_1_weight = pbuilder([embed_dim], dtype=scribe.f32)
                ln_1_bias = pbuilder([embed_dim], dtype=scribe.f32)
                attn_q_weight = pbuilder([embed_dim, attn_dim_tp])
                attn_q_bias = pbuilder([attn_dim_tp])
                attn_k_weight = pbuilder([embed_dim, attn_dim_tp])
                attn_k_bias = pbuilder([attn_dim_tp])
                # embed_dim = 6144, attn_dim_tp = 192
                attn_v_weight = pbuilder([embed_dim, attn_dim_tp])
                attn_v_bias = pbuilder([attn_dim_tp])
                attn_out_weight = pbuilder([attn_dim_tp, embed_dim])
                attn_out_bias = pbuilder([embed_dim])
                ln_2_weight = pbuilder([embed_dim], dtype=scribe.f32)
                ln_2_bias = pbuilder([embed_dim], dtype=scribe.f32)
                # 6144, intermediate_dim_tp = 768
                mlp_in_weight = pbuilder([embed_dim, intermediate_dim_tp])
                mlp_in_bias = pbuilder([intermediate_dim_tp])
                mlp_out_weight = pbuilder([intermediate_dim_tp, embed_dim])
                mlp_out_bias = pbuilder([embed_dim])
                return (
                    ln_1_weight, ln_1_bias, attn_q_weight, attn_q_bias,
                    attn_k_weight, attn_k_bias, attn_v_weight, attn_v_bias,
                    attn_out_weight, attn_out_bias, ln_2_weight, ln_2_bias,
                    mlp_in_weight, mlp_in_bias, mlp_out_weight, mlp_out_bias,
                )

            def gen_ln_lm_head_params():
                ln_f_weight = pbuilder([embed_dim], dtype=scribe.f32)
                ln_f_bias = pbuilder([embed_dim], dtype=scribe.f32)
                lm_head_weight = pbuilder([embed_dim, vocab_size_tp])
                return ln_f_weight, ln_f_bias, lm_head_weight

            blocks_caches = [gen_block_caches() for _ in range(n_layer)]
            blocks_params = [gen_block_params() for _ in range(n_layer)]
            ln_lm_head_params = gen_ln_lm_head_params()
            mask = hlo.decoder_attention_mask(cache_offset, scribe.f32, n_positions)
            return gptneox(self, hidden, pos_embd, cache_offset, mask, blocks_caches, blocks_params, ln_lm_head_params, config)

        return scribable