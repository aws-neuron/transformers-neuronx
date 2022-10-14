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


def build_gptj_block_kernel(config):
    block = gen_scribable_block(config)
    return compiler.build_kernel(block, config.tp_degree)


def build_lm_head_kernel(config):
    ln_lm_head = gen_scribable_ln_lm_head(config)
    return compiler.build_kernel(ln_lm_head, config.tp_degree)


def attention(hidden, q_weight, k_weight, v_weight, out_weight, pos_embd,
              cached_keys, cached_values, cache_offset, mask,
              n_heads, tp_degree):
    # hidden: [4096, a, b]
    # pos_embd: [a, 256, 256]
    # cached_keys: [2048, b, 16, 256]
    # cached_values: [2048, b, 16, 256]
    # mask: [a, 2048]
    dtype = hidden.dtype
    scribe = hidden.scribe
    f32 = scribe.f32
    s32 = scribe.s32
    pred = scribe.pred
    hidden_size, n_active_tokens, n_seqs = hidden_sizes = hidden.sizes
    max_ctx_plus_n_active_tokens, *_ = cached_keys.sizes
    attn_size = hidden_size // tp_degree
    hidden_r_sizes = hidden_size, n_active_tokens * n_seqs
    n_heads_tp = n_heads // tp_degree
    d_head = hidden_size // n_heads
    active_r_sizes = n_active_tokens, n_seqs * n_heads_tp, d_head
    active_sizes = n_active_tokens, n_seqs, n_heads_tp, d_head

    hidden_r = dtype[hidden_r_sizes].Reshape(hidden)    # [4096, a, b] -> [4096, a*b]
    active_q = hlo.dot00(hidden_r, q_weight)            # [4096, a*b] mm [4096, 4096] -> [a*b, 4096]
    active_q = dtype[active_r_sizes].Reshape(active_q)  # [a*b, 4096] -> [a, b*16, 256]
    dot_dims = dict(lhs_batch_dimensions=[0],
                    lhs_contracting_dimensions=[2],
                    rhs_batch_dimensions=[0],
                    rhs_contracting_dimensions=[1])
    # [a, b*16, 256] mm [a, 256, 256] -> [a, b*16, 256]
    active_q = dtype[active_r_sizes].Dot(active_q, pos_embd, dot_dimension_numbers=dot_dims)
    active_q = dtype[active_sizes].Reshape(active_q)

    active_k = hlo.dot00(hidden_r, k_weight)            # [4096, a*b] mm [4096, 4096] -> [a*b, 4096]
    active_k = dtype[active_r_sizes].Reshape(active_k)  # [a*b, 4096] -> [a, b*16, 256]
    dot_dims = dict(lhs_batch_dimensions=[0],
                    lhs_contracting_dimensions=[2],
                    rhs_batch_dimensions=[0],
                    rhs_contracting_dimensions=[1])
    # [a, b*16, 256] mm [a, 256, 256] -> [a, b*16, 256]
    active_k = dtype[active_r_sizes].Dot(active_k, pos_embd, dot_dimension_numbers=dot_dims)
    active_k = dtype[active_sizes].Reshape(active_k)

    active_v = hlo.dot00(hidden_r, v_weight)            # [4096, a*b] mm [4096, 4096] -> [a*b, 4096]
    active_v = dtype[active_sizes].Reshape(active_v)    # [a*b, 4096] -> [a, b, 16, 256]

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
    # [a, b, 16, 256] mm [2048, b, 16, 256] -> [b, 16, a, 2048]
    score = f32[score_sizes].Dot(active_q, cached_keys_f32, dot_dimension_numbers=dot_dims)

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
    # [b, 16, a, 2048] mm [2048, b, 16, 256] -> [b, 16, a, 256]
    output = dtype[sizes].Dot(probs, cached_values, dot_dimension_numbers=dot_dims)
    sizes = n_active_tokens, n_seqs, n_heads_tp, d_head
    # [b, 16, a, 256] -> [a, b, 16, 256]
    output = dtype[sizes].Transpose(output, dimensions=[2, 0, 1, 3])

    output_sizes_2d = n_active_tokens*n_seqs, attn_size
    output = dtype[output_sizes_2d].Reshape(output)     # [a, b, 16, 256] -> [a*b, 4096]
    dot_dims = dict(lhs_contracting_dimensions=[0], rhs_contracting_dimensions=[1])
    # [4096, 4096] mm [a*b, 4096] -> [4096, a*b]
    output = dtype[hidden_r_sizes].Dot(out_weight, output, dot_dimension_numbers=dot_dims)
    output = dtype[hidden_sizes].Reshape(output)
    replica_groups = [list(range(tp_degree))]
    add_func = hlo.gen_add_func(dtype)
    output = dtype[hidden_sizes].AllReduce(output, replica_groups=replica_groups, to_apply=add_func)

    return output, cached_keys, cached_values


def block(hidden, ln_1_weight, ln_1_bias,
          attn_q_weight, attn_k_weight, attn_v_weight, attn_out_weight,
          pos_embd, key_cache, value_cache, cache_offset, mask,
          mlp_in_weight, mlp_in_bias, mlp_out_weight, mlp_out_bias,
          config):
    dtype = hidden.dtype
    scribe = hidden.scribe
    ln_1_weight = hlo.transfer_with_static_ring(ln_1_weight)
    ln_1_bias = hlo.transfer_with_static_ring(ln_1_bias)
    attn_q_weight = hlo.transfer_with_static_ring(attn_q_weight)
    attn_k_weight = hlo.transfer_with_static_ring(attn_k_weight)
    attn_v_weight = hlo.transfer_with_static_ring(attn_v_weight)
    attn_out_weight = hlo.transfer_with_static_ring(attn_out_weight)
    in_key_cache = hlo.transfer_with_static_ring(key_cache)
    in_value_cache = hlo.transfer_with_static_ring(value_cache)
    ln_hidden = hlo.layer_norm(hidden, ln_1_weight, ln_1_bias)
    attn_output, out_key_cache, out_value_cache = attention(
        ln_hidden, attn_q_weight, attn_k_weight, attn_v_weight, attn_out_weight,
        pos_embd, in_key_cache, in_value_cache, cache_offset, mask,
        n_heads=config.n_head, tp_degree=config.tp_degree,
    )
    out_hidden = dtype[hidden.sizes].Add(attn_output, hidden)
    mlp_in_weight = hlo.transfer_with_static_ring(mlp_in_weight)
    mlp_in_bias = hlo.transfer_with_static_ring(mlp_in_bias)
    mlp_out_weight = hlo.transfer_with_static_ring(mlp_out_weight)
    mlp_out_bias = hlo.transfer_with_static_ring(mlp_out_bias)
    mlp_hidden = hlo.mlp(ln_hidden, mlp_in_weight, mlp_in_bias, mlp_out_weight, mlp_out_bias,
                         tp_degree=config.tp_degree)
    out_hidden = dtype[hidden.sizes].Add(mlp_hidden, out_hidden)
    out_hidden.set_alias_to(hidden)
    out_key_cache.set_alias_to(key_cache, must=True)
    out_value_cache.set_alias_to(value_cache, must=True)
    root_shape = scribe.tuple(dtype[hidden.sizes], dtype[key_cache.sizes], dtype[value_cache.sizes])
    return root_shape.Tuple(out_hidden, out_key_cache, out_value_cache)


def gen_scribable_block(config):
    embed_dim = config.n_embd
    n_heads = config.n_head
    n_positions = config.n_positions
    n_active_tokens = config.n_active_tokens
    batch_size = config.batch_size
    intermediate_dim = config.intermediate_dim
    amp = config.amp
    tp_degree = config.tp_degree
    head_dim = embed_dim // n_heads
    attn_dim_tp = embed_dim // tp_degree
    n_heads_tp = n_heads // tp_degree
    intermediate_dim_tp = intermediate_dim // tp_degree

    def scribable(scribe):
        pbuilder = hlo.ParameterBuilder(getattr(scribe, amp))
        hidden = pbuilder([embed_dim, n_active_tokens, batch_size])
        ln_1_weight = pbuilder([embed_dim], dtype=scribe.f32)
        ln_1_bias = pbuilder([embed_dim], dtype=scribe.f32)
        attn_q_weight = pbuilder([embed_dim, attn_dim_tp])
        attn_k_weight = pbuilder([embed_dim, attn_dim_tp])
        attn_v_weight = pbuilder([embed_dim, attn_dim_tp])
        attn_out_weight = pbuilder([attn_dim_tp, embed_dim])
        pos_embd = pbuilder([n_active_tokens, head_dim, head_dim])
        cache_shape = [n_positions, batch_size, n_heads_tp, head_dim]
        key_cache = pbuilder(cache_shape)
        value_cache = pbuilder(cache_shape)
        cache_offset = pbuilder([n_active_tokens], dtype=scribe.s32)
        mask = pbuilder([n_active_tokens, n_positions], dtype=scribe.f32)
        mlp_in_weight = pbuilder([embed_dim, intermediate_dim_tp])
        mlp_in_bias = pbuilder([intermediate_dim_tp])
        mlp_out_weight = pbuilder([intermediate_dim_tp, embed_dim])
        mlp_out_bias = pbuilder([embed_dim])
        return block(hidden, ln_1_weight, ln_1_bias, attn_q_weight, attn_k_weight, attn_v_weight,
                     attn_out_weight, pos_embd, key_cache, value_cache, cache_offset, mask,
                     mlp_in_weight, mlp_in_bias, mlp_out_weight, mlp_out_bias,
                     config)

    return scribable


def ln_lm_head(hidden, ln_f_weight, ln_f_bias, lm_head_weight, lm_head_bias):
    # single:
    #   hidden: [h, a, b]
    #   ln_f_weight: [h]
    #   ln_f_bias: [h]
    #   lm_head_weight: [h, v]
    #   lm_head_bias: [v]
    # t-way tp:
    #   hidden: [h, a, b]
    #   ln_f_weight: [h]
    #   ln_f_bias: [h]
    #   lm_head_weight: [h, v/t]
    #   lm_head_bias: [v/t]
    feature_size, n_active_tokens, batch_size = hidden.sizes
    dtype = hidden.dtype
    ln_f_weight = hlo.transfer_with_static_ring(ln_f_weight)
    ln_f_bias = hlo.transfer_with_static_ring(ln_f_bias)
    lm_head_weight = hlo.transfer_with_static_ring(lm_head_weight)
    lm_head_bias = hlo.transfer_with_static_ring(lm_head_bias)
    ln_hidden = hlo.layer_norm(hidden, ln_f_weight, ln_f_bias)
    ln_hidden = dtype[feature_size,n_active_tokens*batch_size].Reshape(ln_hidden)
    logits = hlo.dot00_add0(lm_head_weight, ln_hidden, lm_head_bias)
    vocab_size, _ = logits.sizes
    return dtype[vocab_size,n_active_tokens,batch_size].Reshape(logits)


def gen_scribable_ln_lm_head(config):
    embed_dim = config.n_embd
    vocab_size = config.vocab_size
    n_active_tokens = config.n_active_tokens
    batch_size = config.batch_size
    amp = config.amp
    tp_degree = config.tp_degree
    if vocab_size % tp_degree:
        vocab_size = (vocab_size // tp_degree + 1) * tp_degree
    vocab_size_tp = vocab_size // tp_degree

    def scribable(scribe):
        pbuilder = hlo.ParameterBuilder(getattr(scribe, amp))
        hidden = pbuilder([embed_dim, n_active_tokens, batch_size])
        ln_f_weight = pbuilder([embed_dim], dtype=scribe.f32)
        ln_f_bias = pbuilder([embed_dim], dtype=scribe.f32)
        lm_head_weight = pbuilder([embed_dim, vocab_size_tp])
        lm_head_bias = pbuilder([vocab_size_tp])
        return ln_lm_head(hidden, ln_f_weight, ln_f_bias, lm_head_weight, lm_head_bias)

    return scribable
