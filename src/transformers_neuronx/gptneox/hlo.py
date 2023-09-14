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
from transformers_neuronx.layers import attention, transformer

def build_gptneox_hlo_module(config, n_active_tokens, n_positions, debugger=None):
    gptneox = gen_scribable_gptneox(debugger, config, n_active_tokens, n_positions)
    hlo = compiler.compile_py_func(gptneox)
    return hlo


def gptneox_attention(debugger, hidden, q_weight, q_bias, k_weight, k_bias, v_weight, v_bias,
              out_weight, out_bias, pos_embd, cached_keys, cached_values,
              cache_ids, mask, active_mask, tp_degree, d_head):

    # Q = (hidden @ wQ) + bQ
    # K = (hidden @ wK) + bK
    # V = (hidden @ wV) + bV
    query, key, value = attention.query_key_value(
        hidden,
        q_weight, None, q_bias,
        k_weight, None, k_bias,
        v_weight, None, v_bias,
        d_head,
    )

    # Q = Q @ E
    # K = K @ E
    query, key = attention.query_key_projection(query, key, pos_embd)

    # Q = Q / sqrt(d_head)
    query = attention.scale(query, d_head)

    # Single Token Generation ("Prefetch"-style)
    if active_mask is not None:

        # Sp = Q @ Kp
        prior_scores = attention.score(query, cached_keys)
        prior_scores = attention.mask(prior_scores, mask)

        # Sa = Q @ Ka
        active_score = attention.score(query, key)
        active_score = attention.mask(active_score, active_mask)

        # C = softmax(Sa, Sp) @ (Va, Vp)
        context = attention.context(prior_scores, active_score, cached_values, value)

        # KCache[I] = K
        # VCache[I] = V
        updated_keys = attention.update_cache(cached_keys, cache_ids, key)
        updated_values = attention.update_cache(cached_values, cache_ids, value)

    # Multi-Token Context Encoding
    else:

        # S = Q @ K
        score = attention.score(query, key)
        score = attention.mask(score, mask)
        context = attention.context_combined(score, value)

        # KCache = K
        # VCache = V
        updated_keys = key
        updated_values = value

    # O = (C @ wO) + bO
    output = attention.output(context, out_weight, None, out_bias, tp_degree)
    return output, updated_keys, updated_values

def block(debugger, hidden, ln_1_weight, ln_1_bias,
          attn_q_weight, attn_q_bias, attn_k_weight, attn_k_bias,
          attn_v_weight, attn_v_bias, attn_out_weight, attn_out_bias,
          pos_embd, key_cache, value_cache, cache_offset, mask, active_mask, ln_2_weight, ln_2_bias,
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
    # Parallel Attention + FF Layers pseudocode:
    #   x = x + attn(ln1(x)) + mlp(ln2(x))
    hidden = hlo.transpose210(hidden)
    ln_hidden = hlo.layer_norm_bsh(hidden, ln_1_weight, ln_1_bias)  # input_layernorm
    attn_output, out_key_cache, out_value_cache = gptneox_attention(
        debugger,
        hidden=ln_hidden, q_weight=attn_q_weight, q_bias=attn_q_bias, k_weight=attn_k_weight, k_bias=attn_k_bias,
        v_weight=attn_v_weight, v_bias=attn_v_bias, out_weight=attn_out_weight, out_bias=attn_out_bias, pos_embd=pos_embd,
        cached_keys=in_key_cache, cached_values=in_value_cache, cache_ids=cache_offset, mask=mask, active_mask=active_mask,
        tp_degree=config.tp_degree, d_head=config.n_embd // config.n_head
    )
    out_hidden = dtype[hidden.sizes].Add(attn_output, hidden)
    ln_2_weight = hlo.transfer_with_static_ring(ln_2_weight)
    ln_2_bias = hlo.transfer_with_static_ring(ln_2_bias)
    mlp_in_weight = hlo.transfer_with_static_ring(mlp_in_weight)
    mlp_in_bias = hlo.transfer_with_static_ring(mlp_in_bias)
    mlp_out_weight = hlo.transfer_with_static_ring(mlp_out_weight)
    mlp_out_bias = hlo.transfer_with_static_ring(mlp_out_bias)
    out_ln_hidden = hlo.layer_norm_bsh(hidden, ln_2_weight, ln_2_bias) # post_attention_layernorm
    mlp_hidden = hlo.mlp_bsh(out_ln_hidden, mlp_in_weight, mlp_in_bias, mlp_out_weight, mlp_out_bias,
                             activation_function=config.activation_function, tp_degree=config.tp_degree)
    out_hidden = dtype[hidden.sizes].Add(mlp_hidden, out_hidden)
    out_hidden = hlo.transpose210(out_hidden)
    out_key_cache.set_alias_to(key_cache, must=True)
    out_value_cache.set_alias_to(value_cache, must=True)
    return out_hidden, out_key_cache, out_value_cache


def ln_lm_head(debugger, hidden, ln_f_weight, ln_f_bias, lm_head_weight):
    ln_f_weight = hlo.transfer_with_static_ring(ln_f_weight)
    ln_f_bias = hlo.transfer_with_static_ring(ln_f_bias)
    lm_head_weight = hlo.transfer_with_static_ring(lm_head_weight)
    return transformer.ln_lm_head(hidden, ln_f_weight, ln_f_bias, lm_head_weight, None)


def gptneox(debugger, hidden, pos_embd, cache_offset, mask, active_mask, blocks_caches, blocks_params, ln_lm_head_params, config):
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
        hidden, out_key_cache, out_value_cache = block(debugger, hidden, ln_1_weight,
          ln_1_bias, attn_q_weight, attn_q_bias, attn_k_weight, attn_k_bias,
          attn_v_weight, attn_v_bias, attn_out_weight, attn_out_bias,
          pos_embd, key_cache, value_cache, cache_offset, mask, active_mask, ln_2_weight, ln_2_bias,
          mlp_in_weight, mlp_in_bias, mlp_out_weight, mlp_out_bias,
          config)
        outputs.append(out_key_cache)
        outputs.append(out_value_cache)
    outputs.extend(debugger.get_tensors())

    ln_f_weight, ln_f_bias, lm_head_weight = ln_lm_head_params
    logits = ln_lm_head(debugger, hidden, ln_f_weight, ln_f_bias, lm_head_weight)
    outputs.insert(0, logits)
    root_shapes = [shape.dtype[shape.sizes] for shape in outputs]
    return scribe.tuple(*root_shapes).Tuple(*outputs)


def gen_scribable_gptneox(debugger, config, n_active_tokens, n_positions):
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
        start_ids = pbuilder([batch_size], dtype=scribe.s32)

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
            attn_v_weight = pbuilder([embed_dim, attn_dim_tp])
            attn_v_bias = pbuilder([attn_dim_tp])
            attn_out_weight = pbuilder([attn_dim_tp, embed_dim])
            attn_out_bias = pbuilder([embed_dim])
            ln_2_weight = pbuilder([embed_dim], dtype=scribe.f32)
            ln_2_bias = pbuilder([embed_dim], dtype=scribe.f32)
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

        token_generation = n_active_tokens == 1
        triu_comparison = 'LT' if token_generation else 'LE'
        mask, active_mask = hlo.decoder_attention_mask(
            start_ids,
            cache_offset,
            n_positions,
            triu_comparison=triu_comparison,
            allow_kv_dot_prefetch=token_generation,
            start_mask=True,
        )

        return gptneox(debugger, hidden, pos_embd, cache_offset, mask, active_mask, blocks_caches, blocks_params, ln_lm_head_params, config)

    return scribable