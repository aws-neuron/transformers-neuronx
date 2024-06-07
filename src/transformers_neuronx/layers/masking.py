from transformers_neuronx import hlo

def sliding_window_decoder_attention_mask_lhs_aligned(cache_ids, n_positions, window_size):
    """
    Create sliding window attention masks for LHS-aligned sequences. 
    Note that in sliding window each token attends to last window_size tokens including itself. 
    So during tokengen, we need to access at most window_size - 1 elements from the cache. 
    
    Ref:
    https://github.com/huggingface/transformers/blob/940fde8dafaecb8f17b588c5078291f1c1a420c8/src/transformers/models/mistral/modeling_mistral.py#L369-L373

    This function is focused on the case where n_positions > window_size where the KV cache is of size window_size. 
    In other cases we should simply use the normal masks.

    Args:
        cache_ids: The 2d positions to update in the cache (batch_size x active_mask) 
        n_positions: This is equal to the current bucket size. It determines the mask size when context encoding. 
            We assume n_positions > window_size, otherwise one should use normal masking functions.
        window_size: sliding window size. Each token attends to the last window_size tokens including itself.

    Returns:
        prior_mask: The attention mask to apply to the KV cache for tokengen, and the overall mask for context encoding.
        active_mask: The attention mask to apply to the active tokens (None for context encoding).
    """
    _batch_size, n_active_tokens = cache_ids.sizes
    assert n_positions > window_size, "Use decoder_attention_mask_lhs_aligned if n_positions <= window_size"
    if n_active_tokens == n_positions:
        # Context Encoding
        return sliding_window_decoder_attention_mask_lhs_aligned_context(cache_ids, n_positions, window_size)
    else:
        # Token generation (includes speculative decoding and windowed context encoding)
        return sliding_window_decoder_attention_mask_lhs_aligned_token(cache_ids, window_size)

def sliding_window_decoder_attention_mask_lhs_aligned_context(cache_ids, n_positions, window_size):
    """
    Creates a sliding window mask for LHS-aligned context encoding.

    This mask is static and does not depend on the inputs. During LHS-aligned
    context encoding, there is a guarantee that each token in a sequence must
    attend to all prior positions up to sliding window limit. This is unlike RHS-aligned 
    sequences where batch padding may require that an earlier position must not be attended to.

    Example:

        n_positions = 3
        window_size = 2
        prior_mask = [
            [1, 0, 0], # At position 0 attend to self only
            [1, 1, 0], # At position 1 attend to self and prior
            [0, 1, 1], # At position 2 attend to self and all prior (except first position since it's out of window)
        ]
        active_mask = None

    Args:
        cache_ids: The 2d positions to update in the cache.
        n_positions: Current bucket size.
        window_size: window size for sliding window.

    Returns:
        prior_mask: The attention mask to apply to scores.
        active_mask: None.
    """
    batch_size, _ = cache_ids.sizes
    dtype = cache_ids.scribe.pred
    sizes = (n_positions, n_positions)
    s32 = dtype.scribe.s32
    x = hlo.iota(s32, sizes, [0])
    y = hlo.iota(s32, sizes, [1])
    # we want (i) x >= y and (ii) x < y + window_size 
    condition_1 = hlo.greater_equal(x, y)
    window_size_br = hlo.full(window_size, s32, sizes)
    y_plus_window_size = hlo.add(y, window_size_br)
    condition_2 = hlo.less(x, y_plus_window_size)
    mask = hlo.logical_and(condition_1, condition_2)
    mask = hlo.cast(mask, dtype)

    # Final broadcast to add batch dim
    sizes_with_batch = batch_size, n_positions, n_positions
    mask = hlo.broadcast(mask, sizes_with_batch, [1, 2])

    active_mask = None
    return mask, active_mask

def sliding_window_decoder_attention_mask_lhs_aligned_token(cache_ids, window_size):
    """
    Creates decomposed prior/active masks for LHS-aligned token generation
    with sliding window. Here we assume KV cache of size window_size.

    Unlike LHS-aligned context encoding, this mask cannot be a completely
    static because each batch line may need to attend to a different number
    of prior tokens depending on the current token(s) being computed.

    This function assumes that `cache_ids` are linearly increasing per batch
    line when `n_active_tokens > 1` (speculative & windowed attention).

    Example: Single Token Generation

        window_size = 4
        cache_ids = [
            [2] # less than window size
        ]

        # Attend to all prior positions from the current token (2)
        prior_mask = [
            [1, 1, 0, 0]
        ]
        # Always attend to the current token
        active_mask = [
            [1]
        ]

        window_size = 4
        cache_ids = [
            [6] # more than window size
        ]

        # Attend to all prior positions except 6 % 4 = 2
        # Note we attend to window_size previous tokens including current
        prior_mask = [
            [1, 1, 0, 1]
        ]
        # Always attend to the current token
        active_mask = [
            [1]
        ]
        
    Example: Batched Execution

        window_size = 4
        cache_ids = [
            [5] # Batch 0
            [3] # Batch 1
        ]

        prior_mask = [
            [1, 0, 1, 1], # Batch 0 (skip 5 % 4 = 1)
            [1, 1, 1, 0] # Batch 1 (everything < 3)
        ]
        # Always attend to the current token on each batch line
        active_mask = [
            [1], # Batch 0
            [1], # Batch 1
        ]

    Example: Speculative Sampling/Windowed CE

        window_size = 4
        n_active_tokens = 3
        cache_ids = [
            [1, 2, 3], # Batch 0 
            [2, 3, 4], # Batch 1 (modulo 4 = [2, 3, 0])
            [5, 6, 7], # Batch 2 (modulo 4 = [1, 2, 3])
            [10, 11, 12], # Batch 3 (modulo 4 = [2, 3, 0])
        ]

        # prior mask is batch_size x n_active_tokens x window_size
        prior_mask = 
        [
            [
                [1, 0, 0, 0], # positions: 0, 1, 2, 3
                [1, 0, 0, 0],
                [1, 0, 0, 0],
            ], # batch 0 (simply determined by minimum cache_id 1)
            [
                [1, 1, 0, 0], # positions: 0, 1, 2, 3
                [1, 1, 0, 0],
                [0, 1, 0, 0]
            ], # batch 1 (cache_id 2 and 3 attend to all past tokens, 4 only attends to 1 past token)
            [
                [1, 0, 1, 1], # positions: 4, 1, 2, 3 (rolling KV cache)
                [1, 0, 0, 1],
                [1, 0, 0, 0],
            ], # batch 2 (cache_id 5 attends to all but index 1, and 6/7 attend to less due to sliding window)
            [
                [1, 1, 0, 1], # positions: 8, 9, 6, 7 (rolling KV cache)
                [1, 1, 0, 0],
                [0, 1, 0, 0],
            ], # batch 3 (cache_id 10 attends to all but 2 index, and 11/12 attend to less due to sliding window)
        ]

        # Use lower triangular mask for each active set of tokens
        active_mask = [
            [1, 0, 0],
            [1, 1, 0],
            [1, 1, 1],
        ]

    Args:
        cache_ids: The 2d positions to update in the cache.
        window_size: Window size for sliding window

    Returns:
        prior_mask: The attention mask to apply to the KV cache
        active_mask: The attention mask to apply to the active tokens.
    """
    batch_size, n_active_tokens = cache_ids.sizes
    assert n_active_tokens <= window_size, "Window size for windowed CE or Speculation length must be less than sliding window size"

    dtype = cache_ids.scribe.pred
    s32 = dtype.scribe.s32
    batch_size, n_active_tokens = cache_ids.sizes

    # Active mask (this is same as without sliding window case since n_active_tokens <= window_size)
    if n_active_tokens == 1:
        # Single token (Always pay attention to self)
        active_mask = hlo.full(1, dtype, (batch_size, n_active_tokens))
    else:
        # Multi-token speculative sampling & windowed attention
        causal_mask = hlo.tril_mask(dtype, (n_active_tokens, n_active_tokens))
        size = (batch_size, n_active_tokens, n_active_tokens)
        active_mask = hlo.broadcast(causal_mask, size, [1, 2])

    # Prior mask
    if n_active_tokens > 1:
        # Multi-token speculative sampling & windowed attention
        min_cache_ids = hlo.reduce_min(cache_ids, dim=1, keepdim=True) # find the min cache id for each batch line
    else:
        min_cache_ids = cache_ids

    sizes = (batch_size, n_active_tokens, window_size)
    y = hlo.iota(s32, sizes, [2]) # window_size dim

    # now we perform the operation y -> (min_cache_ids-window_size) + [(y-min_cache_ids) % window_size + window_size]%window_size
    # Second term ensures positive remainder.
    # This converts y to the actual position in the cache for the window, e.g., 
    # if window size = 4:
    # min_cache_ids = 2 -> y -> [0, 1, -2, -1] (denoting that last two values are garbage)
    # min_cache_ids = 5 -> y -> [4, 1, 2, 3] (rolling KV cache)
    # min_cache_ids = 10 -> y -> [8, 9, 6, 7] (rolling KV cache)

    window_size_br = hlo.full(window_size, s32, sizes)
    min_cache_ids_br = hlo.broadcast(min_cache_ids, sizes, [0, 1])
    modified_y_term1 = hlo.subtract(min_cache_ids_br, window_size_br)

    modified_y_term2 = hlo.subtract(y, min_cache_ids_br)
    modified_y_term2 = hlo.remainder(modified_y_term2, window_size_br)
    modified_y_term2 = hlo.add(modified_y_term2, window_size_br)
    modified_y_term2 = hlo.remainder(modified_y_term2, window_size_br)

    y = hlo.add(modified_y_term1, modified_y_term2)

    # now we mask according to following conditions:
    # (i) y >= 0 (note that we get negative values when KV cache is not filled)
    # (ii) cache_ids_br - y < window_size (so we don't access stuff not in sliding window)
    zeros_br = hlo.full(0, s32, sizes)
    condition_1 = hlo.greater_equal(y, zeros_br)
    cache_ids_br = hlo.broadcast(cache_ids, sizes, [0, 1])
    cache_ids_br_minus_y = hlo.subtract(cache_ids_br, y)
    condition_2 = hlo.less(cache_ids_br_minus_y, window_size_br)
    prior_mask = hlo.logical_and(condition_1, condition_2)
    prior_mask = hlo.cast(prior_mask, dtype)

    return prior_mask, active_mask