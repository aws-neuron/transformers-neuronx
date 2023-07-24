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
import math

import torch

from transformers_neuronx import hlo


def build_slopes(num_heads: int) -> torch.Tensor:
    """
    Builds a fixed slopes tensor to compute the ALiBI positional encoding.

    This tensor must be partitioned across the attention layers so that each
    attention head uses the correct ALiBI slope for the head allocated
    to that NeuronCore.

    Reference: https://github.com/huggingface/transformers/blob/v4.29.2/src/transformers/models/bloom/modeling_bloom.py#L86

    Arguments:
        num_heads: The number of attention heads for the model.

    Returns:
        slopes: The slope for each attention head. Shape: [num_heads, 1]
    """
    closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
    base = 2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3)))
    powers = range(1, 1 + closest_power_of_2)
    slopes = list(map(lambda x: math.pow(base, x), powers))

    if closest_power_of_2 != num_heads:
        extra_base = 2 ** (-(2 ** -(math.log2(2 * closest_power_of_2) - 3)))
        num_remaining_heads = min(closest_power_of_2, num_heads - closest_power_of_2)
        extra_powers = range(1, 1 + 2 * num_remaining_heads, 2)
        extra_slopes = list(map(lambda x: math.pow(extra_base, x), extra_powers))
        slopes.extend(extra_slopes)

    assert len(slopes) == num_heads
    return torch.tensor(slopes).view(num_heads, 1)


def alibi(slopes, attention_mask, active_mask=None):
    """
    Compute the ALiBI positional encoding from the attention mask and slopes.

    This must be used during both context encoding and token generation:
    - For prompt encoding only the `attention_mask` should be provided.
    - For token generation, the prefetch mechanism can be enabled by providing
      both an `attention_mask` for prior tokens and an `active_mask` for
      the newly generated tokens.

    See: alibi.build_slopes
         hlo.decoder_attention_mask

    Arguments:
        slopes: The ALiBI slopes for the current NeuronCore
        attention_mask: The mask to build the encoding for.
        active_mask: An optional mask only used during prefetch-attention.

    Returns:
        alibi: The ALiBI to apply for the given `attention_mask`
        active_alibi: The ALiBI to apply to the optional `active_mask`
    """

    num_heads_tp, *_ = slopes.sizes
    scribe = attention_mask.scribe
    dtype = scribe.f32

    def _alibi(summation, mask):

        size = mask.sizes
        batch_size, n_active_tokens, seq_length = mask.sizes

        one = dtype.Constant(constant_value=1)
        one_br = dtype[size].Broadcast(one, dimensions=[])
        summation_sub = dtype[size].Subtract(summation, one_br)
        sum_mul = dtype[size].Multiply(summation_sub, mask)

        slopes_sh = dtype[batch_size, n_active_tokens, num_heads_tp, 1].Broadcast(slopes, dimensions=[2, 3])
        sum_sh = dtype[batch_size, n_active_tokens, 1, seq_length].Reshape(sum_mul)
        dot_dims = dict(
            lhs_contracting_dimensions=[3],
            lhs_batch_dimensions=[0, 1],
            rhs_contracting_dimensions=[2],
            rhs_batch_dimensions=[0, 1]
        )
        product = dtype[batch_size, n_active_tokens, num_heads_tp, seq_length].Dot(slopes_sh, sum_sh, dot_dimension_numbers=dot_dims)
        result = dtype[batch_size, num_heads_tp, n_active_tokens, seq_length].Transpose(product, dimensions=[0, 2, 1, 3])
        return result

    scribe = attention_mask.scribe
    fp32 = scribe.f32

    # Create alibi for the `attention_mask` tokens
    mask_cast = hlo.cast(attention_mask, fp32)
    summation = hlo.cumsum(mask_cast, -1)
    alibi = _alibi(summation, mask_cast)

    # Create alibi for the `active_mask` tokens:
    #    Since the prior token mask is the `attention_mask` and the
    #    active token mask is the `active_mask`, we need to combine both masks to
    #    find the true cumulative sum.
    if active_mask is not None:
        total = hlo.reduce_sum(mask_cast, 2)
        active_cast = hlo.cast(active_mask, fp32)
        total = fp32[total.sizes].Add(total, active_cast)
        total_sh = hlo.unsqueeze(total, 1)
        active_cast_sh = hlo.unsqueeze(active_cast, 1)
        active_alibi = _alibi(total_sh, active_cast_sh)
        return alibi, active_alibi

    # When no active mask, we do not have an "active" alibi
    return alibi, None
