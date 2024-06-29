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
from transformers_neuronx import hlo


def rotary_embedding(head_dim, cache_ids, base=10000, interpolation_factor=None):
    seq_len = cache_ids.shape[0]
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2) / head_dim))
    t = torch.arange(seq_len, dtype=inv_freq.dtype)
    if interpolation_factor:
        t /= interpolation_factor
    sinusoid_inp = torch.einsum("i , j -> i j", t, inv_freq).float()
    sin = torch.sin(sinusoid_inp)
    cos = torch.cos(sinusoid_inp)
    pos_embd = torch.cat((sin, cos), dim=-1)
    return pos_embd


def hlo_rotary_embedding(dtype, head_dim, cache_ids, base=10000, interpolation_factor=None):

    scribe = cache_ids.scribe
    # Using f16 during compute causes relatively high error
    mtype = scribe.f32

    cache_ids = hlo.cast(cache_ids, mtype)

    use_2d_cache_ids = len(cache_ids.sizes) > 1
    if use_2d_cache_ids:
        batch_size, n_active_tokens = cache_ids.sizes  # 2d cache_ids
        cache_ids = hlo.reshape(cache_ids, [batch_size, n_active_tokens, 1])
        dot_dims = dict(lhs_contracting_dimensions=[2], rhs_contracting_dimensions=[0])
    else:
        n_active_tokens, = cache_ids.sizes  # 1d cache_ids
        cache_ids = hlo.reshape(cache_ids, [n_active_tokens, 1])
        dot_dims = dict(lhs_contracting_dimensions=[1], rhs_contracting_dimensions=[0])
    size = head_dim // 2

    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2) / head_dim))
    inv_freq = hlo.literal(mtype, inv_freq)

    if interpolation_factor:
        cache_ids = hlo.divide(cache_ids, interpolation_factor)

    inv_freq = hlo.reshape(inv_freq, (1, size))
    sinusoid_inp = hlo.dot_general(cache_ids, inv_freq, dimension_numbers=dot_dims)

    sin = hlo.sin(sinusoid_inp)
    cos = hlo.cos(sinusoid_inp)
    sin = hlo.cast(sin, dtype)
    cos = hlo.cast(cos, dtype)
    return sin, cos


def get_up_down(q):
    """
    Given a tensor, returns its upper and lower halves (divided in the last dimension)
    """
    head_dim = q.sizes[-1]
    q_up = hlo.slice_along(q, -1, head_dim//2)
    q_down = hlo.slice_along(q, -1, head_dim, head_dim//2)
    return q_up, q_down

def get_up_down_with_percentage(q, percentage):
    """
    Given a tensor, returns its upper and lower halves with given percentage (divided in the last dimension)
    """
    head_dim = q.sizes[-1]
    q_up = hlo.slice_along(q, -1, int(head_dim * percentage))
    q_down = hlo.slice_along(q, -1, head_dim, int(head_dim * percentage))
    return q_up, q_down

def rotate_vec(q, sin_r, cos_r, rotary_percentage=1):
    """
    Given vectors q, sin, and cos tables, apply rotation to vectors
    """
    if rotary_percentage == 1:
        q_up, q_down = get_up_down(q)
        q_rot_up = hlo.ax_minus_by(cos_r, q_up, sin_r, q_down)
        q_rot_down = hlo.ax_plus_by(cos_r, q_down, sin_r, q_up)
        q_rot = hlo.concatenate([q_rot_up, q_rot_down], dimension=3)
        return q_rot
    else:
        q_rotary, q_pass = get_up_down_with_percentage(q, rotary_percentage)
        q_rotary_up, q_rotary_down = get_up_down(q_rotary)
        q_rotary_rot_up = hlo.ax_minus_by(cos_r, q_rotary_up, sin_r, q_rotary_down)
        q_rotary_rot_down = hlo.ax_plus_by(cos_r, q_rotary_down, sin_r, q_rotary_up)
        q_rotary_rot = hlo.concatenate([q_rotary_rot_up, q_rotary_rot_down], dimension=3)
        return hlo.concatenate([q_rotary_rot, q_pass], dimension=3)


def rotate_half(query, key, sin_cos, rotary_percentage=1, tp_degree=None, shard_over_batch=False):
    """
    A secondary projection to apply to input query/key projections (used in
    specific models: GPT-J/GPT-NeoX/Llama).

    """
    dtype = key.dtype
    if shard_over_batch:
        n_active_tokens, n_seqs_per_nc, n_kv_heads, d_head = active_sizes = key.sizes
        _, _, n_heads, _ = query.sizes
        broadcast_sizes = n_active_tokens, n_seqs_per_nc, n_heads, int((d_head // 2) * rotary_percentage)
        kv_broadcast_sizes = n_active_tokens, n_seqs_per_nc, n_kv_heads, int((d_head // 2) * rotary_percentage)
    else:
        n_active_tokens, n_seqs, n_kv_heads_tp, d_head = active_sizes = key.sizes
        _, _, n_heads_tp, _ = query.sizes

        """
            Vector approach:
            | q_up cos - q_down sin |
            | q_up sin + q_down cos |
        """
        # Rotate query and key
        broadcast_sizes = n_active_tokens, n_seqs, n_heads_tp, int((d_head // 2) * rotary_percentage)
        kv_broadcast_sizes = n_active_tokens, n_seqs, n_kv_heads_tp, int((d_head // 2) * rotary_percentage)

    def _broadcast_sin_cos(sin_cos, broadcast_sizes):
        sin, cos = sin_cos
        use_2d_cache_ids = len(sin.sizes) > 2
        if use_2d_cache_ids:
            # transpose from (n_seqs, n_active_tokens, d_head) to (n_active_tokens, n_seqs, d_head)
            sin_t = hlo.transpose(sin, 0, 1)
            cos_t = hlo.transpose(cos, 0, 1)
            # broadcast from (n_active_tokens, n_seqs, d_head) to (n_active_tokens, n_seqs, n_heads_tp, d_head)
            sin_r = hlo.broadcast(sin_t, broadcast_sizes, [0, 1, 3])
            cos_r = hlo.broadcast(cos_t, broadcast_sizes, [0, 1, 3])
        else:
            # 1D cache_ids
            sin_r = hlo.broadcast(sin, broadcast_sizes, [0, 3])
            cos_r = hlo.broadcast(cos, broadcast_sizes, [0, 3])
        return sin_r, cos_r

    # Get sin and cos as upper and lower half of input embedding
    sin_r, cos_r = _broadcast_sin_cos(sin_cos, broadcast_sizes)

    # Rotate query
    query = rotate_vec(query, sin_r, cos_r, rotary_percentage)

    # Get sin and cos as upper and lower half of input embedding
    kv_sin_r, kv_cos_r = _broadcast_sin_cos(sin_cos, kv_broadcast_sizes)

    # Rotate key
    key = rotate_vec(key, kv_sin_r, kv_cos_r, rotary_percentage)
    return query, key
