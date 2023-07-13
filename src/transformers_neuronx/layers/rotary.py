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


def rotary_embedding(head_dim, cache_ids, base=10000):
    seq_len = cache_ids.shape[0]
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2) / head_dim))
    sinusoid_inp = torch.einsum("i , j -> i j", torch.arange(seq_len), inv_freq).float()
    sin = torch.sin(sinusoid_inp)
    cos = torch.cos(sinusoid_inp)
    pos_embd = torch.cat((sin, cos), dim=-1)
    return pos_embd

def get_up_down(q):
    """
    Given a tensor, returns its upper and lower halves (divided in the last dimension)
    """
    head_dim = q.sizes[-1]
    q_up = hlo.slice_along(q, -1, head_dim//2)
    q_down = hlo.slice_along(q, -1, head_dim, head_dim//2)
    return q_up, q_down

def rotate_vec(q, sin_r, cos_r):
    """
    Given vectors q, sin, and cos tables, apply rotation to vectors
    """
    q_up, q_down = get_up_down(q)
    q_rot_up = hlo.ax_minus_by(cos_r, q_up, sin_r, q_down)
    q_rot_down = hlo.ax_plus_by(cos_r, q_down, sin_r, q_up)
    q_rot = q.dtype[q.sizes].Concatenate(q_rot_up, q_rot_down, dimensions=[3])
    return q_rot

def rotate_half(query, key, sin_cos):
    """
    A secondary projection to apply to input query/key projections (used in
    specific models: GPT-J/GPT-NeoX/Llama).

    """
    dtype = key.dtype
    n_active_tokens, n_seqs, n_heads_tp, d_head = active_sizes = key.sizes
    active_r_sizes = n_active_tokens, n_seqs * n_heads_tp, d_head

    """
        Vector approach:
        | q_up cos - q_down sin |
        | q_up sin + q_down cos |
    """
    # Rotate query and key
    n_active_tokens, head_dim = sin_cos.sizes
    sin_sizes = n_active_tokens, head_dim // 2        
    broadcast_sizes = n_active_tokens, n_seqs, n_heads_tp, d_head // 2
    
    # Get sin and cos as upper and lower half of input embedding
    sin, cos = get_up_down(sin_cos)        
    sin_r = dtype[broadcast_sizes].Broadcast(sin, dimensions=[0,3])
    cos_r = dtype[broadcast_sizes].Broadcast(cos, dimensions=[0,3])

    # Rotate query
    query = rotate_vec(query, sin_r, cos_r)        
    
    # Rotate key
    key = rotate_vec(key, sin_r, cos_r)
    return query, key