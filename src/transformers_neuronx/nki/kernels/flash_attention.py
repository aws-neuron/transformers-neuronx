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

import numpy as np

from neuronxcc.nki.kernels import flash_fwd
from transformers_neuronx.nki.compile import nki_call

def flash_fwd_wrapper(q, k, v, o, lse):
  """
  Inputs:
    q: query tensor of shape (b, h, d, seqlen)
    k: key tensor of shape (b, h, d, seqlen)
    v: value tensor of shape (b, h, seqlen, d)
  Outputs:
    o: output buffer of shape (b, h, seqlen, d)
    lse: log-sum-exp for bwd pass stored in (b, h, nl.tile_size.pmax, seqlen // nl.tile_size.pmax) where nl.tile_size.pmax is 128
  """
  softmax_scale=None
  use_causal_mask=True
  mixed_precision=True  
  flash_fwd(q, k, v, o, lse, softmax_scale=softmax_scale, use_causal_mask=use_causal_mask, mixed_precision=mixed_precision)

if __name__ == "__main__":
    import torch
    from verification import build_run
    
    batch_size, nheads, d, seqlen = 2, 12, 128, 2048    
    dtype = torch.float32

    q = torch.zeros([batch_size, nheads, d, seqlen], dtype=dtype)
    k = torch.zeros([batch_size, nheads, d, seqlen], dtype=dtype)
    v = torch.zeros([batch_size, nheads, seqlen, d], dtype=dtype)
    out_shape = [batch_size, nheads, seqlen, d]
    out_cached_lse_shape = [batch_size, nheads, int(nl.tile_size.pmax), seqlen // nl.tile_size.pmax]
    out_shapes = (out_shape, out_cached_lse_shape)

    def mixed_pyhlo_nki(q, k, v):
        grid_x = q.sizes[0] 
        grid_y = q.sizes[1]
        # Another HLO op just for sake of example
        q = q.dtype[q.sizes].Multiply(q, q)
        o = nki_call(flash_fwd_wrapper, q, k, v, grid=(grid_x, grid_y), output_HloShapes=[q.dtype[sh] for sh in out_shapes])
        return o     
    
    result = build_run(mixed_pyhlo_nki, (q, k, v))
    print(result)
