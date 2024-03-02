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

import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
from neuronxcc.nki.language import par_dim
from transformers_neuronx.nki.compile import nki_call

def flash_fwd(q, k, v, o, lse):
  """
  Inputs:
    q: query tensor of shape (b, h, d, seqlen)
    k: key tensor of shape (b, h, d, seqlen)
    v: value tensor of shape (b, h, seqlen, d)
  Outputs:
    o: output buffer of shape (b, h, seqlen, d)
    lse: log-sum-exp for bwd pass stored in (b, h, nl.tile_size.pmax, seqlen // nl.tile_size.pmax) where nl.tile_size.pmax is 128
  Compile-time Constants:
    softmax_scale: scaling for softmax, is None, default is `1.0/(d**0.5)`
    mixed_precision: flag to set non-matmul ops in fp32 precision, defualt is set to `true`, if false, we use same precision as input types
    causal_mask: flag to set causal masking
  Performance Notes:
    For better performance, the kernel is tiled to be of size `LARGE_TILE_SZ`, and Flash attention math techniques are applied in unit
    of `LARGE_TILE_SZ`. Seqlen that is not divisible by `LARGE_TILE_SZ` is not supported at the moment.

  """
  softmax_scale=None
  use_causal_mask=True
  mixed_precision=True  
  assert k.shape == q.shape, f"Q, K shape have to match, got q: {q.shape}, k: {k.shape}"
  B_F_SIZE=512
  B_P_SIZE=128
  b , h, d, n  = q.shape
  B_D_SIZE = d
  assert tuple(v.shape) == (b, h, n, d), f"V shape does not match layout requirements, expect: {(b, h, n, d)} but got {v.shape}"
  assert d <= 128, f" we do not support head_dim > 128, got head dim {d}"
  kernel_dtype = nl.bfloat16 if mixed_precision else q.dtype
  acc_type =  np.dtype(np.float32) if mixed_precision else kernel_dtype

  i_q_p = nl.arange(B_P_SIZE)[:,None]
  i_q_f = nl.arange(B_F_SIZE)[None, :]
  i_0_f = nl.arange(1)[None, :]
  n_tile_q = n//B_P_SIZE # since q will be loaded on PE

  batch_id = nl.program_id(axis=0)
  head_id = nl.program_id(axis=1)
  softmax_scale = softmax_scale or (1.0 / (d ** 0.5))

  LARGE_TILE_SZ = 2 * 1024
  assert n % LARGE_TILE_SZ == 0, f"seqlen is not divisible by {LARGE_TILE_SZ}"
  num_large_k_tile = n // LARGE_TILE_SZ
  num_k_tile_per_large_tile = LARGE_TILE_SZ // B_F_SIZE
  i_f_k_tiles = nl.arange(num_k_tile_per_large_tile)[None, :]

  REDUCTION_TILE = 1024

  for i in nl.affine_range(n_tile_q):
    i_f_128 = nl.arange(B_P_SIZE)[None, :]
    i_f_d = nl.arange(B_D_SIZE)[None, :]
    i_p_d = nl.arange(B_D_SIZE)[:,None]
    q_tile = nl.ndarray((B_D_SIZE, B_P_SIZE),dtype=kernel_dtype)
    q_tile[i_p_d, i_f_128] = nl.load(q[batch_id, head_id, i_p_d, i*B_P_SIZE + i_f_128], dtype=kernel_dtype)*softmax_scale # load (d, 128) tile in SBUF
    # =============== Global Flash Attention accumulators ====================== #
    o_buffer = nl.full((num_large_k_tile, par_dim(B_P_SIZE), d), 0.0, dtype=acc_type, buffer=nl.sbuf) # zeros does not work
    l_buffer = nl.full((num_large_k_tile, par_dim(B_P_SIZE), 1), 0.0, dtype=acc_type, buffer=nl.sbuf)
    m_buffer = nl.full((num_large_k_tile, par_dim(B_P_SIZE), 1), 0.0, dtype=acc_type) # workaround for nl.full issue
    # =============== Global Flash Attention accumulators END ================== #
    qk_res_buf = nl.ndarray((par_dim(B_P_SIZE), LARGE_TILE_SZ), buffer=nl.sbuf, dtype=kernel_dtype)

    # Perform first large tile of qk, the behaviour writing to global flash attention accumulators in the first iteration
    # is different from rest of the iteration, therefore this loop need to be standalone.
    max_local = nl.ndarray((par_dim(B_P_SIZE), num_k_tile_per_large_tile), dtype=acc_type)
    for k_i in nl.affine_range(num_k_tile_per_large_tile):
      k_tile = nl.ndarray((B_D_SIZE, B_F_SIZE), dtype=kernel_dtype)
      k_tile[i_p_d, i_q_f] = nl.load(k[batch_id, head_id, i_p_d, k_i * B_F_SIZE + i_q_f], dtype=kernel_dtype) # (d, 512)
      qk_psum = nl.zeros((par_dim(B_P_SIZE), B_F_SIZE),
                        dtype=np.float32, buffer=nl.psum)  # (128, 512)
      qk_psum[i_q_p,i_q_f] += nl.matmul(q_tile, k_tile, transpose_x=True) # (p(128), 512)

      if use_causal_mask:
        # Magic number -9984.0 to replace -inf similar to what Tensorizer uses
        qk_res_buf[i_q_p, k_i * B_F_SIZE + i_q_f] = nisa.affine_select(
          pred=(i * B_P_SIZE + i_q_p >= k_i * B_F_SIZE + i_q_f),
          on_true_tile=qk_psum[i_q_p, i_q_f], on_false_value=-9984.0, dtype=kernel_dtype
          )
      else:
        # Simply send psum result back to sbuf
        qk_res_buf[i_q_p, k_i * B_F_SIZE + i_q_f] = \
          nl.copy(qk_psum[i_q_p, i_q_f], dtype=kernel_dtype)

      # Calculate max of the current tile
      max_local[i_q_p, k_i] = nisa.reduce(np.max,qk_res_buf[i_q_p, k_i * B_F_SIZE + i_q_f],axis=(1,),dtype=acc_type, negate=False)

    max_ = nisa.reduce(np.min, max_local[i_q_p, i_f_k_tiles], axis=(1, ), dtype=acc_type, negate=False)
    p = nl.ndarray((par_dim(B_P_SIZE), LARGE_TILE_SZ), dtype=acc_type)


    i_r_f = nl.arange(REDUCTION_TILE)[None,: ]
    p_partial_sum = nl.ndarray((par_dim(B_P_SIZE), LARGE_TILE_SZ//REDUCTION_TILE),dtype=acc_type)
    for k_r_i in nl.affine_range(LARGE_TILE_SZ//REDUCTION_TILE):
      # compute exp(qk-max)
      p[i_q_p, k_r_i*REDUCTION_TILE+i_r_f] = nisa.activation(np.exp, qk_res_buf[i_q_p, k_r_i*REDUCTION_TILE+i_r_f],bias=-1*max_, scale=1.0, dtype=kernel_dtype)
      # Compute partial row-tile sum after exp(qk-max)
      p_partial_sum[i_q_p, k_r_i] = nl.sum(p[i_q_p, k_r_i*REDUCTION_TILE+i_r_f], axis=1, dtype=acc_type)

    ps = nl.sum(p_partial_sum, axis=1,dtype=acc_type)

    v_local = nl.ndarray((LARGE_TILE_SZ//B_P_SIZE, par_dim(B_P_SIZE), B_D_SIZE), dtype=kernel_dtype)
    pv_psum = nl.zeros((par_dim(B_P_SIZE), B_D_SIZE), dtype=np.float32, buffer = nl.psum) # ideally we want to re-use qk_psum buffer here
    for k_i in nl.affine_range(LARGE_TILE_SZ//B_P_SIZE):
        v_local[k_i, i_q_p, i_f_d] = nl.load(v[batch_id, head_id, k_i*B_P_SIZE + i_q_p, i_f_d], dtype=kernel_dtype)
        pv_psum[i_q_p,i_f_d] += nl.matmul(p[i_q_p, k_i*B_P_SIZE + i_f_128],v_local[k_i, i_q_p, i_f_d], transpose_x=False) # (128, 128) (p(Br), d)

    ### ------- udpate flash attention accumulators ----------------------- #
    o_buffer[0, i_q_p, i_f_d] = nl.copy(pv_psum[i_q_p,i_f_d])
    l_buffer[0, i_q_p, i_0_f] = nl.add(nl.log(ps),max_)
    m_buffer[0, i_q_p, i_0_f] = nl.copy(max_)
    # ------------------------Initialization done  --------------------------- #

    for j in nl.sequential_range(num_large_k_tile-1): # range done not work for affine_range(1,n_tile_k)
      j = j + 1

      # -------- calculate q * k ----------------
      # note q is already loaded in the outer loop
      # laod a tile of k [128, 512]

      # mask are used to only apply computation to the lower half of the matrix,
      # which reduce the arthimetic intensity by half
      forward_mask = i * B_P_SIZE >= j * LARGE_TILE_SZ if use_causal_mask else None
      # Negation mask is the negation of `forward_mask`, which is used for the
      # instructions executed on the blocks in the upper triangular section
      # of the matrix.
      # These instructions should not be executed when causual mask is disabled.
      #
      # For example, the o_buffer still needs to be propagated from o[j-1] to o[j] in
      # the upper triangular of the matrix.
      negation_mask = i * B_P_SIZE < j * LARGE_TILE_SZ if use_causal_mask else None


      qk_res_buf = nl.ndarray((par_dim(B_P_SIZE), LARGE_TILE_SZ), buffer=nl.sbuf, dtype=acc_type)
      max_local = nl.ndarray((par_dim(B_P_SIZE), num_k_tile_per_large_tile), dtype=acc_type)
      for k_i in nl.affine_range(num_k_tile_per_large_tile):
        k_tile = nl.ndarray((B_D_SIZE, B_F_SIZE), dtype=kernel_dtype)
        k_tile[i_p_d, i_q_f] = nl.load(k[batch_id, head_id, i_p_d, j*LARGE_TILE_SZ + k_i * B_F_SIZE + i_q_f], dtype=kernel_dtype, mask=forward_mask) # (d, 512)
        qk_psum = nl.zeros((par_dim(B_P_SIZE), B_F_SIZE),
                          dtype=np.float32, buffer=nl.psum)  # (128, 512)
        # multiplication is not required passing the diagonal, masking them out
        multiplication_required_selection = j * LARGE_TILE_SZ + k_i * B_F_SIZE <= i * B_P_SIZE if use_causal_mask else None
        qk_psum[i_q_p,i_q_f] += nl.matmul(q_tile, k_tile, transpose_x=True, mask=multiplication_required_selection) # (p(128), 512)

        if use_causal_mask:
            left_diagonal_selection = i * B_P_SIZE >= j * LARGE_TILE_SZ + (k_i + 1) * B_F_SIZE
            diagonal_and_right_selection = (i * B_P_SIZE < j * LARGE_TILE_SZ + (k_i + 1) * B_F_SIZE) & forward_mask

            # For tiles on and on the right of the diagonal, need to do affine_select.
            # Magic number -9984.0 to replace -inf similar to what Tensorizer uses
            qk_res_buf[i_q_p, k_i * B_F_SIZE + i_q_f] = nisa.affine_select(
              pred=(i * B_P_SIZE + i_q_p >= j * LARGE_TILE_SZ + k_i * B_F_SIZE + i_q_f),
              on_true_tile=qk_psum[i_q_p, i_q_f], on_false_value=-9984.0, dtype=kernel_dtype,
              mask=diagonal_and_right_selection
              )

            # For tiles on the left of the diagonal, direct copy, no select required.
            qk_res_buf[i_q_p, k_i * B_F_SIZE + i_q_f] = \
              nl.copy(qk_psum[i_q_p, i_q_f], dtype=kernel_dtype, mask=left_diagonal_selection)
        else:
            # Simply send psum result back to sbuf
            qk_res_buf[i_q_p, k_i * B_F_SIZE + i_q_f] = \
              nl.copy(qk_psum[i_q_p, i_q_f], dtype=kernel_dtype)

        max_local[i_q_p, k_i] = nisa.reduce(np.max,qk_res_buf[i_q_p, k_i * B_F_SIZE + i_q_f],axis=(1,),dtype=acc_type, negate=False, mask=forward_mask)

      # -------- calculate local max and update with max (j-1) --------- #

      m_previous = nl.copy(m_buffer[j-1,i_q_p, i_0_f])
      m_max_ = nisa.reduce(np.min, max_local[i_q_p, i_f_k_tiles], axis=(1, ), dtype=acc_type, negate=False, mask=forward_mask)
      m_buffer[j, i_q_p, i_0_f] = nl.maximum(m_previous, m_max_, mask=forward_mask)# (128,1)
      if use_causal_mask:
        # FIXME: GIGANTIC HACK! Copy m_buffer[j-1, :, :] to m_buffer[j, :, :] triggers error in the compiler,
        # use multiplication by (1+episilon) in lieu of a copy.
        m_buffer[j, i_q_p, i_0_f] = nl.multiply(m_previous, 1.0000000000001, mask=negation_mask)
      m_current = m_buffer[j, i_q_p, i_0_f]

      p_local = nl.ndarray((par_dim(B_P_SIZE), LARGE_TILE_SZ), dtype=kernel_dtype)

      ## Compute scaling factor
      alpha = nisa.activation(np.exp, m_previous, bias=-1*m_current, scale=1.0, mask=forward_mask) #,mask=forward_mask)
      o_previous = nl.copy(o_buffer[j-1, i_q_p, i_f_d], mask=forward_mask)
      o_previous_scaled = nl.multiply(o_previous, alpha, mask=forward_mask)

      pv_psum = nl.zeros((par_dim(B_P_SIZE),B_D_SIZE), dtype=np.float32, buffer = nl.psum) # ideally we want to re-use qk_psum buffer here

      i_r_f = nl.arange(REDUCTION_TILE)[None,: ]
      p_partial_sum = nl.ndarray((par_dim(B_P_SIZE), LARGE_TILE_SZ//REDUCTION_TILE),dtype=acc_type)
      # Performing reduction in a larger tile to avoid having too many small
      # Vector Engine instructions
      for k_r_i in nl.affine_range(LARGE_TILE_SZ//REDUCTION_TILE):
        p_local[i_q_p, k_r_i*REDUCTION_TILE+i_r_f] = nisa.activation(np.exp, qk_res_buf[i_q_p, k_r_i*REDUCTION_TILE+i_r_f],bias=-1*m_current, scale=1.0, dtype=kernel_dtype, mask=forward_mask)
        p_partial_sum[i_q_p, k_r_i] = nl.sum(p_local[i_q_p, k_r_i*REDUCTION_TILE+i_r_f], axis=1, dtype=acc_type, mask=forward_mask)

      v_local = nl.ndarray((LARGE_TILE_SZ//B_P_SIZE, par_dim(B_P_SIZE), B_D_SIZE), dtype=kernel_dtype)
      for k_i in nl.affine_range(LARGE_TILE_SZ // B_P_SIZE):
        v_local[k_i, i_q_p, i_f_d] = nl.load(v[batch_id, head_id, j*LARGE_TILE_SZ + k_i*B_P_SIZE + i_q_p, i_f_d], dtype=kernel_dtype)
        pv_psum[i_q_p,i_f_d] += matmul(p_local[i_q_p, k_i*B_P_SIZE+i_f_128],
                                          v_local[k_i, i_q_p, i_f_d], transpose_x=False, mask=forward_mask) # (128, 128) (p(Br), d)

      if use_causal_mask:
        o_previous = nl.copy(o_buffer[j-1, i_q_p, i_f_d], mask=negation_mask)
        # FIXME: GIGANTIC HACK! Copy o_buffer[j-1, :, :] to o_buffer[j, :, :] triggers error in the compiler
        # use multiplication by (1+episilon) in lieu of a copy.
        o_buffer[j, i_q_p, i_f_d] = nl.multiply(o_previous, 1.000000000001, mask=negation_mask)
      o_buffer[j, i_q_p, i_f_d] = nl.add(o_previous_scaled, pv_psum, mask=forward_mask)

      p_s = nl.sum(p_partial_sum, 1, dtype=acc_type, mask=forward_mask)
      l_prev = l_buffer[j-1, i_q_p, i_0_f]
      l_exp = nl.add(nl.exp(nl.subtract(l_prev, m_current, mask=forward_mask), mask=forward_mask), p_s, mask=forward_mask)

      l_buffer[j, i_q_p, i_0_f] = nl.add(m_current, nl.log(l_exp, mask=forward_mask), mask=forward_mask)

      if use_causal_mask:
        l_prev = nl.copy(l_buffer[j-1, i_q_p, i_0_f], mask=negation_mask)
        # FIXME: GIGANTIC HACK! Copy l_buffer[j-1, :, :] to l_buffer[j, :, :] triggers error in the compiler
        # use multiplication by (1+episilon) in lieu of a copy.
        l_buffer[j, :, :] = nl.multiply(l_prev, 1.000000000001, mask=negation_mask)

    out = nl.ndarray((par_dim(B_P_SIZE), B_D_SIZE), dtype=kernel_dtype)
    lse_local = nl.zeros((par_dim(B_P_SIZE), 1), dtype=kernel_dtype)

    # -------- write output to buffer on HBM ------------ #
    out[i_q_p, i_f_d] = nl.multiply(o_buffer[num_large_k_tile-1,i_q_p, i_f_d],
                                      nl.exp(m_buffer[num_large_k_tile-1, i_q_p, i_0_f] - l_buffer[num_large_k_tile-1, i_q_p, i_0_f]),
                                      dtype=kernel_dtype)
    lse_local[i_q_p, i_0_f] = nl.copy(l_buffer[num_large_k_tile-1, i_q_p, i_0_f], dtype=kernel_dtype)

    nl.store(o[batch_id, head_id, i*B_P_SIZE + i_q_p, i_f_d], out[i_q_p, i_f_d])
    nl.store(lse[batch_id, head_id, i_q_p, i + i_0_f], lse_local[i_q_p, i_0_f])


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
        o = nki_call(flash_fwd, q, k, v, grid=(grid_x, grid_y), output_HloShapes=[q.dtype[sh] for sh in out_shapes])
        return o     
    
    result = build_run(mixed_pyhlo_nki, (q, k, v))
    print(result)
