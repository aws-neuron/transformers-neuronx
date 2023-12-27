import math
import random
import torch
from dataclasses import dataclass

def create_blk_mask(blks_q, blks_kv, num_global_blks=0, num_local_blks=1, num_random_blks=0, causal=False):
    """
    Create a block mask given the configs, and the number of blocks in each dimension.
    Assume all heads use the same mask.
    """
    blk_mask = torch.zeros((blks_q, blks_kv), dtype=torch.bool)
    # Add global blocks
    if num_global_blks > 0:
        blk_mask[:num_global_blks, :] = 1
        blk_mask[:, :num_global_blks] = 1
    # Add local blocks
    if num_local_blks > 0:
        width = num_local_blks // 2
        for row in range(blks_q):
            start = max(0, row - width)
            end = min(row + width + 1, blks_kv)
            blk_mask[row, start:end] = 1
    # Add random blocks
    if num_random_blks > 0:
        assert blks_kv > num_random_blks, "Number of random blocks must be smaller than total number of col blocks!"
        for row in range(blks_q):
            # If causal, only draw blocks from the lower-triangular part of the matrix
            pool = list(range(0, blks_kv)) if not causal else list(range(0, row+1))
            selected = pool if len(pool) <= num_random_blks else random.sample(pool, num_random_blks)
            for col in selected:
                blk_mask[row, col] = 1

    if causal:
        blk_mask = torch.tril(blk_mask)

    return blk_mask

def build_dense_mask(q_seq_len, k_seq_len, mask, blk_size=128, causal=False):
    row_blks = (q_seq_len + blk_size - 1) // blk_size
    col_blks = (k_seq_len + blk_size - 1) // blk_size
    assert tuple(mask.shape) == (row_blks, col_blks), f'Mask must have shape (q_seq_len // blk_size, k_seq_len // blk_size)'
    dense_mask = torch.zeros((q_seq_len, k_seq_len), dtype=torch.bool)

    for row_id in range(row_blks):
        for col_id in range(col_blks):
            if int(mask[row_id, col_id]) == 1:
                last_row = min(q_seq_len, (row_id+1)*blk_size)
                last_col = min(k_seq_len, (col_id+1)*blk_size)
                dense_mask[row_id*blk_size:last_row, col_id*blk_size:last_col] = 1
    if causal:
        dense_mask = torch.tril(dense_mask)
    return dense_mask


@dataclass
class BlkSparseAttnConfig:
    """ Block-sparse attention specific config """
    blk_size: int = 128
    num_global_blks: int = 0
    num_local_blks: int = 1
    num_random_blks: int = 0

@dataclass
class SlidingWindowAttnConfig:
    """ Sliding-window attention specific config """
    window_size: int = 128


class SparseAttnConfig:
    """
    The config class that contains sparse attention related settings
    - attn_type: string, must be in the list defined by ATTN_TYPE_LIST
    - sparse_attn_config: must be either BlkSparseAttnConfig, SlidingWindowAttnConfig,
      or None, the config type must match the attention type specified by attn_type
    - same_mask_per_layer: bool, set to False if user want to use a different mask
      in each layer of the model
    - sparse_mask_dict: dict, override the default when attn_type == 'custom'. If
      same_mask_per_layer == True, the dict type is (q_seq_len, kv_seq_len) -->
      torch.Tensor. Otherwise, the key type is (layer_id, q_seq_len, kv_seq_len).
      Values must be boolean tensors with shape (q_seq_len, kv_seq_len).
    """

    def __init__(self, attn_type='blk_sparse', causal=False,
                 sparse_attn_config=None,
                 # This flag controls whether we use the same mask in every layer
                 same_mask_per_layer=True,
                 # User can directly provide the masks if needed
                 sparse_mask_dict=dict()):
        ATTN_TYPE_LIST = ['blk_sparse', 'window', 'custom']
        assert attn_type in ATTN_TYPE_LIST, f'Supported attention types are: {ATTN_TYPE_LIST}'

        if attn_type == 'blk_sparse':
            assert sparse_attn_config and isinstance(sparse_attn_config, BlkSparseAttnConfig), \
                "Must provide a valid block sparse attention config!"
        elif attn_type == 'window':
            assert sparse_attn_config and isinstance(sparse_attn_config, SlidingWindowAttnConfig), \
                "Must provide a valid sliding window attention config!"

        self.sparse_mask_dict = sparse_mask_dict
        self.sparse_attn_config = sparse_attn_config
        self.attn_type = attn_type
        self.causal = causal
        self.same_mask_per_layer = same_mask_per_layer
        self.skip_masking_decode = (self.attn_type == 'window')

    def create_blk_sparse_mask(self, q_seq_len, kv_seq_len):
        blks_q = math.ceil(q_seq_len / self.sparse_attn_config.blk_size)
        blks_kv = math.ceil(kv_seq_len / self.sparse_attn_config.blk_size)
        blk_mask = create_blk_mask(
            blks_q, blks_kv,
            self.sparse_attn_config.num_global_blks,
            self.sparse_attn_config.num_local_blks,
            self.sparse_attn_config.num_random_blks,
            self.causal and (q_seq_len != 1)
        )
        dense_mask = build_dense_mask(
            q_seq_len, kv_seq_len,
            blk_mask, self.sparse_attn_config.blk_size,
            self.causal and (q_seq_len != 1)
        )
        return dense_mask.detach()

    def create_sliding_window_mask(self, q_seq_len, kv_seq_len):
        dense_mask = torch.zeros((q_seq_len, kv_seq_len), dtype=torch.bool)
        # In causal mode we only attend to tokens on the left
        window_size_l, window_size_r = (self.sparse_attn_config.window_size, 0) if self.causal \
            else (self.sparse_attn_config.window_size // 2, self.sparse_attn_config.window_size // 2)

        for row in range(q_seq_len):
            start = max(0, row-window_size_l)
            end = row if self.causal else min(kv_seq_len, row+window_size_r)
            dense_mask[row, start:end] = 1
        return dense_mask.detach()

    def create_sparse_mask(self, q_seq_len, kv_seq_len, layer_id=0):
        """ Create a mask that defines how the new tokens attend to the old tokens """
        assert ((q_seq_len == 1) or (q_seq_len == kv_seq_len)), \
            "Only supporting decode mode (q_seq_len=1) or self-attention mode (q_seq_len=k_seq_len)!"
        key = (q_seq_len, kv_seq_len) if self.same_mask_per_layer else (layer_id, q_seq_len, kv_seq_len)
        if key in self.sparse_mask_dict:
            return self.sparse_mask_dict[key]

        # Don't generate mask if q_seq_len = 1 (decode) and user is using window attention
        skip_masking = (q_seq_len == 1) and self.skip_masking_decode
        if skip_masking:
            return None

        if self.attn_type == 'blk_sparse':
            mask = self.create_blk_sparse_mask(q_seq_len, kv_seq_len)
        elif self.attn_type == 'window':
            mask = self.create_sliding_window_mask(q_seq_len, kv_seq_len)
        self.sparse_mask_dict[key] = mask
        return mask
