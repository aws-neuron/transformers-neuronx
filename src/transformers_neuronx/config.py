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

from transformers_neuronx.utils import build_dense_mask, create_blk_mask

class QuantizationConfig:
    """ The config class that contains all quantization related settings """

    def __init__(self, quant_dtype='s8', dequant_dtype='f16', quantize_method='vector_dynamic',
                 quantize_attn=True):
        QUANT_DTYPE_LIST = ['s8',]
        QUANTIZE_METHOD_LIST = ['vector_dynamic',]

        # The data type that the parameter is quantized into
        self.quant_dtype = quant_dtype
        if self.quant_dtype not in QUANT_DTYPE_LIST:
            raise NotImplementedError(f"{self.quant_dtype} is not implemented. \
                                      Available options are {','.join(QUANT_DTYPE_LIST)}")

        # The data type that is dequantized to
        self.dequant_dtype = dequant_dtype

        # Which quantization algorithm to use
        self.quantize_method = quantize_method
        if self.quantize_method not in QUANTIZE_METHOD_LIST:
            raise NotImplementedError(f"{self.quantize_method} is not implemented. \
                                      Available options are {','.join(QUANTIZE_METHOD_LIST)}")

        # Decide whether the attention layer needs be quantized
        self.quantize_attn = quantize_attn

class SparseAttnConfig:
    """ The config class that contains sparse attention related settings """
    def __init__(self, attn_type='blk_sparse', causal=False,
                 # blk-sparse configs
                 blk_size=128,
                 num_global_blks=0, num_local_blks=1, num_random_blks=0,
                 # User can directly provide the masks if needed
                 # Must provide a dict mapping from seq_len --> mask
                 sparse_mask_dict=None, active_sparse_mask_dict=None):
        ATTN_TYPE_LIST = ['blk_sparse', 'custom']

        assert attn_type in ATTN_TYPE_LIST, f'Supported attention types are: {ATTN_TYPE_LIST}'
        if attn_type == 'blk_sparse':
            self.blk_size = blk_size
            self.num_global_blks = num_global_blks
            self.num_local_blks = num_local_blks
            self.num_random_blks = num_random_blks
            self.sparse_mask_dict = {}
            self.active_sparse_mask_dict = {}
        else:
            self.sparse_mask_dict = sparse_mask_dict
            self.active_sparse_mask_dict = active_sparse_mask_dict
        self.attn_type = attn_type
        self.causal = causal

    def create_sparse_mask(self, q_seq_len, kv_seq_len):
        """ Create a mask that defines how the new tokens attend to the old tokens """
        assert ((q_seq_len == 1) or (q_seq_len == kv_seq_len)), \
            "Only supporting decode mode (q_seq_len=1) or self-attention mode (q_seq_len=k_seq_len)!"
        key = (q_seq_len, kv_seq_len)
        if self.attn_type == 'custom':
            return self.sparse_mask_dict[key]

        # If using blk-sparse, search a cache first
        if key in self.sparse_mask_dict:
            return self.sparse_mask_dict[key]
        # If not found, generate it
        blks_q = math.ceil(q_seq_len / self.blk_size)
        blks_kv = math.ceil(kv_seq_len / self.blk_size)
        blk_mask = create_blk_mask(
            blks_q, blks_kv,
            self.num_global_blks,
            self.num_local_blks,
            self.num_random_blks,
            self.causal and (q_seq_len != 1)
        )
        dense_mask = build_dense_mask(
            q_seq_len, kv_seq_len,
            blk_mask, self.blk_size,
            self.causal and (q_seq_len != 1)
        )
        self.sparse_mask_dict[key] = dense_mask
        return dense_mask.detach()

    def create_active_sparse_mask(self, n_active_tokens):
        """ Create a mask that defines how the new tokens attend to each other """
        if self.attn_type == 'custom':
            return self.active_sparse_mask_dict[n_active_tokens]

        # Same as above, except that we have q_seq_len = 1 now
        if n_active_tokens in self.active_sparse_mask_dict:
            return self.active_sparse_mask_dict[n_active_tokens]
        # If not found, generate it
        blks_q = 1
        blks_kv = math.ceil(n_active_tokens / self.blk_size)
        blk_mask = create_blk_mask(
            blks_q, blks_kv,
            self.num_global_blks,
            self.num_local_blks,
            self.num_random_blks,
            False # causal
        )
        dense_mask = build_dense_mask(
            1, n_active_tokens,
            blk_mask, self.blk_size,
            False # causal
        )
        self.active_sparse_mask_dict[n_active_tokens] = dense_mask
        return dense_mask.detach()


class NeuronConfig():
    """ The class contains all Neuron related configs """
    def __init__(self, **kargs):
        # Quantization related configurations
        self.quant = kargs.pop('quant', None)
        # Sparse attention related configurations
        self.sparse_attn = kargs.pop('sparse_attn', None)

class GenerationConfig:

    def __init__(self, *,
        max_length = None,      # Default: Infer max sequence length from model
        do_sample = False,      # Default: Greedy
        top_k = 50,             # Default: Top 50 (when sampling)
        eos_token_id = None,    # Default: Ignore EOS token
        early_stopping = None,  # Default: Open-ended generation
        temperature = None,     # Default: No temperature application
    ):
        self.max_length = max_length
        self.do_sample = do_sample
        self.top_k = top_k
        self.eos_token_id = eos_token_id
        self.early_stopping = early_stopping
        self.temperature = temperature

