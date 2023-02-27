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
from transformers_neuronx import utils


class GPTNeoXConfig:

    def __init__(self, config, batch_size, amp, tp_degree, **kwargs):
        self.activation_function = config.hidden_act
        self.n_embd = config.hidden_size
        self.n_head = config.num_attention_heads
        self.n_layer = config.num_hidden_layers
        self.n_positions = config.max_position_embeddings
        self.n_ctx = self.n_positions # TODO - Needed for now because self.n_positions will be overridden by the kwargs n_positions
        """
        rotary_dim calculation
        Reference: transformers/models/gpt_neox/modeling_gpt_neox.py, class "GPTNeoXAttention"
            self.num_attention_heads = config.num_attention_heads
            self.hidden_size = config.hidden_size
            self.head_size = self.hidden_size // self.num_attention_heads
            self.rotary_ndims = int(self.head_size * config.rotary_pct)
        """
        head_size = self.n_embd // self.n_head
        self.rotary_dim = int(head_size * config.rotary_pct)
        self.vocab_size = config.vocab_size
        self.eos_token_id = config.eos_token_id
        self.rotary_emb_base = config.rotary_emb_base
        self.use_parallel_residual = config.use_parallel_residual
        utils.maybe_override_attributes(self, kwargs)
        self.intermediate_dim = config.intermediate_size # `intermediate_size` in GPT-NeoX is given in the config file
        self.batch_size = batch_size
        self.amp = amp
        self.tp_degree = tp_degree
