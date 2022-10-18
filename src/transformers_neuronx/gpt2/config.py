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


class GPT2Config:

    def __init__(self, config, batch_size, n_active_tokens, amp, tp_degree, **kwargs):
        self.activation_function = config.activation_function
        self.n_ctx = config.n_ctx
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.n_layer = config.n_layer
        self.n_positions = config.n_positions
        self.vocab_size = config.vocab_size
        self.eos_token_id = config.eos_token_id
        utils.maybe_override_attributes(self, kwargs)
        self.intermediate_dim = self.n_embd * 4
        self.batch_size = batch_size
        self.n_active_tokens = n_active_tokens
        self.amp = amp
        self.tp_degree = tp_degree
