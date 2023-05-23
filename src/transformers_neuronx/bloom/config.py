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

class BloomConfig:

    def __init__(
            self,
            config,
            n_positions,
            batch_size,
            amp,
            tp_degree,
            activation='gelu',
            **kwargs
        ):

        # Extract configs used for building HLO
        self.embed_dim = config.hidden_size
        self.hidden_size = config.hidden_size
        self.attention_head_size = config.hidden_size // config.n_head
        self.n_head = config.n_head
        self.n_layer = config.num_hidden_layers
        self.vocab_size = config.vocab_size
        self.layer_norm_epsilon = config.layer_norm_epsilon
        self.bos_token_id = config.bos_token_id
        self.eos_token_id = config.eos_token_id

        utils.maybe_override_attributes(self, kwargs)

        # Add required Neuron configs
        self.n_positions = n_positions
        self.batch_size = batch_size
        self.amp = amp
        self.tp_degree = tp_degree
        self.activation = activation
