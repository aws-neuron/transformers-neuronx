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

class MistralConfig:

    def __init__(
            self,
            config,
            n_positions,
            batch_size,
            amp,
            tp_degree,
            **kwargs
        ):

        # Extract configs used for building HLO
        self.intermediate_size = config.intermediate_size
        self.hidden_size = config.hidden_size
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads if hasattr(config, "num_key_value_heads") else config.num_attention_heads
        self.num_hidden_layers = config.num_hidden_layers
        self.vocab_size = config.vocab_size
        self.hidden_act = config.hidden_act
        self.bos_token_id = config.bos_token_id
        self.eos_token_id = config.eos_token_id
        self.max_position_embeddings = config.max_position_embeddings
        self.rms_norm_eps = config.rms_norm_eps
        self.rotary_percentage = getattr(config, "rotary_percentage", 1)
        self.rope_theta = getattr(config, "rope_theta", 10000)
        self.position_interpolation_factor = getattr(config, "position_interpolation_factor", None)
        self.sliding_window = getattr(config, "sliding_window", None)

        utils.maybe_override_attributes(self, kwargs)

        # Add required Neuron configs
        self.n_positions = n_positions
        self.batch_size = batch_size
        self.amp = amp
        self.tp_degree = tp_degree
        self.model_type = 'mistral'

