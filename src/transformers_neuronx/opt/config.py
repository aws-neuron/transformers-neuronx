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
import transformers
from transformers.configuration_utils import PretrainedConfig
from transformers_neuronx import utils
from transformers_neuronx.gpt2.config import GPT2Config


class OPTConfig(transformers.OPTConfig):
    def __init__(self, config, n_positions, batch_size, amp, tp_degree , **kwargs):
        if "do_layer_norm_before" in kwargs and not kwargs["do_layer_norm_before"]:
            raise NotImplementedError('do_layer_norm_before=False not implemented')
        kwargs.update(config.to_dict())
        super().__init__(**kwargs)

        self.n_positions = n_positions
        self.batch_size = batch_size
        self.amp = amp
        self.tp_degree = tp_degree


def opt_config_to_gpt2_config(config):
    if config.ffn_dim != 4 * config.hidden_size:
        raise NotImplementedError(f'ffn_dim={config.ffn_dim} and hidden_size={config.hidden_size}')
    gpt2_config = PretrainedConfig()
    gpt2_config.activation_function = config.activation_function
    gpt2_config.n_ctx = config.max_position_embeddings
    gpt2_config.n_embd = config.hidden_size
    gpt2_config.n_head = config.num_attention_heads
    gpt2_config.n_layer = config.num_hidden_layers
    gpt2_config.n_positions = config.n_positions
    gpt2_config.vocab_size = config.vocab_size
    gpt2_config.eos_token_id = config.eos_token_id
    batch_size = config.batch_size
    amp = config.amp
    tp_degree = config.tp_degree
    return GPT2Config(gpt2_config, batch_size, amp, tp_degree)
