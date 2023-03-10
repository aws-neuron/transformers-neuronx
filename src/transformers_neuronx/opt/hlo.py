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
from transformers_neuronx import compiler
from transformers_neuronx.gpt2 import hlo as gpt2_hlo
from transformers_neuronx.opt.config import opt_config_to_gpt2_config


def build_opt_multi_layer_hlo_module(config, n_active_tokens, n_positions, n_layers):
    config = opt_config_to_gpt2_config(config)
    multi_layer = gpt2_hlo.gen_scribable_multi_block(config, n_active_tokens, n_positions, n_layers)
    return compiler.compile_py_func(multi_layer)


def build_ln_lm_head_hlo_module(config, n_active_tokens):
    config = opt_config_to_gpt2_config(config)
    ln_lm_head = gpt2_hlo.gen_scribable_ln_lm_head(config, n_active_tokens)
    return compiler.compile_py_func(ln_lm_head)


def build_opt_hlo_module(config, n_active_tokens, n_positions, blocks_u8_bounds=None):
    config = opt_config_to_gpt2_config(config)
    gpt2 = gpt2_hlo.gen_scribable_gpt2(config, n_active_tokens, n_positions, blocks_u8_bounds)
    return compiler.compile_py_func(gpt2)
