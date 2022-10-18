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
from transformers_neuronx.gpt2 import hlo as gpt2_hlo
from transformers_neuronx.opt.config import opt_config_to_gpt2_config


def build_opt_block_kernel(config):
    config = opt_config_to_gpt2_config(config)
    return gpt2_hlo.build_gpt2_block_kernel(config)


def build_lm_head_kernel(config):
    config = opt_config_to_gpt2_config(config)
    return gpt2_hlo.build_lm_head_kernel(config)


def build_opt_kernel(config):
    config = opt_config_to_gpt2_config(config)
    return gpt2_hlo.build_gpt2_kernel(config)
