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

import argparse
import json
import os
import torch
from transformers.models.gpt2 import GPT2Config
from transformers_neuronx.module import sanitize_file_name, _KEY_TO_FILENAME_JSON
from transformers.configuration_utils import PretrainedConfig

def gen_random_pretrained(model_name, save, empty=False, print_shapes=False):
    if '.json' in model_name:
        config = json.load(open(model_name))
    else:
        config = GPT2Config.from_pretrained(model_name).to_dict()
    os.makedirs(save, exist_ok=True)
    with open(os.path.join(save, 'config.json'), 'w') as fp:
        json.dump(config, fp, indent=2)
    vocab_size = config['vocab_size']
    hidden_size = config['n_embd']
    max_position_embeddings = config['n_ctx']
    ffn_dim = hidden_size #TODO: Check this ffn_dim = hidden_size or ffn_dim = hidden_size * 4 (this fails)
    num_hidden_layers = config['n_layer']
    torch_dtype = config.get('torch_dtype', None)
    if torch_dtype is None:
        torch_dtype = 'float16'
    init_std = 0.001
    name2shape = {
        'transformer.wte.weight': [vocab_size, hidden_size],
        'transformer.wpe.weight': [max_position_embeddings, hidden_size],
    }
    layer_name2shape = {
        'ln_1.weight': [hidden_size],
        'ln_1.bias': [hidden_size],      
        'attn.bias': [hidden_size],
        'attn.masked_bias': [hidden_size],
        'attn.c_attn.weight': [hidden_size, 3 * hidden_size],
        'attn.c_attn.bias': [3 * hidden_size],
        'attn.c_proj.weight': [hidden_size, hidden_size],
        'attn.c_proj.bias': [hidden_size],
        'ln_2.weight': [hidden_size],
        'ln_2.bias': [hidden_size],  
        'mlp.c_fc.weight': [ffn_dim, hidden_size], 
        'mlp.c_fc.bias': [ffn_dim],
        'mlp.c_proj.weight': [hidden_size, ffn_dim],
        'mlp.c_proj.bias': [hidden_size],
    }
    for idx in range(num_hidden_layers):
        for name, shape in layer_name2shape.items():
            name2shape[f'transformer.h.{idx}.{name}'] = shape
    name2shape['transformer.ln_f.weight'] = [hidden_size]
    name2shape['transformer.ln_f.bias'] = [hidden_size]
    name2shape['lm_head.weight'] = [vocab_size, hidden_size]
    if print_shapes:
        print("Components' shapes")
        [print(x) for x in name2shape.items()]
    key_to_filename = {}
    for idx, key in enumerate(name2shape.keys()):
        key_to_filename[key] = f'p{idx}.{sanitize_file_name(key)}'
        if empty:
            key_to_filename[key] = f'{key_to_filename[key]}.empty_json'
    split_param_dir = os.path.join(save, 'pytorch_model.bin')
    os.makedirs(split_param_dir, exist_ok=True)
    with open(os.path.join(split_param_dir, _KEY_TO_FILENAME_JSON), 'w') as fp:
        json.dump(key_to_filename, fp, indent=2)
    dtype = getattr(torch, torch_dtype)
    for name, shape in name2shape.items():
        save_path = os.path.join(split_param_dir, key_to_filename[name])
        factor = 0.0 if 'layer_norm' in name or 'bias' in name else init_std
        if empty:
            empty_json = {
                'torch_dtype': torch_dtype,
                'shape': shape,
            }
            with open(save_path, 'w') as fp:
                json.dump(empty_json, fp, indent=2)
            continue
        init_param = factor * torch.randn(shape)
        init_param = init_param.to(dtype)
        torch.save(init_param, save_path)
        print(f'done saving {save_path}')