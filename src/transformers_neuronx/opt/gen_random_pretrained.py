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
from transformers.models.opt import OPTConfig
from transformers_neuronx.module import sanitize_file_name, _KEY_TO_FILENAME_JSON


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('name', help="OPT model name or path to config.json")
    parser.add_argument('save', help="target folder to save the model")
    parser.add_argument('--empty', action='store_true')
    args = parser.parse_args()
    gen_random_pretrained(args.name, args.save, args.empty)


def gen_random_pretrained(model_name, save, empty=False):
    if 'json' in model_name:
        config = json.load(open(model_name))
    elif model_name == 'facebook/opt-175b':
        config = opt_175b_config()
    else:
        config = OPTConfig.from_pretrained(model_name).to_dict()
    os.makedirs(save, exist_ok=True)
    with open(os.path.join(save, 'config.json'), 'w') as fp:
        json.dump(config, fp, indent=2)
    vocab_size = config['vocab_size']
    hidden_size = config['hidden_size']
    max_position_embeddings = config['max_position_embeddings']
    ffn_dim = config['ffn_dim']
    num_hidden_layers = config['num_hidden_layers']
    init_std = config['init_std']
    torch_dtype = config['torch_dtype']
    name2shape = {
        'model.decoder.embed_tokens.weight': [vocab_size, hidden_size],
        'model.decoder.embed_positions.weight': [max_position_embeddings + 2, hidden_size],
        'model.decoder.final_layer_norm.weight': [hidden_size],
        'model.decoder.final_layer_norm.bias': [hidden_size],
    }
    layer_name2shape = {
        'self_attn.k_proj.weight': [hidden_size, hidden_size],
        'self_attn.k_proj.bias': [hidden_size],
        'self_attn.v_proj.weight': [hidden_size, hidden_size],
        'self_attn.v_proj.bias': [hidden_size],
        'self_attn.q_proj.weight': [hidden_size, hidden_size],
        'self_attn.q_proj.bias': [hidden_size],
        'self_attn.out_proj.weight': [hidden_size, hidden_size],
        'self_attn.out_proj.bias': [hidden_size],
        'self_attn_layer_norm.weight': [hidden_size],
        'self_attn_layer_norm.bias': [hidden_size],
        'fc1.weight': [ffn_dim, hidden_size],
        'fc1.bias': [ffn_dim],
        'fc2.weight': [hidden_size, ffn_dim],
        'fc2.bias': [hidden_size],
        'final_layer_norm.weight': [hidden_size],
        'final_layer_norm.bias': [hidden_size],
    }
    for idx in range(num_hidden_layers):
        for name, shape in layer_name2shape.items():
            name2shape[f'model.decoder.layers.{idx}.{name}'] = shape
    name2shape['lm_head.weight'] = [vocab_size, hidden_size]
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
                'init_std': factor,
            }
            with open(save_path, 'w') as fp:
                json.dump(empty_json, fp, indent=2)
            continue
        init_param = factor * torch.randn(shape)
        init_param = init_param.to(dtype)
        torch.save(init_param, save_path)
        print(f'done saving {save_path}')


def opt_175b_config():
    vocab_size = 50272
    hidden_size = 12288
    max_position_embeddings = 2048
    ffn_dim = 49152
    num_hidden_layers = 96
    init_std = 0.02
    config = dict(
        _name_or_path='facebook/opt-175b',
        _remove_final_layer_norm=False,
        activation_dropout=0.0,
        activation_function='relu',
        architectures=['OPTForCausalLM'],
        attention_dropout=0.0,
        bos_token_id=2,
        do_layer_norm_before=True,
        dropout=0.1,
        eos_token_id=2,
        ffn_dim=ffn_dim,
        hidden_size=hidden_size,
        init_std=init_std,
        layerdrop=0.0,
        max_position_embeddings=max_position_embeddings,
        model_type='opt',
        num_attention_heads=96,
        num_hidden_layers=num_hidden_layers,
        output_projection=True,
        pad_token_id=1,
        prefix='</s>',
        torch_dtype='float16',
        transformers_version='4.23.1',
        use_cache=True,
        vocab_size=vocab_size,
        word_embed_proj_dim=hidden_size,
    )
    return config
