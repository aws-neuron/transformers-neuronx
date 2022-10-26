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
import json
import math
import os
import re
import time
import torch
from transformers import AutoConfig


def save_pretrained_split(model, save_directory):
    model.save_pretrained(save_directory, save_function=save_split, max_shard_size='10000GB')


_KEY_TO_FILENAME_JSON = 'key_to_filename.json'


def save_split(state_dict, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    key_to_filename = {}
    for idx, key in enumerate(state_dict.keys()):
        key_to_filename[key] = f'p{idx}.{sanitize_file_name(key)}'
    with open(os.path.join(save_dir, _KEY_TO_FILENAME_JSON), 'w') as f:
        json.dump(key_to_filename, f, indent=2)
    for key, tensor in state_dict.items():
        torch.save(tensor, os.path.join(save_dir, key_to_filename[key]))


def sanitize_file_name(name):
    sanitized = name.strip().replace(' ', '_')
    sanitized = re.sub(r'(?u)[^-\w.]', '', sanitized)
    if sanitized in {'', '.', '..'}:
        raise ValueError(f'Could not sanitize "{name}" to file name.')
    return sanitized


class LowMemoryModule(torch.nn.Module):

    def materialize(self):
        with torch.no_grad():
            for param in self.parameters():
                if not hasattr(param, '_file_path'):
                    continue
                input_param = torch.load(param._file_path)
                if torch.nn.parameter.is_lazy(param):
                    param.materialize(input_param.shape)
                param.copy_(input_param)

    def load_state_dict_low_memory(self, state_dict):

        def load(module, prefix=''):
            module._load_from_state_dict_low_memory(state_dict, prefix)
            for name, child in module.named_modules():
                if child is module:
                    continue
                if child is not None:
                    load(child, prefix + name + '.')

        load(self)

    def _load_from_state_dict_low_memory(self, state_dict, prefix):
        local_state = {k: v for k, v in self.named_parameters() if v is not None}
        for name, param in local_state.items():
            key = prefix + name
            if key in state_dict:
                input_param = state_dict.pop(key)
                with torch.no_grad():
                    if torch.nn.parameter.is_lazy(param):
                        param.materialize(input_param.shape)
                    param.copy_(input_param)

    def load_state_dict_dir(self, state_dict_dir):
        with open(os.path.join(state_dict_dir, _KEY_TO_FILENAME_JSON)) as f:
            key_to_filename = json.load(f)
        state_dict_dir = os.path.realpath(state_dict_dir)

        def load(module, key_to_filename, prefix=''):
            module._load_from_state_dict_dir(key_to_filename, state_dict_dir, prefix)
            for name, child in module.named_modules():
                if child is module:
                    continue
                if child is not None:
                    load(child, prefix + name + '.')

        load(self, key_to_filename)

    def _load_from_state_dict_dir(self, key_to_filename, state_dict_dir, prefix):
        local_state = {k: v for k, v in self.named_parameters() if v is not None}
        for name, param in local_state.items():
            key = prefix + name
            if key in key_to_filename:
                param._file_path = os.path.join(state_dict_dir, key_to_filename[key])


class LowMemoryModuleList(torch.nn.ModuleList, LowMemoryModule): ...
class LowMemoryEmbedding(torch.nn.Embedding, LowMemoryModule): ...
class LowMemoryLayerNorm(torch.nn.LayerNorm, LowMemoryModule): ...
class LowMemoryLazyLinear(torch.nn.LazyLinear, LowMemoryModule): ...


class PretrainedModel(LowMemoryModule):

    @classmethod
    def from_pretrained(cls, pretrained_model_path, *model_args, print_latency=False, **kwargs):
        config = AutoConfig.from_pretrained(pretrained_model_path)
        model = cls(config, *model_args, **kwargs)
        state_dict_path = os.path.join(pretrained_model_path, 'pytorch_model.bin')
        if os.path.isdir(state_dict_path):
            model.load_state_dict_dir(state_dict_path)
        else:
            state_dict = torch.load(state_dict_path)
            model.load_state_dict_low_memory(state_dict)
        if print_latency:
            latency_printer = LatencyPrinter()
            model.register_forward_pre_hook(latency_printer.pre_hook)
            model.register_forward_hook(latency_printer.hook)
        return model


class LatencyPrinter:

    def __init__(self):
        self.start = None

    def pre_hook(self, *args):
        self.start = time.time()

    def hook(self, *args):
        latency_ms = math.ceil((time.time() - self.start) * 1000)
        print(f'Latency: {latency_ms} ms')
