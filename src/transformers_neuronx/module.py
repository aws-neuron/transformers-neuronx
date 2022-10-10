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
import math
import os
import time
import torch
from transformers import AutoConfig


class LowMemoryModule(torch.nn.Module):

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
