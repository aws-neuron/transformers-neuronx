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
import os
import re
import warnings
from typing import Optional

import torch
from safetensors import safe_open
from torch.nn.parameter import UninitializedParameter
from transformers import AutoConfig

# Disable lazy module warning since torch-neuronx version is pinned
warnings.filterwarnings("ignore", category=UserWarning, module='torch.nn.modules.lazy')

def save_pretrained_split(model, save_directory):
    model.save_pretrained(save_directory, save_function=save_split, max_shard_size='10000GB', safe_serialization=False)


_SAFETENSORS_MODEL_INDEX_FILENAME_JSON = 'model.safetensors.index.json'
_SAFETENSORS_MODEL_FILENAME = 'model.safetensors'
_PYTORCH_MODEL_BIN_INDEX_FILENAME_JSON = 'pytorch_model.bin.index.json'
_PYTORCH_MODEL_BIN_FILENAME = 'pytorch_model.bin'
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
                if param._file_path.endswith('.empty_json'):
                    with open(param._file_path) as fp:
                        empty_json = json.load(fp)
                    torch.manual_seed(0)
                    input_param = empty_json.get('init_std', 1.0) * torch.randn(empty_json['shape'])
                    dtype = getattr(torch, empty_json['torch_dtype'])
                    input_param = input_param.to(dtype)
                elif param._file_path.endswith('.safetensors'):
                    with safe_open(param._file_path, framework="pt") as f:
                        if param._global_key in f.keys():
                            input_param = f.get_tensor(param._global_key)
                        else:
                            raise FileNotFoundError(f'Could not find a weight for {param._global_key} in {param._file_path}')
                else:
                    input_param = torch.load(param._file_path)
                if torch.nn.parameter.is_lazy(param):
                    param.materialize(input_param.shape)
                param.copy_(input_param)

    def nullify(self):

        def _nullify(module):
            for name, param in module.named_parameters():
                if '.' not in name and hasattr(module, name):
                    blank = UninitializedParameter()
                    # Note: Allow the parameter to be reloaded
                    if hasattr(param, '_file_path'):
                        blank._file_path = param._file_path
                        blank._global_key = param._global_key
                    setattr(module, name, blank)
            for child in module.modules():
                if child is module:
                    continue
                if child is not None:
                    _nullify(child)

        _nullify(self)

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

    def _get_ties_mapping(self):
        mapping = {}
        for first, second in self.get_tied_parameter_paths():
            mapping[first] = second
            mapping[second] = first
        return mapping

    def _load_from_state_dict_dir(self, state_dict_dir, key_to_filename, prefix=''):
        state_dict_dir = os.path.realpath(state_dict_dir)
        ties = self._get_ties_mapping()
        local_state = {k: v for k, v in self.named_parameters() if v is not None}
        for key, param in local_state.items():
            key = prefix + key
            # If the key is not present but it has a tie, attept to use the tie
            if key not in key_to_filename:
                key = ties.get(key, '')

            if key in key_to_filename:
                path = os.path.join(state_dict_dir, key_to_filename[key])
                param._file_path = path
                param._global_key = key

    def load_pytorch_model_bin(self, state_dict_dir):
        """
        Eagerly load the pytorch model binary artifact.
        """
        state_dict_path = os.path.join(state_dict_dir, _PYTORCH_MODEL_BIN_FILENAME)
        state_dict = torch.load(state_dict_path)
        self.load_state_dict_low_memory(state_dict)

    def load_pytorch_model_bin_sharded(self, state_dict_dir):
        """
        Eagerly load the the pytorch model binary shards.
        """
        index = os.path.join(state_dict_dir, _PYTORCH_MODEL_BIN_INDEX_FILENAME_JSON)
        with open(index, 'r') as f:
            key_to_filename = json.load(f)["weight_map"]
        shard_filenames = set(key_to_filename.values())
        for shard_filename in shard_filenames:
            path = os.path.join(state_dict_dir, shard_filename)
            state_dict = torch.load(path)
            self.load_state_dict_low_memory(state_dict)

    def load_safetensors(self, state_dict_dir):
        """
        Lazily load the safetensors by associating each weight with the filename.
        """
        filename = os.path.join(state_dict_dir, _SAFETENSORS_MODEL_FILENAME)
        with safe_open(filename, framework="pt") as f:
            keys = f.keys()
        key_to_filename = dict(zip(keys, [_SAFETENSORS_MODEL_FILENAME] * len(keys)))
        self._load_from_state_dict_dir(state_dict_dir, key_to_filename)

    def load_safetensors_sharded(self, state_dict_dir):
        """
        Lazily load the safetensors by associating each weight with a shard filename.
        """
        index = os.path.join(state_dict_dir, _SAFETENSORS_MODEL_INDEX_FILENAME_JSON)
        with open(index, 'r') as f:
            key_to_filename = json.load(f)["weight_map"]
        self._load_from_state_dict_dir(state_dict_dir, key_to_filename)

    def load_split_checkpoint(self, state_dict_dir):
        """
        Lazily load the manually split checkpoint by associating each weight with the weight filename.
        """
        weight_directory = os.path.join(state_dict_dir, _PYTORCH_MODEL_BIN_FILENAME)
        with open(os.path.join(weight_directory, _KEY_TO_FILENAME_JSON)) as f:
            key_to_filename = json.load(f)
        self._load_from_state_dict_dir(weight_directory, key_to_filename)

    def get_tied_parameter_paths(self):
        """
        Get parameter path pairs whose weights are identical.

        Note that this is only necessary for safetensors models because at
        serialization time `transformers` does not save tensors which are
        identical to the "model.safetensors" file. Instead they save only one
        weight and then check which parameters are tied according to the
        parameter structure at load time. Since we do not load the original
        parameter structure, we need an alternative method of determining
        "ties".
        """
        return []

class LowMemoryModuleList(torch.nn.ModuleList, LowMemoryModule): ...
class LowMemoryLazyLinear(torch.nn.LazyLinear, LowMemoryModule): ...

class LowMemoryLayerNorm(torch.nn.LayerNorm, LowMemoryModule):

    def __init__(self, normalized_shape, eps: float = 1e-5, elementwise_affine: bool = True,
                 device=None, dtype=None) -> None:
        super(LowMemoryLayerNorm, self).__init__(0)
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.weight = UninitializedParameter()
        self.bias = UninitializedParameter()

    def reset_parameters(self) -> None:
        pass


class LowMemoryEmbedding(torch.nn.Embedding, LowMemoryModule):

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None,
                 max_norm: Optional[float] = None, norm_type: float = 2., scale_grad_by_freq: bool = False,
                 sparse: bool = False, _weight: Optional[torch.Tensor] = None,
                 device=None, dtype=None) -> None:
        super(LowMemoryEmbedding, self).__init__(0, 0)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        if padding_idx is not None:
            if padding_idx < 0:
                padding_idx = self.num_embeddings + padding_idx
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse
        self.weight = UninitializedParameter()

    def reset_parameters(self) -> None:
        pass


class PretrainedModel(LowMemoryModule):

    @classmethod
    def from_pretrained(cls, pretrained_model_path, *model_args, **kwargs):

        def _sanity_check(**kwargs):
            context_length_estimate = kwargs.get("context_length_estimate", None)
            n_positions = kwargs.get("n_positions", 2048)
            neuron_config = kwargs.get("neuron_config", None)
            continuous_batching = neuron_config and neuron_config.continuous_batching
            if continuous_batching:
                batch_size_for_shared_caches = neuron_config.continuous_batching.batch_size_for_shared_caches
                expected_batch_size = kwargs.get("batch_size")
                assert batch_size_for_shared_caches == expected_batch_size, \
                    f"invalid batch_size_for_shared_caches ({batch_size_for_shared_caches}), {expected_batch_size} is expected"
                assert isinstance(context_length_estimate, list) and len(context_length_estimate) == 1
                assert isinstance(n_positions, list) and len(n_positions) == 1
                assert context_length_estimate == n_positions, \
                    "To use continuous batching features, context length estimate should equal to n_positions."

        _sanity_check(**kwargs)
        config = AutoConfig.from_pretrained(pretrained_model_path)
        model = cls(config, *model_args, **kwargs)
        model.load_state_dict_dir(pretrained_model_path)
        return model

    def load_state_dict_dir(self, pretrained_model_path):

        # Standard checkpoint filenames
        state_dict_path = os.path.join(pretrained_model_path, _PYTORCH_MODEL_BIN_FILENAME)
        state_dict_safetensor_path = os.path.join(pretrained_model_path, _SAFETENSORS_MODEL_FILENAME)
        safetensors_index_path = os.path.join(pretrained_model_path, _SAFETENSORS_MODEL_INDEX_FILENAME_JSON)
        pytorch_model_bin_index_path = os.path.join(pretrained_model_path, _PYTORCH_MODEL_BIN_INDEX_FILENAME_JSON)

        # Loading is done in priority of fastest -> slowest (in case multiple variants exist)

        # Non-sharded safetensors checkpoint
        if os.path.isfile(state_dict_safetensor_path):
            self.load_safetensors(pretrained_model_path)
        # Sharded safetensors checkpoint
        elif os.path.exists(safetensors_index_path):
            self.load_safetensors_sharded(pretrained_model_path)
        # Manually split `save_pretrained_split` checkpoint
        elif os.path.isdir(state_dict_path):
            self.load_split_checkpoint(pretrained_model_path)
        # Non-sharded pytorch_model.bin checkpoint
        elif os.path.isfile(state_dict_path):
            self.load_pytorch_model_bin(pretrained_model_path)
        # Sharded pytorch model bin
        elif os.path.isfile(pytorch_model_bin_index_path):
            self.load_pytorch_model_bin_sharded(pretrained_model_path)
        else:
            raise FileNotFoundError(f"Can not find model.safetensors or pytorch_model.bin in {pretrained_model_path}")


class WrappingCheckpointCompatibleModel(PretrainedModel):

    def __init__(self, chkpt_model_cls, *args, **kwargs):
        super().__init__()
        self.chkpt_model = chkpt_model_cls(*args, **kwargs)

    def load_state_dict_dir(self, pretrained_model_path):
        self.chkpt_model.load_state_dict_dir(pretrained_model_path)
