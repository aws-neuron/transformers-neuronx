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

import os
os.environ['NEURON_RT_NUM_CORES'] = '1' # Dev Optimization: Only initialize one NeuronCore
from typing import Callable, Tuple, Union

import torch

from transformers_neuronx import compiler

# -----------------------------------------------------------------------------
# Testing Utilities
# -----------------------------------------------------------------------------

class PyHloSingleCoreExecutable(torch.nn.Module):

    def __init__(self, hlo_module) -> None:
        super().__init__()
        neff = compiler.compile_hlo_module(hlo_module)
        metaneff  = compiler.hlo2metaneff(hlo_module)
        metaneff = metaneff.SerializeToString()
        self.model = torch.classes.neuron.Model(neff, metaneff)

    def forward(self, *tensors):
        tensors = list(tensors)
        results = torch.ops.neuron.forward_v2(tensors, self.model)
        if len(results) == 1:
            return results[0]
        return results


def lower(func: Callable, inputs: Union[Tuple[torch.Tensor, ...], torch.Tensor]):

    # Build a PyHLO interface from the input tensors
    def interface(scribe):
        dtypes = {
            torch.float32: scribe.f32,
            torch.int32: scribe.s32,
            torch.long: scribe.u64,
            torch.float16: scribe.f16,
            torch.bfloat16: scribe.bf16,
            torch.bool: scribe.pred,
        }

        params = []
        for i, input in enumerate(inputs):
            shape = list(input.shape)
            param = dtypes[input.dtype][shape].Parameter(parameter_number=i)
            params.append(param)

        result = func(*params)
        if isinstance(result, tuple):
            nonnull = [item for item in result if item is not None]
            root_shapes = [item.dtype[item.sizes] for item in nonnull]
            return scribe.tuple(*root_shapes).Tuple(*nonnull)

        return result

    hlo_module = compiler.compile_py_func(interface)
    return hlo_module

def build_run(network, example):
    hlo_module = lower(network, example)
    neuron = PyHloSingleCoreExecutable(hlo_module)
    return neuron(*example)