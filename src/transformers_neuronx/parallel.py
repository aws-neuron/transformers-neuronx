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
from concurrent.futures import ThreadPoolExecutor
import torch
from transformers_neuronx import ops


def parallel_load(models):
    degree = len(models)
    with ThreadPoolExecutor(degree) as executor:
        futures = []
        for ordinal, model in enumerate(models):
            args = model, ordinal, 1, ordinal, degree
            fut = executor.submit(ops.load_collectives, *args)
            futures.append(fut)
        [future.result() for future in futures]  # wait for load_collectives calls


class TensorManipulator:

    def __init__(self, tp_degree):
        self.tp_degree = tp_degree

    def duplicate(self, tensor):
        tensors = [tensor for ordinal in range(self.tp_degree)]
        return to_nc(tensors)

    def shard_along(self, tensor, dim):
        size = tensor.shape[dim]
        shard_size = size // self.tp_degree
        slices = [slice(None) for _ in tensor.shape]
        tensors = []
        for start in range(0, size, shard_size):
            slices[dim] = slice(start, start+shard_size, 1)
            shard = tensor[tuple(slices)].contiguous()
            tensors.append(shard)
        return to_nc(tensors)

    def primary_only(self, tensor):
        tensors = [tensor]
        tensors.extend(torch.zeros_like(tensor) for _ in range(1, self.tp_degree))
        return to_nc(tensors)

    def unshard_along(self, sharded_tensors, dim):
        return torch.cat(cpu(sharded_tensors), dim=dim)

    def slice_on_nc(self, tensors, dim, start, end, step):
        return [ops.slice(ts, dim, start, end, step) for ts in tensors]


def to_nc(sharded_tensors):
    return [ops.to_nc(ts, ordinal) for ordinal, ts in enumerate(sharded_tensors)]


def cpu(sharded_tensors):
    return [ops.cpu(ts) for ts in sharded_tensors]


class Executor:

    def __init__(self, tp_degree):
        self.executor = ThreadPoolExecutor(tp_degree)

    def execute(self, models, *inputs_cores):
        futures = []
        for model, *inputs in zip(models, *inputs_cores):
            fut = self.executor.submit(ops.execute, model, inputs)
            futures.append(fut)
        cores_outputs = [fut.result() for fut in futures]
        outputs_cores = [list(outputs) for outputs in zip(*cores_outputs)]
        return outputs_cores
