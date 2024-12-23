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


class ParallelTensorManipulator:

    def __init__(self, tp_degree, rank_id=0, local_tp_degree=None):
        self.tp_degree = tp_degree
        self.rank_id = rank_id
        if local_tp_degree is None:
            local_tp_degree = tp_degree
        self.local_tp_degree = local_tp_degree

    def duplicate_on_cpu(self, tensor):
        return [tensor for ordinal in range(self.local_tp_degree)]

    def duplicate(self, tensor):
        return ops.parallel_to_nc([tensor.contiguous() for ordinal in range(self.local_tp_degree)])

    def shard_along_on_cpu(self, tensor, dim):
        size = tensor.shape[dim]
        shard_size = size // self.tp_degree
        slices = [slice(None) for _ in tensor.shape]
        tensors = []
        if self.local_tp_degree != self.tp_degree: # for multi-instance tp
            slice_start = self.rank_id * self.local_tp_degree * shard_size
            slice_end = (self.rank_id + 1) * self.local_tp_degree * shard_size
            for start in range(slice_start, slice_end, shard_size):
                slices[dim] = slice(start, start+shard_size, 1)
                shard = tensor[tuple(slices)].contiguous()
                tensors.append(shard)
        else:
            slice_start = 0
            slice_end = size
            slice_range = range(slice_start, slice_end, shard_size)
            for start in slice_range:
                slices[dim] = slice(start, start+shard_size, 1)
                shard = tensor[tuple(slices)].contiguous()
                if len(slice_range) == 1:
                    # edge case for save_presharded flow where something is "sharded"
                    # but in reality is a no-op causing some tensors to share memory
                    # safetensors cannot share memory so we make a copy
                    shard = shard.clone()
                tensors.append(shard)
        if len(tensors) != self.local_tp_degree:
            raise ValueError(
                f'Weight with shape {tensor.shape} cannot be sharded along dimension {dim}. '
                f'This results in {len(tensors)} weight partitions which cannot be distributed to {self.local_tp_degree} NeuronCores evenly. '
                f'To fix this issue either the model parameters or the `tp_degree` must be changed to allow the weight to be evenly split'
            )
        return tensors

    def shard_along(self, tensor, dim):
        return ops.parallel_to_nc(self.shard_along_on_cpu(tensor, dim))

    def duplicate_or_shard_along(self, tensor, dim):
        if dim is None:
            return self.duplicate(tensor)
        return self.shard_along(tensor, dim)

    def primary_only(self, tensor):
        tensors = [tensor]
        tensors.extend(torch.zeros_like(tensor) for _ in range(1, self.local_tp_degree))
        return ops.parallel_to_nc(tensors)

    def unshard_along(self, sharded_tensors, dim):
        return torch.cat(ops.parallel_cpu(sharded_tensors), dim=dim)

    def slice_on_nc(self, tensors, dim, start, end, step):
        return ops.parallel_slice(tensors, dim, start, end, step)


def layers_to_neuron(num_workers, layers, n_positions_list, to_neuron_hooks):
    with ThreadPoolExecutor(num_workers) as pool:
        futures = [pool.submit(layer.to_neuron, n_positions_list) for layer in layers]
        for idx, future in enumerate(futures):
            future.result()
            for hook in to_neuron_hooks:
                hook(idx)


class CacheBroadcaster:

    def __init__(self, tp_degree, shard_dim, batch_dim, batch_size):
        self.manipulator = ParallelTensorManipulator(tp_degree)
        self.shard_dim = shard_dim
        self.batch_dim = batch_dim
        self.batch_size = batch_size

    def broadcast(self, source, target):
        source_batch_size = source.shape[self.batch_dim]
        source = self.manipulator.unshard_along(source, dim=self.shard_dim)
        repeats = [1 for _ in source.shape]
        repeats[self.batch_dim] = self.batch_size // source_batch_size
        source = source.repeat(repeats)
        source = self.manipulator.shard_along_on_cpu(source, dim=self.shard_dim)
        ops.parallel_write(target, source)