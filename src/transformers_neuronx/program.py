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


class DecoderProgram:

    def setup(self, layers, ln_lm_head):
        raise NotImplementedError(DecoderProgram)

    def init_length(self, this_length, init_n_active_tokens):
        return this_length // init_n_active_tokens * init_n_active_tokens

    def init_step(self, init_n_active_tokens):
        return init_n_active_tokens

    def run(self, bucket_id):
        raise NotImplementedError(DecoderProgram)


class DoNothingDecoder(DecoderProgram):

    def setup(self, layers, ln_lm_head):
        pass

    def init_length(self, this_length, init_n_active_tokens):
        return 0

    def init_step(self, init_n_active_tokens):
        return 1

    def run(self, bucket_id):
        pass


class MultiLayerDecoder(DecoderProgram):

    def __init__(self, config, multi_layer_hlo_modules, head_hlo_module, n_layers, buffers):
        num_hidden_layers = config.num_hidden_layers
        if num_hidden_layers % n_layers:
            raise ValueError(f'n_layers={n_layers} does not divide num_hidden_layers={num_hidden_layers}')
        self.n_layers = n_layers
        tp_degree = config.tp_degree
        self.multi_layer_kernels = [compiler.build_parallel_kernel(hm, tp_degree)
                                    for hm in multi_layer_hlo_modules]
        self.head_kernel = compiler.build_parallel_kernel(head_hlo_module, tp_degree)
        self.multi_layers_memories = []
        for _ in range(num_hidden_layers // n_layers):
            memories = [compiler.ParallelMemory(hm, tp_degree) for hm in multi_layer_hlo_modules]
            self.multi_layers_memories.append(memories)
        self.head_memory = compiler.ParallelMemory(head_hlo_module, tp_degree)
        self.buffers = buffers

    def setup(self, layers, ln_lm_head):
        for kernel in self.multi_layer_kernels:
            kernel.load()
        self.head_kernel.load()
        for memories in self.multi_layers_memories:
            for memory in memories:
                memory.init()
        self.head_memory.init()
        buffers = self.buffers
        buffers.to_neuron()
        hidden_buffer, *_ = buffers.get_input_buffers(0)
        multi_layer_starts = range(0, len(layers), self.n_layers)
        multi_layers = [layers[start:start+self.n_layers] for start in multi_layer_starts]
        for memories, multi_layer in zip(self.multi_layers_memories, multi_layers):
            cache_slices, params = cache_slices_and_parameters(multi_layer)
            setup_memories(memories, buffers, cache_slices, params, hidden_buffer)
        head_inputs = [hidden_buffer, *ln_lm_head.get_parameters()]
        for index, input_buffer in enumerate(head_inputs):
            self.head_memory.inputs.add(index, input_buffer)
        self.head_memory.outputs.add(0, buffers.output_buffer)

    def run(self, bucket_id):
        for memories in self.multi_layers_memories:
            self.multi_layer_kernels[bucket_id](memories[bucket_id])
        self.head_kernel(self.head_memory)
        return self.buffers.output_buffer


class FullyUnrolledDecoder(DecoderProgram):

    def __init__(self, config, hlo_modules, buffers):
        self.kernels = [compiler.build_parallel_kernel(hm, config.tp_degree) for hm in hlo_modules]
        self.memories = [compiler.ParallelMemory(hm, config.tp_degree) for hm in hlo_modules]
        self.buffers = buffers

    def setup(self, layers, ln_lm_head):
        for kernel in self.kernels:
            kernel.load()
        for memory in self.memories:
            memory.init()
        cache_slices, params = cache_slices_and_parameters(layers)
        params.extend(ln_lm_head.get_parameters())
        buffers = self.buffers
        buffers.to_neuron()
        setup_memories(self.memories, buffers, cache_slices, params, buffers.output_buffer)

    def run(self, bucket_id):
        self.kernels[bucket_id](self.memories[bucket_id])
        return self.buffers.output_buffer


def cache_slices_and_parameters(layers):
    cache_slices = []
    params = []
    for layer in layers:
        cache_slices.extend(layer.get_cache_slices())
        params.extend(layer.get_parameters())
    return cache_slices, params


def setup_memories(memories, buffers, cache_slices, params, output_buffer):
    for bucket_id, memory in enumerate(memories):
        input_buffers = buffers.get_input_buffers(bucket_id)
        for index, input_buffer in enumerate(input_buffers):
            memory.inputs.add(index, input_buffer)
        cache_start_index = len(input_buffers)
        for index, bucketed_caches in enumerate(cache_slices, start=cache_start_index):
            memory.inputs.add(index, bucketed_caches[bucket_id])
        param_start_index = cache_start_index + len(cache_slices)
        for index, param in enumerate(params, start=param_start_index):
            memory.inputs.add(index, param)
        memory.outputs.add(0, output_buffer)
        for index, bucketed_caches in enumerate(cache_slices, start=1):
            memory.outputs.add(index, bucketed_caches[bucket_id])
