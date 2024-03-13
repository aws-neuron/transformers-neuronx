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
import torch
from transformers_neuronx import compiler
from transformers_neuronx import parallel
from transformers_neuronx.compiler import DataTypeConverter
from typing import List, Tuple, Dict


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

    def __init__(self, num_hidden_layers, tp_degree, multi_layer_hlo_modules, head_hlo_module,
                 n_layers, buffers):
        if num_hidden_layers % n_layers:
            raise ValueError(f'n_layers={n_layers} does not divide num_hidden_layers={num_hidden_layers}')
        self.n_layers = n_layers
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

    def __init__(self, tp_degree, hlo_modules, buffers, debugger=None):
        self.kernels = [compiler.build_parallel_kernel(hm, tp_degree) for hm in hlo_modules]
        self.memories = [compiler.ParallelMemory(hm, tp_degree) for hm in hlo_modules]
        self.buffers = buffers
        self.debugger = debugger

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

class Debugger:
    counter = 0

    def __init__(self, debug=False):
        self.debug = debug
        self.debug_output_tensors = []
        self.debug_output_names = []

    def get_counter(self):
        self.counter += 1
        return self.counter

    def add_var(self, var, name=None):
        if not self.debug:
            return
        if name is None:
            name = f"debug_var_{self.get_counter()}"
        self.debug_output_tensors.append(var)
        self.debug_output_names.append(name)

    def get_tensors(self):
        return self.debug_output_tensors

    def get_names(self):
        return self.debug_output_names

def cache_slices_and_parameters(layers):
    cache_slices = []
    params = []
    for layer in layers:
        cache_slices.extend(layer.get_cache_slices())
        params.extend(layer.get_parameters())
    return cache_slices, params


def setup_memories(memories, buffers, cache_slices, params, output_buffer):
    # Since not all buffers have debug_outputs implemented, we need to check.
    # TODO: Remove hasattr when this is added to all models
    debug_outputs = buffers.debug_outputs if hasattr(buffers, "debug_outputs") else []
    for bucket_id, memory in enumerate(memories):
        input_buffers = buffers.get_input_buffers(bucket_id)
        inputs = input_buffers + [cache_slices[i][bucket_id] for i in range(len(cache_slices))] + params
        outputs = [output_buffer] + [cache_slices[i][bucket_id] for i in range(len(cache_slices))] + debug_outputs
        memory.setup(inputs, outputs, len(debug_outputs))


class HloMetadata:

    @staticmethod
    def inputs(hlo_module) -> List[Tuple[Tuple[int], torch.dtype]]:
        inputs = list()
        converter = DataTypeConverter()
        for parameter in hlo_module.host_program_shape.parameters:
            shape = tuple(dim for dim in parameter.dimensions)
            dtype = converter.hlo2torch(parameter.element_type)
            inputs.append((shape, dtype))
        return inputs

    @staticmethod
    def outputs(hlo_module) -> List[Tuple[Tuple[int], torch.dtype]]:
        outputs = list()
        converter = DataTypeConverter()

        parameters = hlo_module.host_program_shape.result.tuple_shapes
        if len(parameters) == 0: # Edge case: single output
            parameters = [hlo_module.host_program_shape.result]

        for parameter in parameters:
            shape = tuple(dim for dim in parameter.dimensions)
            dtype = converter.hlo2torch(parameter.element_type)
            outputs.append((shape, dtype))
        return outputs

    @staticmethod
    def aliases(hlo_module) -> Dict[int, int]:
        results = dict()
        for entry in hlo_module.input_output_alias.entries:
            if len(entry.output_shape_index) == 1:
                results[entry.output_shape_index[0]] = entry.parameter_number
            elif len(entry.output_shape_index) == 0: # Edge case: Single output
                results[0] = entry.parameter_number
            else:
                raise RuntimeError(f"Expected one-to-one alias but found one-to-many: {entry}")
        return results


class ParallelProgram:

    def __init__(self, hlo_module, num_inputs, num_outputs, neuron_config, tp_degree=1, tag=None):
        """
        A simple executable for a given HLOModuleProto
        """
        self.neuron_config = neuron_config
        self.kernel = compiler.ParallelKernel(
            hlo_module,
            self.neuron_config.get_local_tp(tp_degree),
            self.neuron_config.get_g_start_device_id(tp_degree),
            self.neuron_config.get_g_device_count(tp_degree),
            tag=tag
        )
        self.memory = None
        self.executor = None
        self.manipulator = parallel.ParallelTensorManipulator(
            tp_degree,
            rank_id=self.neuron_config.rank_id,
            local_tp_degree=self.neuron_config.get_local_tp(tp_degree)
        )
        self.tp_degree = tp_degree
        self.tag = tag
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.input_buffers = list()
        self.output_buffers = list()

    def setup(self, parameters):

        hlo = self.kernel.hlo_module

        # Set up inputs
        inputs_metadata = HloMetadata.inputs(hlo)
        assert (self.num_inputs + len(parameters)) == len(inputs_metadata), (
            f"The number of user inputs ({self.num_inputs}) plus input weights "
            f"({len(parameters)}) does not equal the expected number of "
            f"inputs {len(inputs_metadata)}"
        )
        inputs = [torch.zeros(shape, dtype=dtype) for shape, dtype in inputs_metadata[:self.num_inputs]]
        self.input_buffers = [self.manipulator.duplicate(input) for input in inputs]
        inputs = self.input_buffers + parameters

        aliases = HloMetadata.aliases(hlo)
        inverse = {value: key for key, value in aliases.items()}

        for index, ((shape, dtype), parameter) in enumerate(zip(inputs_metadata[self.num_inputs:], parameters)):
            assert parameter.dtype == dtype, (
                f"Input Type Error: Input type {parameter.dtype} at index "
                f"{index} does not match HloModuleProto type {dtype}"
            )
            # FIXME: This inverse check does not work
            # if index not in inverse: # Skip aliases since sizes may not match
            #     assert parameter.shape == shape, (
            #         f"Alias Type Error: Input shape {parameter.shape} at index "
            #         f"{index} does not match HloModuleProto shape {shape}"
            #     )

        # Set up outputs
        outputs_metadata = HloMetadata.outputs(hlo)
        assert self.num_outputs <= len(outputs_metadata), (
            f"The requested number of outputs ({self.num_outputs}) is larger "
            f"than the number of HLO outputs ({len(outputs_metadata)})"
        )

        outputs = list()
        for index, (shape, dtype) in enumerate(outputs_metadata):
            if index in aliases:
                output = inputs[aliases[index]]
                assert output.dtype == dtype, (
                    f"Alias Type Error: Input type {output.dtype} at index "
                    f"{aliases[index]} does not match output type {dtype} "
                    f"at index {index}"
                )
                # NOTE: We do not check the shape since when using bucketed
                #       caches, the input shape will be larger than the output
                #       shape
            else:
                output = torch.zeros(shape, dtype=dtype)
                output = self.manipulator.duplicate(output)
            outputs.append(output)
        self.output_buffers = outputs[:self.num_outputs]

        # Build the kernel
        self.kernel.build()
        self.kernel.load(io_ring_cache_size=1)
        self.memory = self.kernel.build_memory()
        self.memory.setup(
            inputs,
            outputs,
        )
        self.executor = self.kernel.build_executor(self.memory, self.input_buffers, self.output_buffers)
        # Warmup kernels to avoid unexpected initialization at runtime
        self.kernel.warmup()

    def build(self):
        self.kernel.build()

    def get_kernels(self):
        return [self.kernel]

    def execute(self, *inputs, return_ranks=-1):
        return self.executor(inputs, return_ranks)
