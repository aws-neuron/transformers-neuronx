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
import shlex
import subprocess
import hashlib
import tarfile
import tempfile
import numpy as np
from textwrap import dedent
import torch
import logging
from torch_neuronx.pyhlo import xla_data_pb2
from torch_neuronx.pyhlo.scribe import HloScribe
from torch_neuronx.pyhlo.constant.serialize_torch import serialize_torch
from torch_neuronx.proto import metaneff_pb2
from transformers_neuronx import ops
from transformers_neuronx import parallel
from neuronxcc import __version__ as compiler_version

def get_hash_module(hlo_module, flags):
    # Hashing is pretty fast and neglegible compared to compilation time
    hash_gen = hashlib.sha256()
    text = str(hlo_module) + flags.replace(" ", "")
    hash_gen.update(text.encode('utf-8'))
    hash = str(hash_gen.hexdigest())[:20]
    return hash

def compile_py_func(py_func):
    return HloScribe(serialize_torch)(py_func).module_proto


def build_kernel(py_func, tp_degree):
    hlo_module = compile_py_func(py_func)
    neff_bytes = compile_hlo_module(hlo_module)
    metaneff = hlo2metaneff(hlo_module)
    return Kernel(hlo_module, neff_bytes, metaneff, tp_degree)


def build_parallel_kernel(hlo_module, tp_degree):
    kernel = ParallelKernel(hlo_module, tp_degree)
    kernel.build()
    return kernel


def compile_hlo_module(hlo_module, tag=None):
    flags = os.environ.get('NEURON_CC_FLAGS', '')
    hash = get_hash_module(hlo_module, flags)
    # By default cache is on since hash of HLO is used to store neff and no collision can occur
    cache = os.environ.get('NEURONX_CACHE', 'off')
    # tag is used to make folder name more clear (e.g. add bucket-size to folder name)
    if tag is None:
        hlo_module_name = f'{hlo_module.name}.{compiler_version}.{hash}'
    else:
        hlo_module_name = f'{hlo_module.name}.{tag}.{compiler_version}.{hash}'

    dump = "NEURONX_DUMP_TO" in os.environ
    dump_to = os.environ.get('NEURONX_DUMP_TO', '/tmp')
    dump_to = os.path.join(dump_to, hlo_module_name)
    os.makedirs(dump_to, exist_ok=True)

    hlo_module_path = os.path.join(dump_to, f'{hlo_module_name}.pb')
    hlo_module_path = os.path.realpath(hlo_module_path)
    if not os.path.isfile(hlo_module_path) or cache != 'on':
        dump_proto(hlo_module, hlo_module_path)
    neff_path = f'{hlo_module_path}.neff'
    neff_path = os.path.realpath(neff_path)
    if not os.path.isfile(neff_path) or cache != 'on':
        command_line = ['neuronx-cc', 'compile', '--framework=XLA', '--target=trn1',
                        hlo_module_path, f'--output={neff_path}']
        if dump:
            command_line.extend(['--verbose=INFO', '--pipeline', 'compile', 'SaveTemps'])
        else:
            command_line.extend(['--verbose=35'])
        flags = shlex.split(flags)
        command_line.extend(flags)
        subprocess.check_call(command_line, cwd=dump_to)
    with open(neff_path, 'rb') as f:
        neff_bytes = f.read()
    return neff_bytes


class GlobalCounter:

    _counter = 0

    def __call__(self):
        GlobalCounter._counter += 1
        return GlobalCounter._counter


def dump_proto(proto, path):
    with open(path, 'wb') as f:
        f.write(proto.SerializeToString())


def hlo2metaneff(hlo_module):
    prog_shape = hlo_module.host_program_shape
    dtype_converter = DataTypeConverter()

    def fill_with(target, names, shapes):
        for name, shape in zip(names, shapes):
            tensor = target.add()
            tensor.name = name.encode()
            tensor.shape[:] = shape.dimensions
            tensor.data_type = dtype_converter.hlo2metaneff(shape.element_type)

    input_names = find_input_names(hlo_module)
    metaneff = metaneff_pb2.MetaNeff()
    fill_with(metaneff.input_tensors, input_names, prog_shape.parameters)
    output_names = find_output_names(hlo_module)
    if prog_shape.result.element_type == xla_data_pb2.PrimitiveType.TUPLE:
        output_shapes = prog_shape.result.tuple_shapes
    else:
        output_shapes = [prog_shape.result]
    fill_with(metaneff.output_tensors, output_names, output_shapes)
    for entry in hlo_module.input_output_alias.entries:
        assert len(entry.parameter_shape_index) == 0
        metaneff.output_aliases_to[entry.output_shape_index[0]] = entry.parameter_number
    return metaneff


def find_input_names(hlo_module):
    # TODO: read names from hlo_module
    prog_shape = hlo_module.host_program_shape
    return [f'input{idx}' for idx in range(len(prog_shape.parameters))]


def find_output_names(hlo_module):
    # TODO: read names from hlo_module
    prog_shape = hlo_module.host_program_shape
    if prog_shape.result.element_type != xla_data_pb2.PrimitiveType.TUPLE:
        return ['output0']
    return [f'output{idx}' for idx in range(len(prog_shape.result.tuple_shapes))]


class DataTypeConverter:

    def __init__(self):
        name_mapping = '''
            PRED    UINT8       bool
            S8      INT8        int8
            S16     INT16       int16
            S32     INT32       int32
            S64     INT64       int64
            U8      UINT8       uint8
            U16     UINT16      int16
            U32     INT32       int32
            U64     INT64       int64
            F16     FLOAT16     float16
            F32     FLOAT       float32
            F64     DOUBLE      float64
            BF16    BFLOAT16    bfloat16
        '''
        name_mapping = dedent(name_mapping)
        name_mapping = name_mapping.lstrip().strip()
        self.hlo2metaneff_mapping = {}
        self.hlo2torch_mapping = {}
        self.torch2name_mapping = {}
        self.torch2hlo_mapping = {}
        for line in name_mapping.split('\n'):
            line = line.lstrip().strip()
            pname, dname, tname = line.split()
            primitive_type = getattr(xla_data_pb2.PrimitiveType, pname)
            metaneff_dtype = getattr(metaneff_pb2.MetaTensor.DataType, dname)
            torch_dtype = getattr(torch, tname)
            self.hlo2metaneff_mapping[primitive_type] = metaneff_dtype
            self.hlo2torch_mapping[primitive_type] = torch_dtype
            self.torch2name_mapping[torch_dtype] = pname.lower()
            self.torch2hlo_mapping[torch_dtype] = primitive_type

    def hlo2metaneff(self, primitive_type):
        return self.hlo2metaneff_mapping[primitive_type]

    def hlo2torch(self, primitive_type):
        return self.hlo2torch_mapping[primitive_type]

    def torch2name(self, torch_dtype):
        return self.torch2name_mapping[torch_dtype]

    def torch2hlo(self, torch_dtype):
        return self.torch2hlo_mapping[torch_dtype]


class Kernel:

    def __init__(self, hlo_module, neff_bytes, metaneff, tp_degree):
        self.hlo_module = hlo_module
        self.neff_bytes = neff_bytes
        metaneff_bytes = metaneff.SerializeToString()
        model_cls = torch.classes.neuron.Model
        self.models = [model_cls(neff_bytes, metaneff_bytes) for _ in range(tp_degree)]
        self.executor = parallel.Executor(tp_degree)

    def load(self):
        ops.init()
        parallel.parallel_load(self.models)

    def __call__(self, inputs):
        return self.executor.execute(self.models, *inputs)

    def profile_start(self, profile_dir):
        for model, ntff_path in zip(self.models, self._ntff_paths(profile_dir)):
            ops.profile_start(model, ntff_path)

    def profile_stop(self, profile_dir):
        ntff_paths = self._ntff_paths(profile_dir)
        for model, ntff_path in zip(self.models, ntff_paths):
            ops.profile_stop(ntff_path)
        ntff_tar_path = os.path.join(profile_dir, f'{self.hlo_module.name}.ntff.tar')
        with tarfile.open(ntff_tar_path, 'w|') as fp:
            for idx, ntff_path in enumerate(ntff_paths):
                fp.add(ntff_path, f'profile_rank_{idx}.ntff')

    def _ntff_paths(self, profile_dir):
        paths = []
        for idx in range(len(self.models)):
            filename = f'{self.hlo_module.name}.{idx:03d}.ntff'
            paths.append(os.path.join(profile_dir, filename))
        return paths


class ParallelMemory:

    def __init__(self, hlo_module, tp_degree):
        input_names = find_input_names(hlo_module)
        output_names = find_output_names(hlo_module)
        self.inputs = torch.classes.neuron.ParallelTensorSet(input_names, tp_degree)
        self.outputs = torch.classes.neuron.ParallelTensorSet(output_names, tp_degree)
        self.input_tensors = None
        self.output_tensors = None
        self.n_debug_tensors = 0 # How many of output tensors are for debugging

    def init(self):
        self.inputs.init()
        self.outputs.init()

    def setup(self, input_tensors, output_tensors, n_debug_tensors=0):
        self.n_debug_tensors = n_debug_tensors
        self.inputs.init()
        self.outputs.init()
        for idx, tensor in enumerate(input_tensors):
            self.inputs.add(idx, tensor)
        for idx, tensor in enumerate(output_tensors):
            self.outputs.add(idx, tensor)
        self.input_tensors = input_tensors
        self.output_tensors = output_tensors

    def get_debug_tensors(self):
        if self.n_debug_tensors > 0:
            return self.output_tensors[-self.n_debug_tensors:]
        else:
            return []


class Executor:

    def __init__(self, model, memory, inputs, outputs):
        """
        An optimized executor class that allocates a thread for each model rank.

        This implements a number of optimizations to speed up execution time:
        - The primary optimization is that this class creates set of fixed
          threads that are always blocking on a barrier waiting to execute.
          In practice, this can significantly speed up the thread startup
          time for model executions compared to other threadpool
          implementations.
        - A second optimization is that each thread is responsible for
          both the I/O and the execution of the model rank. This means that
          fewer `libtorchneuron` API calls are made an fewer threadpool starts
          are being triggered.
        The combination of the above optimization can have a huge effect
        especially on small/fast models.
        """
        self.inputs = inputs
        self.outputs = outputs
        self.executor = torch.classes.neuron.ParallelExecutor(
            model,
            memory.inputs,  # All inputs (inputs, caches, weights)
            memory.outputs, # All outputs (outputs, caches)
            inputs,         # User provided inputs
            outputs,        # Returned outputs
        )

    def __call__(self, inputs, return_ranks: int = -1):
        """
        Execute the kernel with the given inputs.

        Arguments:
            inputs: A set of input tensors to copy to each model rank.
            return_ranks: Specifies which ranks to return to python. This is
                useful if data is only required from a specific number of
                NeuronCores. For example:
                    * 0: Do not return any data.
                    * 1: Return data only from the first rank. This is useful
                         when the data is synchronized across ranks.
                    * -1: Return data form all ranks.

        Returns:
            result: The output tensors from each rank concatenated along dim 0.
        """
        casted = []
        for cpu, buf in zip(inputs, self.inputs):
            if cpu.dtype != buf.dtype:
                cpu = cpu.to(buf.dtype)
            casted.append(cpu)
        outputs = torch.ops.neuron._parallel_executor_run(self.executor, casted, return_ranks)
        if return_ranks == 1:
            result = tuple(shards[0] for shards in outputs)
        else:
            result = tuple(torch.cat(shards, dim=0) for shards in outputs)
        if len(result) == 1:
            return result[0]
        return result


class ParallelKernel:
    hlo_snapshot_iter = 0
    def __init__(self, hlo_module, tp_degree, g_start_device_id=0, g_device_count=None):
        self.hlo_module = hlo_module
        self.tp_degree = tp_degree
        self.neff_bytes = None
        self.model = None
        self.hlo_snapshot = None
        self.generate_hlo_snapshot()
        self.g_start_device_id = g_start_device_id
        if g_device_count is None:
            g_device_count = tp_degree
        self.g_device_count = g_device_count

    def generate_hlo_snapshot(self, tensors=None):
        if tensors is None:
            self.hlo_snapshot_folder = os.environ.get("HLO_SNAPSHOT_PATH", None)
            self.hlo_snapshot = self.hlo_snapshot_folder is not None
            if self.hlo_snapshot:
                os.makedirs(f"{self.hlo_snapshot_folder}", exist_ok=True)
        elif self.hlo_snapshot:
            folder = os.path.join(self.hlo_snapshot_folder, f"iter{ParallelKernel.hlo_snapshot_iter}")
            os.makedirs(folder, exist_ok=True)
            for i, tensor in enumerate(tensors):
                filename = os.path.join(folder, f"{i}.npy")
                tensor_cpu = ops.parallel_cpu(tensor)
                if isinstance(tensor_cpu, list):
                    tensor_cpu = tensor_cpu[0]
                if tensor_cpu.dtype == torch.bfloat16:
                    tensor_cpu = tensor_cpu.view(torch.int16)
                    tensor_cpu = tensor_cpu.numpy()
                    tensor_cpu = tensor_cpu.view('|V2')
                else:
                    tensor_cpu = tensor_cpu.detach().numpy()
                np.save(filename, tensor_cpu)
            ParallelKernel.hlo_snapshot_iter += 1


    def build_memory(self):
        return ParallelMemory(self.hlo_module, self.tp_degree)

    def build(self, tag=None):
        # Avoid rebuilding NEFF. This path occurs during deserialization
        if self.neff_bytes is not None:
            return
        self.neff_bytes = compile_hlo_module(self.hlo_module, tag)

    def load(self):
        self.model = torch.classes.neuron.ParallelModel(self.neff_bytes, self.tp_degree, self.g_start_device_id, self.g_device_count)
        self.model.load()

    def __call__(self, memory):
        if self.hlo_snapshot:
            self.generate_hlo_snapshot(memory.input_tensors)
        return ops.parallel_run(self.model, memory.inputs, memory.outputs)

    def build_executor(self, memory, inputs, outputs):
        return Executor(self.model, memory, inputs, outputs)


def gen_zero_input(hlo_module, index):
    shape_proto = hlo_module.host_program_shape.parameters[index]
    shape = [dim for dim in shape_proto.dimensions]
    dtype = DataTypeConverter().hlo2torch(shape_proto.element_type)
    return torch.zeros(shape, dtype=dtype)


def gen_zero_output(hlo_module, index=None):
    shape_proto = hlo_module.host_program_shape.result
    if index is not None:
        shape_proto = shape_proto.tuple_shapes[index]
    shape = [dim for dim in shape_proto.dimensions]
    dtype = DataTypeConverter().hlo2torch(shape_proto.element_type)
    return torch.zeros(shape, dtype=dtype)


def gen_zero_inputs(hlo_module):
    return gen_randn_inputs(hlo_module, std=0)


def gen_randn_inputs(hlo_module, std=0.01, int_func=torch.zeros, treat_as_int=None):
    if treat_as_int is None:
        treat_as_int = []
    dtype_converter = DataTypeConverter()
    inputs = []
    for idx, param in enumerate(hlo_module.host_program_shape.parameters):
        shape = list(param.dimensions)
        dtype = dtype_converter.hlo2torch(param.element_type)
        if std and dtype.is_floating_point and idx not in treat_as_int:
            tensor = std * torch.randn(shape, dtype=dtype)
        else:
            tensor = int_func(shape, dtype=dtype)
        inputs.append(tensor)
    return inputs

def gen_zero_output_from_shape(input):
    shape_proto = input.shape_proto
    shape = tuple(shape_proto.dimensions)
    dtype = DataTypeConverter().hlo2torch(shape_proto.element_type)
    return torch.zeros(shape, dtype=dtype)

def get_debug_outputs(program, bucket_id=0):
    debug_tensors = program.memories[bucket_id].get_debug_tensors()
    debug_tensors = [ops.parallel_cpu(x) for x in debug_tensors]
    debug_names = program.debugger.get_names() if hasattr(program, "debugger") else []
    return debug_tensors, debug_names

class HLOKernel:
    def __init__(self, hlo_program, tp, start_g_nc_id=0, g_nc_count=None):
        self.hlo_program = hlo_program
        self.tp = tp
        self.start_g_nc_id = start_g_nc_id
        if g_nc_count is None:
            g_nc_count = tp
        self.g_nc_count = g_nc_count
        self.manipulator = parallel.ParallelTensorManipulator(tp_degree=self.tp)

    def compile(self):
        self.hlo_module = compile_py_func(self.hlo_program)
        self.build()

    def build(self):
        # wrap HLO with kernel and compile{
        logging.debug(f"Build hlo module with tp {self.tp} g_start_device_id {self.start_g_nc_id} g_device_count {self.g_nc_count}")
        self.kernel = ParallelKernel(self.hlo_module, tp_degree=self.tp, g_start_device_id=self.start_g_nc_id, g_device_count=self.g_nc_count)
        # load NEFF
        self.kernel.build()

    def load(self):
        logging.debug(f"loading {self.hlo_module.name}")
        self.kernel.load()

    def setup(self, nc_input_buffers, nc_output_buffers, output_count=None):
        self.memories = self.kernel.build_memory()
        if len(nc_output_buffers) == 0:
            if output_count is None:
                cpu_output_buffers = [gen_zero_output(self.hlo_module, None)] # index is only needed when indexing output tuple
            else:
                cpu_output_buffers = [gen_zero_output(self.hlo_module, i) for i in range(output_count)]
            nc_output_buffers = []
            for o in cpu_output_buffers:
                nc_output_buffers.append(self.manipulator.duplicate(o)) 
        self.memories.setup(nc_input_buffers, nc_output_buffers) # Segmentation fault (core dumped)

    def run(self):
        logging.debug(f"running {self.hlo_module.name}")
        self.kernel(self.memories)
