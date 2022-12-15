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
import tarfile
import tempfile
from textwrap import dedent
import torch
from torch_neuronx.pyhlo import xla_data_pb2
from torch_neuronx.pyhlo.scribe import HloScribe
from torch_neuronx.pyhlo.constant.serialize_torch import serialize_torch
from torch_neuronx.proto import metaneff_pb2
from transformers_neuronx import ops
from transformers_neuronx import parallel


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


def compile_hlo_module(hlo_module):
    with tempfile.TemporaryDirectory() as tmpdir:
        flags = os.environ.get('NEURON_CC_FLAGS', '')
        flags = shlex.split(flags)
        dump_to = os.environ.get('NEURONX_DUMP_TO', None)
        if dump_to is not None:
            dump_to = os.path.join(dump_to, f'{hlo_module.name}.{GlobalCounter()()}')
            os.makedirs(dump_to, exist_ok=True)
            tmpdir = dump_to
        hlo_module_path = os.path.join(tmpdir, 'hlo_module.pb')
        hlo_module_path = os.path.realpath(hlo_module_path)
        dump_proto(hlo_module, hlo_module_path)
        neff_path = f'{hlo_module_path}.neff'
        neff_path = os.path.realpath(neff_path)
        command_line = ['neuronx-cc', 'compile', '--framework=XLA', '--target=trn1',
                        hlo_module_path, f'--output={neff_path}', *flags]
        if dump_to is not None:
            command_line.extend(['--verbose=INFO', '--pipeline', 'compile', 'SaveTemps'])
            command_line.append('--tensorizer-options=--dump-after-all=penguin')
        subprocess.check_call(command_line, cwd=tmpdir)
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
        for line in name_mapping.split('\n'):
            line = line.lstrip().strip()
            pname, dname, tname = line.split()
            primitive_type = getattr(xla_data_pb2.PrimitiveType, pname)
            metaneff_dtype = getattr(metaneff_pb2.MetaTensor.DataType, dname)
            torch_dtype = getattr(torch, tname)
            self.hlo2metaneff_mapping[primitive_type] = metaneff_dtype
            self.hlo2torch_mapping[primitive_type] = torch_dtype
            self.torch2name_mapping[torch_dtype] = pname.lower()

    def hlo2metaneff(self, primitive_type):
        return self.hlo2metaneff_mapping[primitive_type]

    def hlo2torch(self, primitive_type):
        return self.hlo2torch_mapping[primitive_type]

    def torch2name(self, torch_dtype):
        return self.torch2name_mapping[torch_dtype]


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

    def init(self):
        self.inputs.init()
        self.outputs.init()

    def setup(self, input_tensors, output_tensors):
        self.inputs.init()
        self.outputs.init()
        for idx, tensor in enumerate(input_tensors):
            self.inputs.add(idx, tensor)
        for idx, tensor in enumerate(output_tensors):
            self.outputs.add(idx, tensor)
        self.input_tensors = input_tensors
        self.output_tensors = output_tensors


class ParallelKernel:

    def __init__(self, hlo_module, tp_degree):
        self.hlo_module = hlo_module
        self.tp_degree = tp_degree
        self.neff_bytes = None
        self.model = None

    def build_memory(self):
        return ParallelMemory(self.hlo_module, self.tp_degree)

    def build(self):
        self.neff_bytes = compile_hlo_module(self.hlo_module)

    def load(self):
        self.model = torch.classes.neuron.ParallelModel(self.neff_bytes, self.tp_degree)
        self.model.load()

    def __call__(self, memory):
        return ops.parallel_run(self.model, memory.inputs, memory.outputs)


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
