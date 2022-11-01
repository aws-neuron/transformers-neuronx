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
import argparse
import torch
from pyhlo import hlo_pb2, xla_data_pb2
from transformers_neuronx import compiler


def main_randn():
    parser = argparse.ArgumentParser()
    parser.add_argument('module')
    parser.add_argument('snapshot')
    parser.add_argument('--std', type=float, default=0.01)
    parser.add_argument('--int', default='zeros')
    parser.add_argument('--treat_as_int', nargs='*', type=int, default=None)
    args = parser.parse_args()
    hlo_module = hlo_pb2.HloModuleProto()
    with open(args.module, 'rb') as fp:
        hlo_module.ParseFromString(fp.read())
    torch.manual_seed(15213)
    int_func = getattr(torch, args.int)
    hlo_snapshot = gen_randn_hlo_snapshot(hlo_module, args.std, int_func, args.treat_as_int)
    with open(args.snapshot, 'wb') as fp:
        fp.write(hlo_snapshot.SerializeToString())


def gen_randn_hlo_snapshot(hlo_module, std, int_func, treat_as_int):
    randn_inputs = compiler.gen_randn_inputs(hlo_module, std, int_func, treat_as_int)
    hlo_snapshot = hlo_pb2.HloSnapshot()
    hlo_snapshot.hlo.hlo_module.CopyFrom(hlo_module)
    for tensor, param in zip(randn_inputs, hlo_module.host_program_shape.parameters):
        argument = hlo_snapshot.arguments.add()
        argument.shape.CopyFrom(param)
        name = xla_data_pb2.PrimitiveType.Name(param.element_type)
        attrname = f'{name.lower()}s'
        attr = getattr(argument, attrname)
        array = tensor.numpy().ravel()
        if isinstance(attr, bytes):
            setattr(argument, attrname, array.tobytes())
        else:
            attr.extend(array.tolist())
    return hlo_snapshot
