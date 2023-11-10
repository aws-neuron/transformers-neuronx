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
import torch
import base64
import numpy as np
from transformers_neuronx import compiler
from neuronxcc.thor import decltensor, trace

def thor_call(func, *args, **kwargs):
    """ 
    This function applies Thor kernel function (func) to inputs (*args) in PyHLO.
    
    Args:
        func: Thor kernel function
        args: inputs of func
        kwargs:
            kernel_attrs (kernel attributes)
            grid (grid used in Thor kernel function)
            output_HloShapes (HloShapes of outputs of Thor kernel)


    Example:
        def mixed_pyhlo_thor(x, y):
            h = x.dtype[x.sizes].Multiply(x, x)
            o = thor_call(add_kernel, h, y, grid=32, output_HloShapes=x.dtype[x.sizes])
            return o        
    """
    
    kernel_attrs = kwargs.pop("kernel_attrs", ())
    grid = kwargs.pop("grid", None)   
    return ThorHloKernel(func, grid, kernel_attrs)(*args, **kwargs)


class ThorHloKernel:
    """
    This class lowers a user defined compiler kernel to PyHLO op.

    This is the FAL binding for the Thor API for compiler to program Neuron Device directly.

    Parameters:
        func: the function of the baremetal kernel defition
        grid: launch grid configuration
        kernel_attrs: List[str], string attribute to control code injection point in compiler

    There are 2 steps to use a thor kernel:
        1) Define Thor kernel
        2) Use Thor kernel within PyHLO by thor_call

    Example:
        # 1) Define Thor Kernel

        def add_subtract_kernel(a_ptr, b_ptr, c_ptr, d_ptr):
            ix = arange(128)[:, None]
            iy = arange(512)[None, :]
            tile_size = 128 * 512
            block_size = 8 * tile_size
            n_elements = 32 * 8 * 128 * 512
            j = program_id(axis=0)
            for i in affine_range(8):
                offset = j * block_size + i * tile_size + 512 * ix + iy
                mask = offset < n_elements
                a_ptr = a_ptr + offset
                b_ptr = b_ptr + offset
                c_ptr = c_ptr + offset
                d_ptr = d_ptr + offset
                a = load(a_ptr)
                b = load(b_ptr)
                c = a + b
                d = a - b
                store(c_ptr, value=c)
                store(d_ptr, value=d)   

        # 2) Use Thor kernel by thor_call:

            def mixed_pyhlo_thor(x, y):
                h = x.dtype[x.sizes].Multiply(x, x)
                o = thor_call(add_subtract_kernel, h, y, grid=32, output_HloShapes=[y.dtype[y.sizes], y.dtype[y.sizes]])
                return o

    """

    def __init__(self, func, grid=None, kernel_attrs=("tiled")):
        self.func = func
        self.func_literal = None
        self.grid = ()        
        self.kernel_attrs = kernel_attrs
        self.func_name = func.__name__
        if grid is not None:
           self.set_grid(grid)

    def dump_config(self):
        config = {}
        config["func_literal"] = self.func_literal
        config["kernel_attrs"] = self.kernel_attrs
        config["grid"] = self.grid
        config["func_name"] = self.func_name
        return base64.b64encode(json.dumps(config).encode("utf-8")).decode("utf-8")

    def set_grid(self, grid):
        if not isinstance(grid, (tuple, list)):
            grid = [grid]
        self.grid = grid
        return self

    @staticmethod
    def _to_neuron_dtype(hloShape):
        """
        Translate a pytorch dtype to neuron specific dtype representation in numpy
        """
        dtype_converter = compiler.DataTypeConverter()
        _dtype = dtype_converter.hlo2torch(hloShape.shape_proto.element_type)
        if _dtype == torch.bfloat16:            
            return np.dtype("|V2")        
        return torch.empty(1, dtype=_dtype).numpy().dtype

    def __call__(self, *args, output_HloShapes=None):        
        if output_HloShapes is None: 
           raise ValueError("output_shape should be set in ThorHloKernel !")
        if not isinstance(output_HloShapes, (list, tuple)):
            output_HloShapes = [output_HloShapes]
        get_shape = lambda hloShape: tuple([d for d in hloShape.shape_proto.dimensions])
        virtual_tensors = [
            decltensor(shape=get_shape(hloShape), dtype=self._to_neuron_dtype(hloShape))
            for hloShape in (*args, *output_HloShapes)
        ]
        
        traced = trace(
            func=self.func, grid=self.grid, kernel_attrs=self.kernel_attrs
        ).specialize(*virtual_tensors)

        self.func_literal = traced.serialize_ir_string(f"{self.func_name}_ir")
        config_str=self.dump_config()
        if len(output_HloShapes) > 1: 
            output_HloShapes = args[0].scribe.tuple(*output_HloShapes) 
        else:
            output_HloShapes, = output_HloShapes

        output = output_HloShapes.CustomCall(*args, 
                                             backend_config=str.encode(config_str), 
                                             custom_call_target='AwsNeuronCustomNativeKernel')

        return output

