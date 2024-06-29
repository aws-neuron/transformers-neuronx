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

from collections import namedtuple
import contextlib
from typing import Optional
from torch_neuronx.pyhlo.scribe import HloShape
import functools

debug_tensors = None

DebugTensor = namedtuple("DebugTensor", ["tensor", "unshard_dim", "metadata"])


def tap(name: str, tensor: HloShape, unshard_dim: Optional[int] = None):
    """
    Debug function "taps" a tensor in an HLO program for retrieval during
    program execution. This is useful for viewing intermediate tensor values.

    Args:
        name: the name of the tagged tensor that will be displayed.
        tensor: the intermidate tensor variable.

    Example Usage:
        In a model's hlo.py file, tag a tensor such as the attention output:
        ```
        output = attention.output(context, out_weight, out_scales, out_bias, tp_degree, self.neuron_config)
        from transformers_neuronx import global_debugger
        global_debugger.tap("output", output)
        ```
    """
    if debug_tensors is None:
        return tensor
    debug_tensors[name] = DebugTensor(tensor, unshard_dim, {})
    return tensor


@contextlib.contextmanager
def debug_context():
    """
    Context manager that retrieves debug_tensors during execution.

    Example Usage:
        During model execution, retrieve the debug tensors by wrapping the
        inference call:
        ```
        with global_debugger.debug_context():
            outputs = model_neuron.sample(inputs, sequence_length=128, top_k=1)
            debug_tensors = global_debugger.debug_tensors
            print(f'debug_tensors: {debug_tensors}')
        ```
    """
    global debug_tensors
    debug_tensors = {}
    try:
        yield
    finally:
        debug_tensors = None

# Decorates a scribe function so that instead of returning a tuple,
# it instead returns a list of outputs to which the debug tensors are
# added.
def populate_debug_tensors(debug_outputs={}):
    from transformers_neuronx import global_debugger as gdbg
    def inner(func):
        @functools.wraps(func)
        def wrapper(scribe):
            with gdbg.debug_context():
                # This func code should run, and inside of it users can tag their tensors
                # any tensors that are tagged will be added to the outputs
                outputs = func(scribe)
                for (tag_name, debug_tensor) in gdbg.debug_tensors.items():
                    debug_tensor.metadata['output_index'] = len(outputs)
                    debug_outputs[tag_name] = debug_tensor
                    outputs.append(debug_tensor.tensor)
            # Now we just have to actually return a tuple
            root_shapes = [shape.dtype[shape.sizes] for shape in outputs]
            return scribe.tuple(*root_shapes).Tuple(*outputs)
        return wrapper
    return inner