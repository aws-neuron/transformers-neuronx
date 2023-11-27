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

import numpy as np
from neuronxcc.nki import program_id, affine_range
from neuronxcc.nki.ops import load, store, arange
from neuronxcc.nki.isa import reduce_free, activation

def add_subtract_kernel(a_ptr, b_ptr, c_ptr, d_ptr):
    """
    This NKI kernel generates both a+b and a-b
    """
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
