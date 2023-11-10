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
from neuronxcc.thor import program_id, affine_range
from neuronxcc.thor.ops import load, store, arange
from neuronxcc.thor.isa import reduce_free, activation

def softmax_inlined_single_tile(a_ptr, b_ptr):
    ix = arange(0, 128)[:, None]
    iy = arange(0, 512)[None, :]

    src = load(a_ptr[ix, iy])

    # -max(src)
    negate_src_max = reduce_free(np.max, tensor=src, axis=(0,), negate=True)
    
    # exp(src + negate_src_max)
    exp_ = activation(np.exp, tensor=src, bias=negate_src_max)

    exp_sum = reduce_free(np.add, tensor=exp_, axis=(0,))
    exp_sum = exp_sum.expand_dims(tuple(range(exp_sum.rank, exp_.rank)))

    dst = exp_ / exp_sum
    store(b_ptr[ix, iy], value=dst)