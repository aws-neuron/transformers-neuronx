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
from typing import List
import multiprocessing.pool

class TensorPool:
    """ A pool that helps manage the liveness of the torch tensors"""

    def __init__(self):
        self.tensor_pool: List[torch.Tensor] = list()
        self.thread_pool = multiprocessing.pool.ThreadPool(processes=1)

    def push(self, tensors):
        if isinstance(tensors, torch.Tensor):
            self.tensor_pool.append(tensors)
        elif isinstance(tensors, (List, tuple)):
            for t in tensors:
                self.tensor_pool.append(t)
        else:
            raise TypeError(f"Unsupported type {type(tensors)} to TensorPool")

    def clear(self):
        self.tensor_pool.clear()

    def async_clear(self):
        task = self.thread_pool.apply_async(self.clear)
        return task