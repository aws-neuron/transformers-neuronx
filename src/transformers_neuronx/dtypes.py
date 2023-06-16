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


def to_torch_dtype(dtype):
    mapping = {
        'f32': torch.float32,
        'f16': torch.float16,
        'bf16': torch.bfloat16,
        's8': torch.int8,
    }
    return mapping[dtype]


def to_amp(dtype):
    mapping = {
        torch.float32: 'f32',
        torch.float16: 'f16',
        torch.bfloat16: 'bf16',
    }
    return mapping[dtype]
