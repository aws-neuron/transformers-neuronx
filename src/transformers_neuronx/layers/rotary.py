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


def rotary_embedding(head_dim, cache_ids, base=10000):
    seq_len = cache_ids.shape[0]
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2) / head_dim))
    sinusoid_inp = torch.einsum("i , j -> i j", torch.arange(seq_len), inv_freq).float()
    sin = torch.sin(sinusoid_inp)
    cos = torch.cos(sinusoid_inp)
    pos_embd = torch.cat((sin, cos), dim=-1)
    return pos_embd