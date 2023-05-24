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
from torch import nn

def LayerNormCPU(torch_ln, hidden_dim_ratio):
    """ Creates a modified layernorm im pytorch 

    Args:
        torch_ln (nn.Module): original layernorm
        hidden_dim_ratio (float): ratio of target hidden dim to source one for correction

    Returns:
        nn.Module: modified layer norm
    """
    shape = list(torch_ln.normalized_shape)
    eps = torch_ln.eps
    ln = LayerNorm_(shape, eps=eps, hidden_dim_ratio=hidden_dim_ratio)
    ln.weight = torch_ln.weight
    ln.bias = torch_ln.bias
    return ln

class LayerNorm_(nn.Module):
    def __init__(self, normalized_shape, *,
                 eps=1e-5,
                 elementwise_affine=True,
                 hidden_dim_ratio=1.0):
        super().__init__()
    
        if isinstance(normalized_shape, int):
            normalized_shape = torch.Size([normalized_shape])
        elif isinstance(normalized_shape, list):
            normalized_shape = torch.Size(normalized_shape)
        assert isinstance(normalized_shape, torch.Size)

        self.hidden_dim_ratio = hidden_dim_ratio
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x: torch.Tensor):
        assert self.normalized_shape == x.shape[-len(self.normalized_shape):]
        dims = [-(i + 1) for i in range(len(self.normalized_shape))]
        mean = x.mean(dim=dims, keepdim=True)
        mean_x2 = (x * x).mean(dim=dims, keepdim=True)
        var = mean_x2 - mean * mean             
        mean *= self.hidden_dim_ratio
        var *= self.hidden_dim_ratio
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        if self.elementwise_affine:
            x_norm = self.weight * x_norm + self.bias
        return x_norm