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
from transformers_neuronx.gpt_demo import demo
from transformers_neuronx.gptneox.model import GPTNeoXForSampling


def amp_callback(model, dtype):
    # cast attention and mlp to low precisions only; layernorms stay as f32
    for layer in model.gpt_neox.layers:
        layer.attention.to(dtype)
        layer.mlp.to(dtype)
    model.embed_out.to(dtype)


def main():
    demo('EleutherAI/gpt-neox-20b', GPTNeoXForSampling, amp_callback)
