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
import os
import time
from unittest import TestCase
from transformers.configuration_utils import PretrainedConfig
from transformers_neuronx import parallel
from transformers_neuronx.compiler import gen_zero_inputs
from transformers_neuronx.opt.config import OPTConfig
from transformers_neuronx.opt.hlo import build_opt_block_kernel


class OPTCompilerTest(TestCase):

    def test_opt_kernel(self):
        config = opt_125m_config(batch_size=4)
        warmup_steps = 2
        timing_steps = 10
        profile_dir = os.environ.get('NEURON_PROFILE', None)
        kernel = build_opt_block_kernel(config, n_active_tokens=1, n_positions=config.n_positions)
        kernel.load()
        manipulator = parallel.TensorManipulator(config.tp_degree)
        zero_inputs = gen_zero_inputs(kernel.hlo_module)
        zero_inputs = [manipulator.duplicate(tensor) for tensor in zero_inputs]
        for _ in range(warmup_steps):
            zero_outputs = kernel(zero_inputs)
        elapsed_list = []
        for _ in range(timing_steps):
            start = time.time()
            zero_outputs = kernel(zero_inputs)
            elapsed_ms = (time.time() - start) * 1000
            elapsed_list.append(elapsed_ms)
        avg_elapsed = sum(elapsed_list) / len(elapsed_list)
        print(f'avg_elapsed {avg_elapsed:.2f} ms')
        if profile_dir is not None:
            os.makedirs(profile_dir, exist_ok=True)
            neff_path = os.path.join(profile_dir, f'{kernel.hlo_module.name}.neff')
            with open(neff_path, 'wb') as f:
                f.write(kernel.neff_bytes)
            print(f'NEFF written to {neff_path}')
            kernel.profile_start(profile_dir)
            zero_outputs = kernel(zero_inputs)
            kernel.profile_stop(profile_dir)
            print(f'NTFF dumped to {profile_dir}/')
        zero_outputs = [parallel.cpu(tensor) for tensor in zero_outputs]
        for tensor_cores in zero_outputs:
            for tensor in tensor_cores:
                self.assertEqual((tensor * tensor).sum(), 0.0)


def opt_125m_config(batch_size):
    config = PretrainedConfig(
        activation_function='relu',
        do_layer_norm_before=True,
        eos_token_id=2,
        ffn_dim=3072,
        hidden_size=768,
        max_position_embeddings=2048,
        num_attention_heads=12,
        num_hidden_layers=12,
        pad_token_id=1,
        torch_dtype='float16',
        vocab_size=50272,
        word_embed_proj_dim=768,
    )
    return OPTConfig(config, n_positions=2048, batch_size=batch_size, amp='f16', tp_degree=2)


def opt_13b_config(batch_size):
    config = PretrainedConfig(
        activation_function='relu',
        do_layer_norm_before=True,
        eos_token_id=2,
        ffn_dim=20480,
        hidden_size=5120,
        max_position_embeddings=2048,
        num_attention_heads=40,
        num_hidden_layers=40,
        pad_token_id=1,
        torch_dtype='float16',
        vocab_size=50272,
        word_embed_proj_dim=5120,
    )
    return OPTConfig(config, n_positions=2048, batch_size=batch_size, amp='f16', tp_degree=2)
