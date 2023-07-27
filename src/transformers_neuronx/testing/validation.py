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
# =============================================================================
import os
from typing import Union, Tuple, List, Optional

import torch
from transformers import PretrainedConfig, PreTrainedModel
from transformers_neuronx.generation_utils import HuggingFaceGenerationModelAdapter


def validate(
    model: Union[torch.nn.Module, PreTrainedModel],
    neuron: torch.nn.Module,
    config: Union[dict, PretrainedConfig],
    inputs: Union[List, Tuple, torch.Tensor],
    sequence_length: int,
    compiler_args: Optional[Union[List, str]] = None,
    rtol: float = 1e-5,
    atol: float = 1e-5,
) -> None:
    """
    Compare the outputs of a Neuron model to a CPU model.

    Args:
        model: The CPU model golden reference.
        neuron: The Neuron model to be validated.
        config: The config for the model to be validated.
        inputs: The inputs to the model.
        sequence_length: The sequence length of the input + output tokens.
        compiler_args: The compiler arguments to be passed to the Neuron model.
        rtol: Relative tolerance when checking result equality.
        rtol: Absolute tolerance when checking result equality.

    Raises:
        AssertionError: Error when the values differ.
    """
    if compiler_args is None:
        os.environ["NEURON_CC_FLAGS"] = "--model-type=transformer --auto-cast=none"

    neuron.to_neuron()

    print("Configuration:")
    print(config)

    input_ids, attention_mask, start_ids = inputs
    batch_size, context_length = input_ids.shape
    new_tokens = sequence_length - context_length

    print("Input Ids:")
    print(input_ids)
    print("Attention Mask:")
    print(attention_mask)
    print("Start Ids:")
    print(start_ids)

    expected = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=new_tokens,
        min_new_tokens=new_tokens,
        top_k=1,
    )

    wrapper = HuggingFaceGenerationModelAdapter(config, neuron)
    actual_generate = wrapper.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=new_tokens,
        min_new_tokens=new_tokens,
        top_k=1,
    )
    neuron.reset()
    actual_sample = neuron.sample(
        input_ids, sequence_length=sequence_length, start_ids=start_ids, top_k=1
    )

    print("CPU Result:")
    print(expected)
    print("Neuron Result (Generate)")
    print(actual_generate)
    print("Neuron Result (Sample)")
    print(actual_sample)

    torch.testing.assert_close(
        actual=actual_generate, expected=expected, rtol=rtol, atol=atol
    )
    torch.testing.assert_close(
        actual=actual_sample, expected=expected, rtol=rtol, atol=atol
    )
