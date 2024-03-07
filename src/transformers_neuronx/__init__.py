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
from transformers_neuronx.version import __version__

from transformers_neuronx.config import NeuronConfig, QuantizationConfig, ContinuousBatchingConfig, GenerationConfig
from transformers_neuronx.constants import GQA
from transformers_neuronx.generation_utils import HuggingFaceGenerationModelAdapter

from transformers_neuronx.bloom.model import BloomForSampling
from transformers_neuronx.llama.model import LlamaForSampling
from transformers_neuronx.gpt2.model import GPT2ForSamplingWithContextBroadcasting
from transformers_neuronx.gptneox.model import GPTNeoXForSampling
from transformers_neuronx.gptj.model import GPTJForSampling
from transformers_neuronx.mistral.model import MistralForSampling
from transformers_neuronx.mixtral.model import MixtralForSampling
from transformers_neuronx.opt.model import OPTForSampling

from transformers_neuronx.modeling_auto import NeuronAutoModelForCausalLM

from . import testing
