# Transformers Neuron (``transformers-neuronx``) Developer Guide

Transformers Neuron for Trn1 and Inf2 is a software package that enables
PyTorch users to perform large language model (LLM) inference on
second-generation Neuron hardware (See: [NeuronCore-v2](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/arch/neuron-hardware/neuron-core-v2.html)).


# Introduction

The [Transformers Neuron repository](https://github.com/aws-neuron/transformers-neuronx)
contains the source code of the AWS Neuron Transformers integration project.
As it stands now, it mainly serves the purpose of
running transformer decoder inference (autoregressive sampling)
workflows on the Neuron platform.

Note: This project is **actively** in development. The Neuron team is
still heavily modifying the Neuron optimized module classes. The
functionality provided in this repository will not maintain long-term
API stability until version >= 1.0.0. For applications willing to reuse
code from this repository, we recommend treating the Neuron optimized
module implementations as samples, and pin the version of the main
library package ``torch-neuronx`` to avoid breaking interface changes as
new features are developed.

# Installation

## Stable Release

To install the most rigorously tested stable release, use the PyPI pip wheel:

```
pip install transformers-neuronx --extra-index-url=https://pip.repos.neuron.amazonaws.com
```

## Development Version


To install the development version with the latest features and improvements, use ``git`` to install from the
[Transformers Neuron repository](https://github.com/aws-neuron/transformers-neuronx):

```
pip install git+https://github.com/aws-neuron/transformers-neuronx.git
```
<details>
<summary>Installation Alternatives</summary>
<br>

Without `git`, save the package contents locally and use:
```
pip install transformers-neuronx/ # This directory contains `setup.py`
```

Similarly, a standalone wheel can be created using the `wheel` package
with the local repository contents:
```
pip install wheel
cd transformers-neuronx/  # This directory contains `setup.py`
python setup.py bdist_wheel
pip install dist/transformers_neuronx*.whl
```
This generates an installable `.whl` package under the `dist/` folder.
</details>

# Checkpoint compatibility with HuggingFace Transformers

``transformers-neuronx`` is checkpoint-compatible with HuggingFace
Transformers. While the Neuron team reimplemented some HuggingFace
Transformers models from scratch for the purpose of maximizing the
execution efficiency of transformer decoders on Neuron, the
implementations are done with maximizing compatibility in mind, meaning
one can train transformer decoder models, say GPT2, using the standard
HuggingFace Transformers library, and then construct an
inference-optimized decoder model using transformers-neuronx's
``GPT2ForSampling`` class. If training was done with other libraries
such as MegatronLM, then it is still possible to convert the obtained
checkpoint to the standard HuggingFace Transformers checkpoint format,
and then move on to transformers-neuronx's optimized decoder
implementations.

# Neuron optimized transformer decoders implemented in XLA High Level Operations (HLO)

Due to the stateful nature of the autoregressive sampling computation, an efficient implementation
of autoregressive sampling on the Neuron platform requires rewriting the model forward function into
a pure-function computation running on fixed-shape tensors. Furthermore, we want the pure-function
computation be implemented in a compiled language so that the Neuron compiler can perform extensive
code analysis and optimization. We chose XLA High Level Operations (HLO) as the compiled language
for implementing Neuron optimized transformer decoder classes. The source code of these classes
contains Python functions written in a syntax called "PyHLO", name of a Neuron internal tool for
writing/compiling the HLO language in Python. As a example, a "language model head" implemented in
PyHLO may look like the following.

```
class LmHeadHlo:

    ...

    def lm_head(self, scribe):
        dtype = self.dtype
        hidden_size = self.hidden_size
        n_active_tokens = self.n_active_tokens
        batch_size = self.batch_size
        vocab_size = self.vocab_size
        hidden = dtype[hidden_size, n_active_tokens, batch_size].Parameter(parameter_number=0)
        weight = dtype[hidden_size, vocab_size].Parameter(parameter_number=1)
        rhs_size = n_active_tokens * batch_size
        hidden = dtype[hidden_size, rhs_size].Reshape(hidden)
        dot_dims = dict(lhs_contracting_dimensions=[0], rhs_contracting_dimensions=[0])
        logits = dtype[vocab_size, rhs_size].Dot(weight, hidden, dot_dimension_numbers=dot_dims)
        return dtype[vocab_size, n_active_tokens, batch_size].Reshape(logits)

    ...
```

The `transformers_neuronx.compiler.compile_py_func` function can convert the Python `lm_head`
function into `HloModuleProto`, a valid input format for the `neuronx-cc` compiler.

# Tensor-parallelism support

For transformer decoders used in large language models,
tensor-parallelism is neccessary as it provides a way to shard the
models' large weight matrices onto multiple NeuronCores, and having
NeuronCores working on the same matrix multiply operation
collaboratively. transformers-neuronx's tensor-parallelism support makes
heavy use of collective operations such as all-reduce, which is
supported natively by the Neuron runtime.

There are some principles for setting tensor-parallelism degree (number
of NeuronCores participating in sharded matrix multiply operations) for
Neuron-optimized transformer decoder models.

1. The number of attention heads needs to be divisible by the
   tensor-parallelism degree. (Note: this limitation only applies to NeoX/GPTJ, it will be removed in future release.)
2. The total data size of model weights and key-value caches needs to be
   smaller than 16 GB times the tensor-parallelism degree.
3. Currently, the Neuron runtime supports tensor-parallelism degrees 1,
   2, 8, 16, and 32 on Trn1/Trn1n and supports tensor-parallelism degrees 1, 2, 4,
   8, and 24 on Inf2.

Some examples:

1. ``facebook/opt-13b`` has 40 attention heads, and when running at
   batch size 1 and float16 precision the model requires ~29 GB memory,
   therefore a ``trn1.2xlarge`` with 32 GB device memory is sufficient.
2. ``facebook/opt-30b`` has 56 attention heads, and at batch size 1 and
   float16 precision the model requires ~66 GB memory, therefore it can
   run on 8 NeuronCores on one ``trn1.32xlarge`` using 128 GB device
   memory.
3. ``gpt2-xl`` has 25 attention heads and requires ~4 GB memory at
   bfloat16 precision. It runs without tensor-parallelism only.


# Features

## Hugging Face generate() API support

Transformers Neuron models support the Hugging Face [generate()](https://huggingface.co/docs/transformers/v4.28.1/en/main_classes/text_generation#transformers.GenerationMixin.generate)
API via the ``HuggingFaceGenerationModelAdapter`` adapter class. In the following example we
demonstrate how to run sampling with temperature using the ``GPT2`` model:

```
import os
from transformers_neuronx.gpt2.model import GPT2ForSampling
from transformers_neuronx.generation_utils import HuggingFaceGenerationModelAdapter
from transformers_neuronx.module import save_pretrained_split
from transformers import AutoModelForCausalLM, AutoTokenizer
os.environ['NEURON_CC_FLAGS'] = '--model-type=transformer-inference'

# Load and save the CPU model
model_cpu = AutoModelForCausalLM.from_pretrained('gpt2')
save_pretrained_split(model_cpu, 'gpt2-split')

# Create and compile the Neuron model
model_neuron = GPT2ForSampling.from_pretrained('gpt2-split', batch_size=1, tp_degree=2, n_positions=256, amp='f32', unroll=None)
model_neuron.to_neuron()

# Use the `HuggingFaceGenerationModelAdapter` to access the generate API
model = HuggingFaceGenerationModelAdapter(model_cpu.config, model_neuron)

# Get a tokenizer and exaple input
tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = 'left'
text = "Hello, I'm a language model,"
encoded_input = tokenizer(text, return_tensors='pt', padding=True)

# Run inference using temperature
model.reset_generation()
sample_output = model.generate(
    input_ids=encoded_input.input_ids,
    attention_mask=encoded_input.attention_mask,
    do_sample=True,
    max_length=256,
    temperature=0.7,
)
print([tokenizer.decode(tok) for tok in sample_output])
```

## int8 weight storage support

Transformers Neuron supports int8 weight storage for the `GPT2` model class.
int8 weight storage can be used to reduce memory bandwidth usage to improve
model performace. int8 weight storage support for additional model classes
will be added in an uncoming relesae. In the following example we demonstrate
how to apply int8 weight storage to the `GPT2` model via the
`QuantizationConfig` and `NeuronConfig` configs:

```
import os
import torch
from transformers_neuronx.gpt2.model import GPT2ForSampling
from transformers_neuronx.module import save_pretrained_split
from transformers_neuronx.config import NeuronConfig, QuantizationConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
os.environ['NEURON_CC_FLAGS'] = '--model-type=transformer-inference'

# Cast attention and mlp layers to low precisions only; layernorms stay as f32
def amp_callback(model, dtype):
    for block in model.transformer.h:
        block.attn.to(dtype)
        block.mlp.to(dtype)
    model.lm_head.to(dtype)

# Load and save the CPU model with bfloat16 casting
model_cpu = AutoModelForCausalLM.from_pretrained('gpt2')
amp_callback(model_cpu, torch.bfloat16)
save_pretrained_split(model_cpu, 'gpt2-split')

# Set the weight storage config use int8 quantization and bf16 dequantization
neuron_config = NeuronConfig(
    quant=QuantizationConfig(quant_dtype='s8', dequant_dtype='bf16'),
)

# Create and compile the Neuron model
model_neuron = GPT2ForSampling.from_pretrained('gpt2-split', batch_size=1, tp_degree=2, n_positions=256, amp='bf16', neuron_config=neuron_config)
model_neuron.to_neuron()

# Get a tokenizer and exaple input
tokenizer = AutoTokenizer.from_pretrained('gpt2')
text = "Hello, I'm a language model,"
encoded_input = tokenizer(text, return_tensors='pt')

# Run inference
with torch.inference_mode():
    generated_sequence = model_neuron.sample(encoded_input.input_ids, sequence_length=256, start_ids=None)
    print([tokenizer.decode(tok) for tok in generated_sequence])

```

## Parallel Input Prompt Context Encoding

Transformers Neuron supports parallel input prompt context encoding for the `GPT2`
model class. Parallel context encoding can be used to significantly reduce
the latency of the input prompt context encoding before the autoregressive
decoder token generation loop. Parallel context encoding support for additional
model classes will be added in an uncoming release.

The `GPT2ForSamplingWithContextBroadcasting` class has a `context_length_estimate`
variable that determines the number of input prompt tokens that will be processed in
parallel. For optimal results, this should be set to a power of 2 that is
closest to the most frequently seen input prompt length.
In the following example we demonstrate how to apply parallel context encoding
to the `GPT2` model via the `GPT2ForSamplingWithContextBroadcasting` class.
In this example, we set the `context_length_estimate` to be 128, which is
the closest power of 2 the length of the input prompt (97 tokens).

```
import os
import math
import torch
from transformers_neuronx.gpt2.model import GPT2ForSamplingWithContextBroadcasting
from transformers_neuronx.module import save_pretrained_split
from transformers import AutoModelForCausalLM, AutoTokenizer
os.environ['NEURON_CC_FLAGS'] = '--model-type=transformer-inference' # Apply optimal

# Load and save the CPU model with bfloat16 casting
model_cpu = AutoModelForCausalLM.from_pretrained('gpt2')
save_pretrained_split(model_cpu, 'gpt2-split')

# Get a tokenizer and exaple input
tokenizer = AutoTokenizer.from_pretrained('gpt2')
text = "Hello, I'm a generative AI language model. Generative AI is a type of AI that can create new content and ideas, including conversations, stories, images, videos, and music. It is powered by large models that are pre-trained on vast amounts of data and commonly referred to as foundation models (FMs). With generative AI on AWS, you can reinvent your applications, create entirely new customer experiences, drive unprecedented levels of productivity, and transform your business. "
encoded_input = tokenizer(text, return_tensors='pt')

# Set the number of tokens that will be processed in parallel
prompt_len = encoded_input.input_ids.shape[1]
context_length_estimate = int(2 ** math.ceil(math.log(prompt_len, 2))) # Use the closest power of two bucket size

# Create and compile the Neuron model
model_neuron = GPT2ForSamplingWithContextBroadcasting.from_pretrained('gpt2-split', batch_size=1, tp_degree=2, n_positions=256, amp='bf16', context_length_estimate=context_length_estimate)
model_neuron.to_neuron()

# Run inference
with torch.inference_mode():
    generated_sequence = model_neuron.sample(encoded_input.input_ids, sequence_length=256, start_ids=None)
    print([tokenizer.decode(tok) for tok in generated_sequence])
```

The `GPT2ForSamplingWithContextBroadcasting` class can also process
an input prompt that has a different batch size from the batch size of the
autoregressive decoder output. For example, an input prompt with batch size = 1 can
be used to produce an output of batch size = 5 to generate multiple suggestions
for the same input prompt. The input prompt batch size can be specified using
the `prompt_batch_size` argument and the autoregressive decoder output batch
size can be specified using the `batch_size` argument. In the following example
we demonstrate how to apply parallel context encoding to the `GPT2` model
to generate 5 outputs for a single input.

```
import os
import math
import torch
from transformers_neuronx.gpt2.model import GPT2ForSamplingWithContextBroadcasting
from transformers_neuronx.module import save_pretrained_split
from transformers import AutoModelForCausalLM, AutoTokenizer
os.environ['NEURON_CC_FLAGS'] = '--model-type=transformer-inference'

# Load and save the CPU model with bfloat16 casting
model_cpu = AutoModelForCausalLM.from_pretrained('gpt2')
save_pretrained_split(model_cpu, 'gpt2-split')

# Get a tokenizer and exaple input
tokenizer = AutoTokenizer.from_pretrained('gpt2')
text = "Hello, I'm a generative AI language model. Generative AI is a type of AI that can create new content and ideas, including conversations, stories, images, videos, and music. It is powered by large models that are pre-trained on vast amounts of data and commonly referred to as foundation models (FMs). With generative AI on AWS, you can reinvent your applications, create entirely new customer experiences, drive unprecedented levels of productivity, and transform your business. "
encoded_input = tokenizer(text, return_tensors='pt')

# Set the number of tokens that will be processed in parallel
prompt_len = encoded_input.input_ids.shape[1]
context_length_estimate = int(2 ** math.ceil(math.log(prompt_len, 2))) # Use the closest power of two bucket size

# Create and compile the Neuron model
model_neuron = GPT2ForSamplingWithContextBroadcasting.from_pretrained('gpt2-split', prompt_batch_size=1, batch_size=5, tp_degree=2, n_positions=256, amp='bf16', context_length_estimate=context_length_estimate)
model_neuron.to_neuron()

# Run inference
with torch.inference_mode():
    generated_sequence = model_neuron.sample(encoded_input.input_ids, sequence_length=256, start_ids=None)
for i, output in enumerate(generated_sequence):
    print('-'*50)
    print(f'Batch {i} output:')
    print(tokenizer.decode(output))
```


## [Experimental] Serialization support

Transformers Neuron supports model serialization (model saving and loading) for
the `GPT2` model class. Serialization support for additional model classes
will be added in an uncoming relesae. In the following example we demonstrate
how to save and load the `GPT2` model:

```
import os
import torch
from transformers_neuronx.gpt2.model import GPT2ForSampling
from transformers_neuronx.generation_utils import HuggingFaceGenerationModelAdapter
from transformers_neuronx.module import save_pretrained_split
from transformers import AutoModelForCausalLM, AutoTokenizer
os.environ['NEURON_CC_FLAGS'] = '--model-type=transformer-inference'

# Load and save the CPU model
model_cpu = AutoModelForCausalLM.from_pretrained('gpt2')
save_pretrained_split(model_cpu, 'gpt2-split')

# Create and compile the Neuron model
model_neuron = GPT2ForSampling.from_pretrained('gpt2-split', batch_size=1, tp_degree=2, n_positions=256, amp='f32', unroll=None)
model_neuron.to_neuron()

# Save the compiled Neuron model
model_neuron._save_compiled_artifacts('gpt2-neuron')

# Load the Neuron model
model_neuron = GPT2ForSampling.from_pretrained('gpt2-split', batch_size=1, tp_degree=2, n_positions=256, amp='f32', unroll=None)
model_neuron._load_compiled_artifacts('gpt2-neuron') # Load the compiled Neuron artifacts
model_neuron.to_neuron() # Load the model weights but skip compilation
# Get a tokenizer and exaple input
tokenizer = AutoTokenizer.from_pretrained('gpt2')
text = "Hello, I'm a language model,"
encoded_input = tokenizer(text, return_tensors='pt')

# Run inference
with torch.inference_mode():
    generated_sequence = model_neuron.sample(encoded_input.input_ids, sequence_length=256, start_ids=None)
    print([tokenizer.decode(tok) for tok in generated_sequence])
```

## model-type=transformer-inference Compiler Flag

We recommend using the `--model-type=transformer-inference` compiler flag for optimized
decoder-only LLM inference. In a future release, this compiler flag may be enabled
by default. This compiler flag can be enabled via the `NEURON_CC_FLAGS` environment
variable:

```
export NEURON_CC_FLAGS="--model-type=transformer-inference"
```

## Running inference with multiple models

Multiple transformers-neuronx models can be loaded at the same time as long
as the total number of consumed NeuronCores is less than or equal to the total
number of NeuronCores on the instance. For example, three tp-degree=8 models can be
loaded and run in parallel on an inf2.48xlarge which has 24 NeuronCores. The
`NEURON_RT_NUM_CORES` and `NEURON_RT_VISIBLE_CORES` environment variables
can be used to allocate the necessary number of NeuronCores to each process
to run multiple transformers-neuronx models in parallel. See the
[NeuronCore Allocation and Model Placement for Inference](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/frameworks/torch/torch-neuronx/programming-guide/inference/core-placement.html#torch-neuronx-core-placement-guide)
section for additional information about how to use these environment variables.

# Examples

The `examples` folder contains tutorials for running autoregressive sampling using HuggingFace
transformers checkpoints. For example, `examples/facebook-opt-13b-sampling.md` contains instructions
for running HuggingFace `facebook/opt-13b` autoregressive sampling on a trn1.2xlarge instance.

# Currently supported models

-  [GPT2](https://huggingface.co/docs/transformers/model_doc/gpt2)
-  [GPT-J](https://huggingface.co/docs/transformers/model_doc/gptj)
-  [OPT](https://huggingface.co/docs/transformers/model_doc/opt)
-  [GPT-Neox [Experimental]](https://huggingface.co/docs/transformers/model_doc/gpt_neox)
-  [Bloom [Experimental]](https://huggingface.co/docs/transformers/model_doc/bloom)
-  [LLaMA [Prototype]](https://huggingface.co/docs/transformers/main/model_doc/llama)

# Upcoming features

## Performance metrics

The ``transformers-neuronx`` samples currently provide limited
performance data. We are looking into adding additional metrics, such as
``tokens / second`` and latency measurements.

# Troubleshooting

Please refer to our [Contact
Us](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/contact.html)
page for additional information and support resources. If you intend to
file a ticket and you can share your model artifacts, please re-run your
failing script with ``NEURONX_DUMP_TO=./some_dir``. This will dump
compiler artifacts and logs to ``./some_dir``. You can then include this
directory in your correspondance with us. The artifacts and logs are
useful for debugging the specific failure.

# Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

# License

This library is licensed under the Apache License 2.0 License.