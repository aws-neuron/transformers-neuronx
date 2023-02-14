# Introduction

This repository contains the source code of the AWS Neuron Transformers integration project.
As it stands now, it mainly serves the purpose of running transformer decoder inference
(autoregressive sampling) workflows on the Neuron platform.

Note: This repository is **actively** in development. The Neuron team is still heavily modifying
the Neuron optimized module classes. The functionality provided in this repository will not maintain
long-term API stability until version >= 1.0.0. For applications willing to reuse code from this
repository, we recommend treating the Neuron optimized module implementations as samples, and pin
the version of the main library package `torch-neuronx` to avoid breaking interface changes as new
features are developed.

# Installation

If `git` is installed:

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

`transformers-neuronx` is checkpoint-compatible with HuggingFace Transformers. While the Neuron
team reimplemented some HuggingFace Transformers models from scratch for the purpose of maximizing
the execution efficiency of transformer decoders on Neuron, the implementations are done with
maximizing compatibility in mind, meaning one can train transformer decoder models, say GPT2, using
the standard HuggingFace Transformers library, and then construct an inference-optimized decoder
model using transformers-neuronx's GPT2ForSampling class. If training was done with other libraries
such as MegatronLM, then it is still possible to convert the obtained checkpoint to the standard
HuggingFace Transformers checkpoint format, and then move on to transformers-neuronx's optimized
decoder implementations.

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

For transformer decoders used in large language models, tensor-parallelism is neccessary as it
provides a way to shard the models' large weight matrices onto multiple NeuronCores, and having
NeuronCores working on the same matrix multiply operation collaboratively. transformers-neuronx's
tensor-parallelism support makes heavy use of collective operations such as all-reduce, which is
supported natively by the Neuron runtime.

There are some principles for setting tensor-parallelism degree (number of NeuronCores participating
in sharded matrix multiply operations) for Neuron-optimized transformer decoder models.

1. The number of attention heads needs to be divisible by the tensor-parallelism degree.
2. The total data size of model weights and key-value caches needs to be smaller than 16 GB times
the tensor-parallelism degree.
3. Currently, the Neuron runtime supports tensor-parallelism degrees 1, 2, 8, and 32.

Some examples:

1. `facebook/opt-13b` has 40 attention heads, and when running at batch size 1 and float16 precision
the model requires ~29 GB memory, therefore a `trn1.2xlarge` with 32 GB device memory is sufficient.
2. `facebook/opt-30b` has 56 attention heads, and at batch size 1 and float16 precision the model
requires ~66 GB memory, therefore it can run on 8 NeuronCores on one `trn1.32xlarge` using 128 GB
device memory.
3. `gpt2-xl` has 25 attention heads and requires ~4 GB memory at bfloat16 precision. It runs without
tensor-parallelism only.

# Examples

The `examples` folder contains tutorials for running autoregressive sampling using HuggingFace
transformers checkpoints. For example, `examples/facebook-opt-13b-sampling.md` contains instructions
for running HuggingFace `facebook/opt-13b` autoregressive sampling on a trn1.2xlarge instance.

# Currently supported models

- [EleutherAI/gpt-j-6B](https://huggingface.co/EleutherAI/gpt-j-6B)
- [facebook/opt-125m](https://huggingface.co/facebook/opt-125m)
- [facebook/opt-1.3b](https://huggingface.co/facebook/opt-1.3b)
- [facebook/opt-2.7b](https://huggingface.co/facebook/opt-2.7b)
- [facebook/opt-6.7b](https://huggingface.co/facebook/opt-6.7b)
- [facebook/opt-13b](https://huggingface.co/facebook/opt-13b)
- [facebook/opt-30b](https://huggingface.co/facebook/opt-30b)
- [gpt2](https://huggingface.co/gpt2)
- [gpt2-medium](https://huggingface.co/gpt2-medium)
- [gpt2-large](https://huggingface.co/gpt2-large)
- [gpt2-xl](https://huggingface.co/gpt2-xl)


# Troubleshooting

## ImportError: WRONG PACKAGE

An error is generated upon importing the package.

### Action:
```
import transformers_neuronx
```

### Error:
```
ImportError: WRONG PACKAGE. Please install the package from Neuron Repository - https://github.com/aws-neuron/transformers-neuronx#installation
```

### Resolution:
This error occurs when the `transformers_neuronx` package is installed from
PyPI rather than from GitHub. The package that is available on
https://pypi.org/ is a stub that ensures that malicious packages are not
uploaded. The `transformers_neuronx` package is intended to be installed
directly from this git repository using the [Installation](#installation)
instructions above.
