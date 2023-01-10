# Introduction

This repository contains the source code of the AWS Neuron Transformers integration project.
As it stands now, it mainly serves the purpose of running autoregressive sampling workflows on
the Neuron platform. The source code of Neuron optimized module classes such as `OPTForSampling`
and `GPT2ForSampling` can demonstrate how autoregressive sampling is implemented on the Neuron
compiler and runtime system.

Note: This repository is **actively** in development. The Neuron team is still heavily modifying
the Neuron optimized module classes. The functionality provided in this repository will not maintain
long-term API stability until version >= 1.0.0. For applications willing to reuse code from this
repository, we recommend treating the Neuron optimized module implementations as samples, and pin
the version of the main library package `torch-neuronx` to avoid breaking interface changes as new
features are developed.

# Installation

The repository installs through standard `python setup.py install`. Additionally, if the `wheel`
package is installed, then `python setup.py bdist_wheel` can generate an installable `whl` package
under the `dist` folder.

# Examples

The `examples` folder contains tutorials for running autoregressive sampling using HuggingFace
transformers checkpoints. For example, `examples/facebook-opt-13b-sampling.md` contains instructions
for running HuggingFace `facebook/opt-13b` autoregressive sampling on a trn1.2xlarge instance.
