Transformers Neuron for Trn1 and Inf2 is a software package that enables
PyTorch users to perform large language model (LLM) inference on
second-generation Neuron hardware (See: [NeuronCore-v2](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/arch/neuron-hardware/neuron-core-v2.html)). The [Neuron performance page](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/benchmarks/inf2/inf2-performance.html#large-language-models-inference-performance) lists expected inference performance for commonly used Large Language Models.

# Transformers Neuron (``transformers-neuronx``) Documentation
Please refer to the [Transformers Neuron documentation](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/transformers-neuronx/) for setup and developer guides.

# Installation

## Stable Release

To install the most rigorously tested stable release, use the PyPI pip wheel:

```
pip install transformers-neuronx --extra-index-url=https://pip.repos.neuron.amazonaws.com
```

## Development Version

The AWS Neuron team is currently restructuring the contribution model of this github repository. This github repository content
does not reflect latest features and improvements of transformers-neuronx library. Please install the stable release version
from https://pip.repos.neuron.amazonaws.com to get latest features and improvements.

# Release Notes and Supported Models

Please refer to the [transformers-neuronx release notes](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/release-notes/torch/transformers-neuronx/index.html) to see the latest supported features and models.



# Troubleshooting

Please refer to our [Support
Page](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/support.html)
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
