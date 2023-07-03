# Transformers Neuron 0.5.0 Release Notes

Date: 2023-07-03

## What's New?

- [Experimental] Added support for GPT-NeoX models.
- [Experimental] Added support for BLOOM models.
- [Experimental] Added support for LLaMA models.
- Added support for more flexible tensor-parallel configurations to GPT2, OPT, and BLOOM. Previously, we had two constraints on `tp_degree`: 1) The attention heads needs to be evenly divisible by `tp_degree` 2) The `tp_degree` needs to satisfy the runtime topologies constraint for collective communication (i.e Allreduce). For more details on supported topologies, see: [Tensor-parallelism support](README.md#tensor-parallelism-support) and https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/arch/neuron-features/collective-communication.html. We now get rid of 1) by using 1-axis padding.
- Added multi-query / multi-group attention support for GPT2.

## Bug Fixes

- Resolved accuracy issues which caused incorrect output.
- Resolved an issue where NaN values could be produced when the context_length argument was used in GPT2/OPT.

## Known Issues and Limitations

- Missing cache reorder support for beam search. 

# Transformers Neuron 0.4.0 Release Notes

Date: 2023-06-12

## What's New?

- Added ``int8`` weight storage for `GPT2` models.
- Improved prompt context encoding performance for `GPT2` models.
- Improved collective communications performance for tp-degrees 4, 8, and 24 on Inf2.
- Improved collective communications performance for tp-degrees 8 and 32 on Trn1.
- Support for the ``--model-type=transformer-inference`` compiler flag for optimized decoder-only LLM inference.

## Bug Fixes

- Added padding to the `GPT-J` ``linear`` layer to correctly handle odd vocabulary sizes.
- Issues where the HuggingFace `generate` method produces incorrect results when
`beam_search` is used have been resolved.


# Transformers Neuron 0.3.0 Release Notes

Date: 2023-04-28

## What's New?

- Added ``transformers-neuronx`` artifacts to PyPI repository.
- Added support for the the Hugging Face [generate()](https://huggingface.co/docs/transformers/v4.28.1/en/main_classes/text_generation#transformers.GenerationMixin.generate)
- Added support for model serialization, including model saving, loading, and
  weight swapping.
- Added support for caching compiled artifacts.
- Improved performance by removing unnecessary KV-cache tensor resetting.
- Improved prompt context encoding performance (`OPT`, `GPT2`).

## Bug Fixes

- Incorrect `GPT-J` ``amp_callback`` import: Fixed the `GPT-J` demo now imports the correct ``amp_callback`` function.

## Known Issues and Limitations

Incorrect output with HuggingFace `beam_search`: When the HuggingFace `generate` method is configured to use `beam_search`, this
can produce incorrect results for certain configurations. It is recommended to
use other generation methods such as `sample` or `greedy_search`.


# Transformers Neuron 0.2.0 Release Notes

Date: 2023-02-24

## What's New?
	 
- Added error handling to check if the desired generated sequence length is valid based on the model configuration
- Improved logging:
   - Reduced overly verbose compiler messages
   - Disabled lazy module warnings
	 
## Bug Fixes

- Updated `src/transformers_neuronx/gptj/demo.py` to correctly use the `amp_callback` function from `transformers_neuronx.gpt2.demo` 
- Extend the `gpt_demo.py` `save` function to support GPT-2 and GPT-J configs
	 
# Transformers Neuron 0.1.0 Release Notes

Date: 2023-02-08

First release of `transformers-neuronx`, a new library that enables LLM model inference on Inf2 & Trn1 using the Neuron SDK. `transformers-neuronx` contains optimized model implementations that are checkpoint-compatible with HuggingFace Transformers, and currently supports Transformer Decoder models like GPT2, GPT-J and OPT.