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
