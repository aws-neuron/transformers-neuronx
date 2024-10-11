import transformers
from transformers import AutoConfig
import transformers_neuronx

NEURON_MODEL_FOR_CAUSAL_LM_MAPPING = {
    "bloom":      transformers_neuronx.BloomForSampling,
    "code_llama": transformers_neuronx.LlamaForSampling,
    "gpt2":       transformers_neuronx.GPT2ForSamplingWithContextBroadcasting,
    "gpt_neox":   transformers_neuronx.GPTNeoXForSampling,
    "gptj":       transformers_neuronx.GPTJForSampling,
    "llama":      transformers_neuronx.LlamaForSampling,
    "mistral":    transformers_neuronx.MistralForSampling,
    "mixtral":    transformers_neuronx.MixtralForSampling,
    "opt":        transformers_neuronx.OPTForSampling,
    "qwen2":       transformers_neuronx.Qwen2ForSampling,
}


CONFIG_MAPPING = {
    transformers.BloomConfig:   "bloom",
    transformers.LlamaConfig:   "llama",
    transformers.GPT2Config:    "gpt2",
    transformers.GPTNeoXConfig: "gpt_neox",
    transformers.GPTJConfig:    "gptj",
    transformers.MistralConfig: "mistral",
    transformers.MixtralConfig: "mixtral",
    transformers.OPTConfig:     "opt",
    transformers.Qwen2Config:   "qwen2",
}


class NeuronAutoModelForCausalLM:

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path)

        if type(config) in CONFIG_MAPPING:
            model_type = CONFIG_MAPPING[type(config)]
        elif hasattr(config, "model_type"): # Fallback for custom/derived config objects
            model_type = config.model_type
        else:
            raise AssertionError(f"Models based on '{type(config)}' are not supported by Neuron")

        if model_type not in NEURON_MODEL_FOR_CAUSAL_LM_MAPPING:
            raise AssertionError(f"The configuration model type '{model_type}' is not supported by Neuron")

        model_class = NEURON_MODEL_FOR_CAUSAL_LM_MAPPING[model_type]
        return model_class.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
