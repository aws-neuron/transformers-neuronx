import os
os.environ["XLA_FLAGS"] = " --xla_dump_to=dump"
os.environ["NEURON_FRAMEWORK_DEBUG"] = "1"

from llama_demo import demo
from transformers_neuronx.llama.model import LlamaForSampling

def amp_callback(model, dtype):
    model.to(dtype)


def main():
    demo('meta-llama/Llama-3.1-8B-Instruct', LlamaForSampling, amp_callback)


if __name__ == "__main__":
    main()