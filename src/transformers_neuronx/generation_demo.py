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
import argparse
import itertools
import math
import time
import json
import os
import torch
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import AutoConfig
from transformers_neuronx import dtypes
from transformers_neuronx.module import save_pretrained_split
from transformers_neuronx.gpt2.model import GPT2ForSampling
from transformers_neuronx.opt.model import OPTForSampling
from transformers_neuronx.gptj.model import GPTJForSampling
from transformers_neuronx.generation_utils import HuggingFaceGenerationModelAdaptor


def amp_callback(model_type, model, dtype):

    if model_type == "opt":
        for block in model.model.decoder.layers:
            block.self_attn.to(dtype)
            block.fc1.to(dtype)
            block.fc2.to(dtype)
        model.lm_head.to(dtype)
    elif model_type == "gpt2" or model_type == "gptj":
        # cast attention and mlp to low precisions only; layernorms stay as f32
        for block in model.transformer.h:
            block.attn.to(dtype)
            block.mlp.to(dtype)
        model.lm_head.to(dtype)

    return model

def main():
    parser = argparse.ArgumentParser()
    amp_choices = ['f32', 'f16', 'bf16']
    floatx_floaty_combinations = list(itertools.product(amp_choices, amp_choices))
    for floatx, floaty in floatx_floaty_combinations:
        amp_choices.append(f'{floatx}-u8-{floaty}')
    parser.add_argument('model_type')
    parser.add_argument('--amp', default='f32', choices=amp_choices)
    subparsers = parser.add_subparsers()
    save_name = 'save'
    save_parser = subparsers.add_parser(save_name)
    save_parser.set_defaults(which=save_name)
    save_parser.add_argument('save', help="Directory to save the model")
    save_parser.add_argument('--random', action='store_true', help="Random weights flag. If true, config.json would be used to generate a model with random weight")
    save_parser.add_argument('--config', type=str, default='', help="Path to config.json file (example: path/to/config.json)")
    run_name = 'run'
    run_parser = subparsers.add_parser(run_name)
    run_parser.set_defaults(which=run_name)
    run_parser.add_argument('load')
    run_parser.add_argument('--batch_size', type=int, default=2, help="Input batch size")
    run_parser.add_argument('--n_positions', type=int, default=128, help="Input sequence length")
    run_parser.add_argument('--tp_degree', type=int, default=2, help="Number of neuron cores used for tensor parallel")
    run_parser.add_argument('--unroll', type=int, default=None)
    run_parser.add_argument('--print_latency', action='store_true', help="Print latency for generation of each output token")
    run_parser.add_argument('--device', type=str, default="cpu")
    run_parser.add_argument('--beam', type=int, default=1)
    run_parser.add_argument('--do_sample', action='store_true')
    run_parser.add_argument('--max_length', type=int, default=30)
    run_parser.add_argument('--top_k', type=int, default=0)
    run_parser.add_argument('--top_p', type=float, default=1.0)
    run_parser.add_argument('--temperature', type=float, default=1.0)
    run_parser.add_argument('--no_repeat_ngram_size', type=int, default=0)
    run_parser.add_argument('--various', action='store_true', help="Generated batched sequence with different length if set; otherwise using same length")
    run_parser.add_argument('--prompt_len', type=int, default=30)

    args = parser.parse_args()

    model_type = args.model_type
    model_cls = None
    hf_model_name = None
    if model_type == 'gpt2':
        model_cls = GPT2ForSampling
        hf_model_name = 'gpt2'
    elif model_type == 'opt':
        model_cls = OPTForSampling
        hf_model_name = 'facebook/opt-125m'
    elif model_type == "gptj":
        model_cls = GPTJForSampling
        hf_model_name = 'EleutherAI/gpt-j-6B'
    assert model_cls is not None, f"Invalid model_type: {model_type}"

    print(f"Running demo with model_type:{model_type}, hf_model_name:{hf_model_name}")

    if args.which == save_name:
        save(args, hf_model_name, model_type)
    elif args.which == run_name:
        run(args, hf_model_name, model_cls)


def save(args, hf_model_name, model_type):
    model = AutoModelForCausalLM.from_pretrained(hf_model_name, low_cpu_mem_usage=True)
    if args.amp != 'f32':
        dtype = dtypes.to_torch_dtype(args.amp)
        amp_callback(model_type, model, dtype)
    save_pretrained_split(model, args.save)


def run(args, hf_model_name, model_cls):
    torch.manual_seed(15213)

    full_prompt_text = "Hello, I'm a language model, not a programming language. \
        I don't know how to write a program, but I know that I can do it.\n\n \
        I'm not going to tell you how I learned to code, or how much I've learned. \
        But I will tell the story of my first programming experience, and how it changed my life."
    if args.various:
        batched_seq_lens = torch.randint(len(full_prompt_text) // 3,
            len(full_prompt_text), (args.batch_size,)).tolist()
    else:
        batched_seq_lens = [args.prompt_len for _ in range(args.batch_size)]

    batched_prompt_text = [full_prompt_text[:l] for l in batched_seq_lens]
    print(batched_prompt_text)

    tokenizer = AutoTokenizer.from_pretrained(hf_model_name, padding_side="left")
    tokenizer.pad_token_id = tokenizer.eos_token_id

    encoded_text = tokenizer(batched_prompt_text, padding=True, return_tensors="pt")
    print(encoded_text)
    input_ids = torch.as_tensor(encoded_text.input_ids)

    if args.device == "neuron":
        os.environ["NEURONX_CACHE"] = "on"
        os.environ["NEURONX_DUMP_TO"] = f"neruonx_cache_{hf_model_name}_{args.batch_size}"

        print(f'running {model_cls.__name__}.from_pretrained')
        config = AutoConfig.from_pretrained(args.load)
        neuron_model = model_cls.from_pretrained(args.load, batch_size=args.batch_size, amp=args.amp,
                                        tp_degree=args.tp_degree, n_positions=args.n_positions,
                                        unroll=args.unroll)
        neuron_model.to_neuron()
        model = HuggingFaceGenerationModelAdaptor(config, neuron_model)
        print('running model.to_neuron')
    else:
        print(f'running {model_cls.__name__}.from_pretrained')
        model = AutoModelForCausalLM.from_pretrained(hf_model_name, low_cpu_mem_usage=True)

    with torch.inference_mode():
        generated_sequence = model.generate(
            input_ids=encoded_text.input_ids,
            attention_mask=encoded_text.attention_mask,
            num_beams=args.beam,
            do_sample=args.do_sample,
            max_new_tokens=args.max_length,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            top_k=args.top_k,
            top_p=args.top_p,
            temperature=args.temperature,
        )

    print('generated_sequence=', generated_sequence)
    outputs = [tokenizer.decode(gen_seq) for gen_seq in generated_sequence]
    print(outputs)

if __name__ == "__main__":
    main()