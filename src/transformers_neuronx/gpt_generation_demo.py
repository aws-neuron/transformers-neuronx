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
from transformers import GPT2Config as GPT2ConfigTransformer
from transformers_neuronx import dtypes
from transformers_neuronx.module import save_pretrained_split
from transformers_neuronx.gpt2.model import GPT2ForHuggingFaceSampling

def demo(model_name, model_cls, amp_callback):
    parser = argparse.ArgumentParser()
    amp_choices = ['f32', 'f16', 'bf16']
    floatx_floaty_combinations = list(itertools.product(amp_choices, amp_choices))
    for floatx, floaty in floatx_floaty_combinations:
        amp_choices.append(f'{floatx}-u8-{floaty}')
    parser.add_argument('--amp', default='f32', choices=amp_choices)
    parser.add_argument('--model_name', default=None, help="Model name for loading a pretrained model")
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
    run_parser.add_argument('--prompt_len', type=int, default=10)

    args = parser.parse_args()
    if args.model_name is not None:
        model_name = args.model_name
    if args.which == save_name:
        save(args, model_name, amp_callback, model_cls)
    elif args.which == run_name:
        run(args, model_name, model_cls)

def load_config(args):
    config_filename = args.config
    assert config_filename, "Please provide the config.json like: --config=./config.json"
    assert os.path.exists(config_filename), f"File {config_filename} does not exist."
    config = json.load(open(config_filename))
    return config

def save(args, model_name, amp_callback, model_cls):
    if args.random:
        config = load_config(args)
        config = GPT2ConfigTransformer(**config)
        model = AutoModelForCausalLM.from_config(config=config)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True)
    if args.amp != 'f32':
        dtype = dtypes.to_torch_dtype(args.amp)
        amp_callback(model, dtype)
    save_pretrained_split(model, args.save)


def run(args, model_name, model_cls):
    torch.manual_seed(15213)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

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

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    tokenizer.pad_token_id = tokenizer.eos_token_id

    encoded_text = tokenizer(batched_prompt_text, padding=True, return_tensors="pt")
    print(encoded_text)
    input_ids = torch.as_tensor(encoded_text.input_ids)

    if args.device == "neuron":
        os.environ["NEURONX_DUMP_TO"] = f"neruonx_cache_{args.model_name}_{args.batch_size}"

        print(f'running {model_cls.__name__}.from_pretrained')
        model = model_cls.from_pretrained(args.load, batch_size=args.batch_size, amp=args.amp,
                                        tp_degree=args.tp_degree, n_positions=args.n_positions,
                                        unroll=args.unroll)
        print('running model.to_neuron')
        model.to_neuron()
        model.reset()
    else:
        print(f'running {model_cls.__name__}.from_pretrained')
        model = AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True)

    with torch.inference_mode():

        generated_sequence = model.generate(input_ids,
            num_beams=args.beam,
            do_sample=args.do_sample,
            max_length=args.max_length,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            top_k=args.top_k,
            top_p=args.top_p,
            temperature=args.temperature,
            attention_mask=encoded_text.attention_mask)

    print('generated_sequence=', generated_sequence)
    outputs = [tokenizer.decode(gen_seq) for gen_seq in generated_sequence]
    print(outputs)

def amp_callback(model, dtype):
    # cast attention and mlp to low precisions only; layernorms stay as f32
    for block in model.transformer.h:
        block.attn.to(dtype)
        block.mlp.to(dtype)
    model.lm_head.to(dtype)
demo('gpt2', GPT2ForHuggingFaceSampling, amp_callback)
