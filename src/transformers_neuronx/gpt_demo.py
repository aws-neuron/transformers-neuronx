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
from transformers import GPTJConfig as GPTJConfigTransformer
from transformers_neuronx import dtypes
from transformers_neuronx.module import save_pretrained_split

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
    run_parser.add_argument('--batch_size', type=int, default=4, help="Input batch size")
    run_parser.add_argument('--n_positions', type=int, default=128, help="Input sequence length")
    run_parser.add_argument('--tp_degree', type=int, default=2, help="Number of neuron cores used for tensor parallel")
    run_parser.add_argument('--unroll', type=int, default=None)
    run_parser.add_argument('--print_latency', action='store_true', help="Print latency for generation of each output token")
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
        if "GPTJ" in str(model_cls):
            config = GPTJConfigTransformer(**config)
        else:
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
    prompt_text = "Hello, I'm a language model,"
    print(f'running {model_cls.__name__}.from_pretrained')
    model = model_cls.from_pretrained(args.load, batch_size=args.batch_size, amp=args.amp,
                                      tp_degree=args.tp_degree, n_positions=args.n_positions,
                                      unroll=args.unroll)
    if args.print_latency:
        latency_printer = LatencyPrinter()
        model.register_forward_pre_hook(latency_printer.pre_hook)
        model.register_forward_hook(latency_printer.hook)
    if hasattr(model, 'register_to_neuron_hook'):
        model.register_to_neuron_hook(lambda idx: print(f'done to_neuron layer {idx}'))
    print('running model.to_neuron')
    model.to_neuron()
    with torch.inference_mode():
        encoded_text = tokenizer.encode(prompt_text)
        input_ids = torch.as_tensor([encoded_text])
        input_ids = torch.cat([input_ids for _ in range(args.batch_size)], dim=0)
        print('running model.sample')
        generated_sequence = model.sample(input_ids, sequence_length=args.n_positions)
        print('generated_sequence=', generated_sequence)
        outputs = [tokenizer.decode(gen_seq) for gen_seq in generated_sequence]
    print(outputs)


class LatencyPrinter:

    def __init__(self):
        self.start = None

    def pre_hook(self, module, input):
        if len(input) == 3:
            _, cache_offset, _ = input
            print(f'cache_offset: {cache_offset}')
        self.start = time.time()

    def hook(self, *args):
        latency_ms = math.ceil((time.time() - self.start) * 1000)
        print(f'Latency: {latency_ms} ms')
