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
import math
import time
import torch
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers_neuronx import dtypes
from transformers_neuronx.module import save_pretrained_split


def demo(model_name, model_cls, amp_callback):
    parser = argparse.ArgumentParser()
    parser.add_argument('--amp', default='f32', choices=['f32', 'f16', 'bf16'])
    parser.add_argument('--model_name', default=None)
    subparsers = parser.add_subparsers()
    save_name = 'save'
    save_parser = subparsers.add_parser(save_name)
    save_parser.set_defaults(which=save_name)
    save_parser.add_argument('save')
    run_name = 'run'
    run_parser = subparsers.add_parser(run_name)
    run_parser.set_defaults(which=run_name)
    run_parser.add_argument('load')
    run_parser.add_argument('--batch_size', type=int, default=4)
    run_parser.add_argument('--n_positions', type=int, default=128)
    run_parser.add_argument('--tp_degree', type=int, default=2)
    run_parser.add_argument('--unroll', type=int, default=None)
    run_parser.add_argument('--print_latency', action='store_true')
    args = parser.parse_args()
    if args.model_name is not None:
        model_name = args.model_name
    if args.which == save_name:
        save(args, model_name, amp_callback)
    elif args.which == run_name:
        run(args, model_name, model_cls)


def save(args, model_name, amp_callback):
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

    def pre_hook(self, *args):
        self.start = time.time()

    def hook(self, *args):
        latency_ms = math.ceil((time.time() - self.start) * 1000)
        print(f'Latency: {latency_ms} ms')
