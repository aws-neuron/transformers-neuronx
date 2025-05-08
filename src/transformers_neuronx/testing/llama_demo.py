import argparse
import itertools
import math
import time
import json
import os
import torch
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers_neuronx import dtypes
from transformers_neuronx.module import save_pretrained_split
from transformers_neuronx.config import NeuronConfig, QuantizationConfig
from transformers_neuronx.sparse_attn_utils import SparseAttnConfig, BlkSparseAttnConfig, SlidingWindowAttnConfig
from transformers_neuronx import global_debugger
import requests, re
from transformers import LlamaConfig

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
    run_parser.add_argument('--quantize', action='store_true', help="Quantize model")
    # Sparse attention configs
    run_parser.add_argument('--sparse_attn', type=str, choices=[None, 'blk_sparse', 'custom', 'window'],
                            default=None, help="Use sparse attention or not. ")
    # Block-sparse configs
    run_parser.add_argument('--blk_size', type=int, default=128, help="Block size in blk-sparse attention")
    run_parser.add_argument('--num_global_blks', type=int, default=0, help="Number of global blocks in blk-sparse attention")
    run_parser.add_argument('--num_local_blks', type=int, default=1, help="Number of local blocks in blk-sparse attention")
    run_parser.add_argument('--num_random_blks', type=int, default=0, help="Number of random blocks in blk-sparse attention")
    # Window attention configs
    run_parser.add_argument('--window_size', type=int, default=128, help="Window size for sliding-window attention. ")
    run_parser.add_argument('--context_length_estimate', type=int, default=None, help="Context length estimate.")
    run_parser.add_argument('--fuse_qkv', action='store_true')
    run_parser.add_argument('--attention_layout', type=str, default='HSB')
    run_parser.add_argument('--collectives_layout', type=str, default='HSB')
    run_parser.add_argument('--cache_layout', type=str, default='SBH')
    run_parser.add_argument('--shard_over_sequence', action='store_true')
    run_parser.add_argument('--gqa', type=str, default=None)
    run_parser.add_argument('--sequence_parallel_norm', action='store_true')
    run_parser.add_argument('--debug', action='store_true')
    run_parser.add_argument('--padding_side', type=str, default='left')
    # TODO: args for custom sparse attention not added

    args = parser.parse_args()
    if args.model_name is not None:
        model_name = args.model_name
    if args.which == save_name:
        save(args, model_name, amp_callback, model_cls)
    elif args.which == run_name:
        run(args, model_name, model_cls)


def save(args, model_name, amp_callback, model_cls):
    if args.random:
        config = load_config(args)
        config = LlamaConfig(**config)
        model = AutoModelForCausalLM.from_config(config=config)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True)
    if args.amp != 'f32':
        dtype = dtypes.to_torch_dtype(args.amp)
        amp_callback(model, dtype)
    save_pretrained_split(model, args.save)


def run(args, model_name, model_cls):
    from pprint import pprint
    print("run args: ")
    pprint(vars(args))
    torch.manual_seed(15213)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    prompt_text = "Hello, I'm a language model,"
    print(f'running {model_cls.__name__}.from_pretrained')
    neuron_config = None
    quant_config = QuantizationConfig(dequant_dtype=args.amp) if args.quantize else None
    sparse_config = None
    if args.sparse_attn:
        if args.sparse_attn == "blk_sparse":
            sparse_attn_config = BlkSparseAttnConfig(
                blk_size=args.blk_size,
                num_global_blks=args.num_global_blks,
                num_local_blks=args.num_local_blks,
                num_random_blks=args.num_random_blks
            )
        elif args.sparse_attn == "window":
            sparse_attn_config = SlidingWindowAttnConfig(window_size=args.window_size)
        else:
            raise NotImplementedError("Interface for other attention patterns not implemented yet!")
        sparse_config = SparseAttnConfig(
            attn_type=args.sparse_attn, causal=True, sparse_attn_config=sparse_attn_config,
            same_mask_per_layer=True)
    if (args.quantize or args.sparse_attn or args.fuse_qkv or args.attention_layout or args.shard_over_sequence
        or args.gqa or args.sequence_parallel_norm or args.collectives_layout or args.cache_layout) :
        neuron_config = NeuronConfig(quant=quant_config, sparse_attn=sparse_config, fuse_qkv=args.fuse_qkv,
                                     attention_layout=args.attention_layout, shard_over_sequence=args.shard_over_sequence,
                                     group_query_attention=args.gqa, sequence_parallel_norm=args.sequence_parallel_norm,
                                     collectives_layout=args.collectives_layout, cache_layout=args.cache_layout)
    if args.context_length_estimate:
        model = model_cls.from_pretrained(args.load, batch_size=args.batch_size, amp=args.amp,
                                      tp_degree=args.tp_degree, n_positions=args.n_positions,
                                      unroll=args.unroll, neuron_config=neuron_config,
                                      context_length_estimate=args.context_length_estimate)
    else:
        model = model_cls.from_pretrained(args.load, batch_size=args.batch_size, amp=args.amp,
                                      tp_degree=args.tp_degree, n_positions=args.n_positions,
                                      unroll=args.unroll, neuron_config=neuron_config)
    if args.print_latency:
        latency_printer = LatencyPrinter()
        model.register_forward_pre_hook(latency_printer.pre_hook)
        model.register_forward_hook(latency_printer.hook)
    if hasattr(model, 'register_to_neuron_hook'):
        model.register_to_neuron_hook(lambda idx: print(f'done to_neuron layer {idx}'))
    print('running model.to_neuron')
    model.to_neuron()
    with torch.inference_mode():
        # prompt = re.sub('<[^<]+?>', '', requests.get("https://arxiv.org/html/2402.19427v1").text) # strip html tags
        # prompt += "\n\n========================THE END======================\n"
        # prompt += "A 10 point summary of the paper in simple words: "
        # # put in prompt format https://llama.meta.com/docs/model-cards-and-prompt-formats/llama3_1/#prompt-format
        # prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|> {prompt} <|eot_id|><|start_header_id|>assistant<|end_header_id|>"

        # input_ids = tokenizer.encode(prompt, return_tensors="pt") 
        # num_input_tokens = len(input_ids[0]) # over 26k tokens
        # print(f"num_input_tokens: {num_input_tokens}")
        encoded_text = tokenizer.encode(prompt_text)
        input_ids = torch.as_tensor([encoded_text])
        input_ids = torch.cat([input_ids for _ in range(args.batch_size)], dim=0)
        print('running model.sample')
        if args.debug:
            with global_debugger.debug_context():
                debug_tensors = global_debugger.debug_tensors
                generated_sequence = model.sample(input_ids, sequence_length=args.n_positions)
                for (tag_name, debug_tensor) in debug_tensors.items():
                    print(tag_name)
                    print(debug_tensor.shape)
                    for item in debug_tensor:
                        torch.set_printoptions(threshold=10_000)
                        print(item)
        else:
            generated_sequence = model.sample(input_ids, sequence_length=args.n_positions)
            print('generated_sequence=', generated_sequence)
            outputs = [tokenizer.decode(gen_seq) for gen_seq in generated_sequence]
            print(outputs)
            logits = model.forward(input_ids, sequence_length=args.n_positions)
            for item in logits:
                torch.set_printoptions(threshold=100_000)
                print(item)


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