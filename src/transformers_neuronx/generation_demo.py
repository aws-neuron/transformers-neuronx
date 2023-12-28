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
import os
import sys
import os
import torch
import time
import neuronxcc
import math
import shutil
import logging
logging.basicConfig(level=logging.DEBUG)
import os, psutil

from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import AutoConfig
from transformers_neuronx import dtypes
from transformers_neuronx.config import NeuronConfig
from transformers_neuronx.module import save_pretrained_split
# TODO: make it one-shot import
from transformers_neuronx.gpt2.model import GPT2ForSampling, GPT2ForSamplingWithContextBroadcasting
from transformers_neuronx.opt.model import OPTForSampling
from transformers_neuronx.gptj.model import GPTJForSampling
from transformers_neuronx.bloom.model import BloomForSampling
from transformers_neuronx.llama.model import LlamaForSampling
from transformers_neuronx.gptneox.model import GPTNeoXForSampling
from transformers_neuronx.generation_utils import HuggingFaceGenerationModelAdapter


def dump(dump_dir, dump_file_name, data):
    os.makedirs(dump_dir, exist_ok=True)
    torch.save(data, os.path.join(dump_dir, dump_file_name))


def percentile(data, q: int = 50) -> float:
    """
    Compute the q-th percentile of a collection of measurements.

    Arguments:
        data: The collection of values to compute the percentile on.
        q: The percentile to compute. Expected to be between 0 and 100.

    Returns:
        A single percentile float.
    """
    index = (len(data) - 1) * (q / 100)
    lower = math.floor(index)
    upper = math.ceil(index)
    alpha = index - lower
    data = sorted(data)
    return (data[lower] * (1 - alpha)) + (data[upper] * alpha)

class MetricsCollectorWrapper:

    def __init__(self, neuron_monitor_wrapper=None):
        self.start = None
        self.latency_list = []
        self.latency_collector = LatencyCollector()
        self.neuron_monitor = NeuronMonitor(neuron_monitor_wrapper)

    def pre_hook(self, *args):
        self.latency_collector.pre_hook()
        self.neuron_monitor.pre_hook()

    def hook(self, *args):
        self.latency_collector.hook()
        self.neuron_monitor.hook()


class LatencyCollector:

    def __init__(self):
        self.start = None
        self.latency_list = []

    def pre_hook(self, *args):
        self.start = time.time()

    def hook(self, *args):
        self.latency_list.append(time.time() - self.start)

    def percentile(self, percent):
        latency_list = self.latency_list
        pos_float = len(latency_list) * percent / 100
        max_pos = len(latency_list) - 1
        pos_floor = min(math.floor(pos_float), max_pos)
        pos_ceil = min(math.ceil(pos_float), max_pos)
        latency_list = sorted(latency_list)
        return latency_list[pos_ceil] if pos_float - pos_floor > 0.5 else latency_list[pos_floor]

def benchmark(
    args,
    neuron_model,
    forward_func,
    context_length,
    output_length,
    warmup=2,
    iterations=5
):
    """
    Benchmark autoregressive inference using greedy decoding.

    Arguments:
        model: The model for benchmarking.
        batch_size: The batch size of the input data.
        n_positions: The length of the input prompt + number of generated tokens.
        context_length: The length of the input prompt.
        iterations: The number of iterations for benchmarking.

    Returns:
        A dictionary of performance statisics.
    """

    # Warmup
    print(f"Warmup ({warmup} iterations) ...")
    for _ in range(warmup):
        forward_func()

    # Add latency hooks
    latency_collector_all = LatencyCollector()
    neuron_model.register_forward_pre_hook(latency_collector_all.pre_hook)
    neuron_model.register_forward_hook(latency_collector_all.hook)


    # Benchmark and record durations
    print(f"Benchmark loop ({iterations} iterations) ...")
    latencies = []
    begin = time.time()
    for _ in range(iterations):
        start = time.time()
        forward_func()
        finish = time.time()
        latencies.append((finish - start) * 1000)
    end = time.time()

    latency_list = latency_collector_all.latency_list
    num_inference_per_iter = len(latency_list) // iterations
    context_encoding_latency_list = latency_list[::int(num_inference_per_iter)]

    total_context_encoding_latency = sum(context_encoding_latency_list) * 1000

    # Compute metrics
    boundaries = [0, 50, 90, 95, 99, 100]
    percentiles = {}
    for boundary in boundaries:
        name = f"sequence_latency_p{boundary}"
        percentiles[name] = percentile(latencies, boundary)
    duration = end - begin
    inferences = len(latencies) * args.batch_size
    mean_e2e_token_gen_latency = (sum(latencies) - total_context_encoding_latency) / len(latencies)

    # Metrics
    metrics = {
        "hf_model": args.hf_model,
        "model": args.hf_model,
        "sequence_length": args.n_positions,
        "context_length": context_length,
        "output_length": output_length,
        "batch_size": args.batch_size,
        "batches": len(latencies),
        "inferences": inferences,
        "total_duration_seconds": duration,
        "mean_context_encoding_latency": 1000* sum(context_encoding_latency_list) / len(context_encoding_latency_list),
        "e2e_output_token_throughput": (inferences * (output_length - context_length)) / duration,
        "mean_output_token_latency": mean_e2e_token_gen_latency / (output_length - context_length),
    }

    print(metrics)
    return metrics

def main():

    us = psutil.Process(os.getpid())
    print(us.cmdline())

    parser = argparse.ArgumentParser()
    amp_choices = ['f32', 'f16', 'bf16']
    floatx_floaty_combinations = list(itertools.product(amp_choices, amp_choices))
    for floatx, floaty in floatx_floaty_combinations:
        amp_choices.append(f'{floatx}-u8-{floaty}')
    parser.add_argument('model_type')
    parser.add_argument('--hf_model', default=None)
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
    # compilation configuration
    run_parser.add_argument('--batch_size', type=int, default=1, help="Input batch size")
    run_parser.add_argument('--n_positions', type=int, default=128, help="Input sequence length")
    run_parser.add_argument('--tp_degree', type=int, default=2, help="Number of neuron cores used for tensor parallel")
    run_parser.add_argument('--unroll', type=int, default=None)
    run_parser.add_argument('--device', type=str, default="cpu")
    run_parser.add_argument('--context_length_estimate', type=int, default=None)

    run_parser.add_argument('--gqa', type=str, default=None)
    # simple_sample
    run_parser.add_argument('--simple_sample', action='store_true')
    run_parser.add_argument('--old', action='store_true') # FIXME: debug
    # generation configuration
    # TODO: could we simply make unparsed arguments as generation configuration?
    run_parser.add_argument('--beam', type=int, default=1)
    run_parser.add_argument('--do_sample', action='store_true')
    run_parser.add_argument('--max_length', type=int, default=None)
    run_parser.add_argument('--max_new_tokens', type=int, default=None)
    run_parser.add_argument('--top_k', type=int, default=1)
    run_parser.add_argument('--top_p', type=float, default=1.0)
    run_parser.add_argument('--temperature', type=float, default=1.0)
    run_parser.add_argument('--no_repeat_ngram_size', type=int, default=0)
    run_parser.add_argument('--num_return_sequences', type=int, default=1)
    run_parser.add_argument('--output_scores', action='store_true')
    run_parser.add_argument('--repetition_penalty', type=float, default=1)
    # input prompt
    run_parser.add_argument('--various', action='store_true', help="Generated batched sequence with different length if set; otherwise using same length")
    run_parser.add_argument('--prompt', type=str, default= "Hello, I'm a language model, not a programming language. " \
        "I don't know how to write a program, but I know that I can do it. " \
        "I'm not going to tell you how I learned to code, or how much I've learned. " \
        "But I will tell the story of my first programming experience, and how it changed my life.")
    run_parser.add_argument('--prompt_len', type=int, default=None)
    # neuron_utils utils
    run_parser.add_argument('--snapshot', action='store_true')
    run_parser.add_argument('--pack_artifacts', action='store_true')
    run_parser.add_argument('--to_s3', default=None)
    # dump logits
    run_parser.add_argument('--dump_logits', action='store_true', default=None)
    run_parser.add_argument('--dump_logits_limit', type=int, default=1)
    # benchmark
    run_parser.add_argument('--benchmark', action='store_true', default=None)
    # suffix
    run_parser.add_argument('--suffix', type=str, default=None)
    # profile
    run_parser.add_argument('--profile', action='store_true')

    logits_analysis_name = 'analyze'
    logits_analysis_parser = subparsers.add_parser(logits_analysis_name)
    logits_analysis_parser.set_defaults(which=logits_analysis_name)
    logits_analysis_parser.add_argument('-d','--dirs', nargs='+', required=True)
    logits_analysis_parser.add_argument('--plot', action='store_true')

    visual_name = 'visual'
    visual_parser = subparsers.add_parser(visual_name)
    visual_parser.set_defaults(which=visual_name)
    visual_parser.add_argument('-d','--dirs', nargs='+', required=True)

    args = parser.parse_args()

    model_type = args.model_type
    model_cls = None
    hf_model_name = args.hf_model

    get_hf_model = lambda x: hf_model_name if hf_model_name is not None else x
    if model_type == 'gpt2':
        model_cls = GPT2ForSampling
        hf_model_name = get_hf_model('gpt2')
    elif model_type == "gpt2-ctx":
        model_cls = GPT2ForSamplingWithContextBroadcasting
        hf_model_name = get_hf_model('gpt2')
    elif model_type == 'opt':
        model_cls = OPTForSampling
        hf_model_name = get_hf_model('facebook/opt-125m')
    elif model_type == "gptj":
        model_cls = GPTJForSampling
        hf_model_name = get_hf_model('EleutherAI/gpt-j-6B')
    elif model_type == "llama":
        model_cls = LlamaForSampling
        hf_model_name = get_hf_model('openlm-research/open_llama_3b')
    elif model_type == "bloom":
        model_cls = BloomForSampling
        hf_model_name = get_hf_model('bigscience/bloom-560m')
    elif model_type == "gptneox":
        model_cls = GPTNeoXForSampling
        hf_model_name = get_hf_model('EleutherAI/gpt-neox-20b')
    assert model_cls is not None, f"Invalid model_type: {model_type}"

    print(f"Running demo with model_type:{model_type}, hf_model_name:{hf_model_name}")

    if args.which == save_name:
        save(args, hf_model_name, model_type)
    elif args.which == run_name:
        run(args, hf_model_name, model_cls)
    elif args.which == logits_analysis_name:
        logits_analysis(args, hf_model_name, model_cls)
    elif args.which == visual_name:
        visual(args)

def visual(args):
    logit_paths = args.dirs
    logit_paths = [os.path.join(path, "0.pt") for path in logit_paths]
    golden_path = logit_paths[-1]

    pairs = []
    for path in logit_paths[:-1]:
        logits = torch.load(path)
        print(path, logits)
        pairs.append([logits])


    golden_logits = torch.load(golden_path)

    for pair in pairs:
        pair.append(golden_logits)


    pairs.append([golden_logits, golden_logits])

    import matplotlib.pyplot as plt

    # Create a new figure
    plt.figure()

    # Plot each line from the tensor pairs
    for x_vals, y_vals in pairs:
        plt.plot(x_vals.reshape(-1), y_vals.reshape(-1))

    # Show legend if needed
    labels = [path.split('/')[-2] for path in logit_paths]
    plt.legend(labels)

    plt.savefig('line_plots.png')

    # Show the plot
    plt.show()

def upload_folder_to_s3(local_folder, s3_url):
    import boto3
    from botocore.exceptions import NoCredentialsError
    s3_url_parsed = s3_url.lstrip('s3://').split('/')
    bucket_name = s3_url_parsed[0]
    s3_prefix = '/'.join(s3_url_parsed[1:])

    s3 = boto3.client('s3')
    try:
        for root, dirs, files in os.walk(local_folder):
            for file in files:
                local_file_path = os.path.join(root, file)
                s3_object_key = os.path.join(s3_prefix, os.path.join(local_folder, local_file_path))

                # Upload the file to S3
                s3.upload_file(local_file_path, bucket_name, s3_object_key)

                print(f'Uploaded: {local_file_path} to {s3_object_key} in {bucket_name}')

        print('Upload completed successfully.')

    except NoCredentialsError:
        print('AWS credentials not available. Please configure your credentials.')


def logits_analysis(args, hf_model_name, model_cls):
    logits_dirs = args.dirs

    # load logits
    def load_logits_from_dir(logits_dir):
        filenames = [f for f in os.listdir(logits_dir) if os.path.isfile(os.path.join(logits_dir, f))]
        filenames = sorted(filenames)
        logits_array = [torch.load(os.path.join(logits_dir, f)) for f in filenames]
        return logits_array


    # load logits
    logits_candidates = [load_logits_from_dir(d) for d in logits_dirs]


    logits_golden = logits_candidates[-1]
    golden_dir = logits_dirs[-1]
    logits_candidates = logits_candidates[:-1]
    candidates_dir = logits_dirs[:-1]


    pairs = []
    for i, logits in enumerate(logits_candidates):
        print(f"========================== Start analysis on \n \t{candidates_dir[i]}\n vs\n \t{golden_dir} (golden) \n==================")
        print(f"sentence length: {len(logits)} vs {len(logits_golden)}")
        min_l = min(len(logits), len(logits_golden))
        pairs.append([logits[:min_l], logits_golden[:min_l]])
        for i in range(min_l):

            # assume we always check with greedy
            token_candidate = torch.argmax(logits[i])
            token_golden = torch.argmax(logits_golden[i])

            allclose_passed = torch.allclose(logits[i], logits_golden[i], atol=1.0, rtol=0.001)
            if not allclose_passed:
                logging.warning(f"Failed to match on step {i}, {logits[i]} vs {logits_golden[i]}")

            if token_candidate != token_golden:
                logging.error(f"mismatch happens on {i}, token_candidate: {token_candidate}, token_golden {token_golden}. candidate score ({logits[i][:, token_candidate]} vs {logits[i][:, token_golden]}). golden score({logits_golden[i][:, token_candidate]} vs {logits_golden[i][:, token_golden]})")
                break

    pairs.append([logits_golden, logits_golden])

    if args.plot:
        import matplotlib.pyplot as plt


        # Create a new figure
        plt.figure()

        # Plot each line from the tensor pairs
        for x_vals, y_vals in pairs:
            plt.plot(torch.concatenate(x_vals).reshape(-1), torch.concatenate(y_vals).reshape(-1))

        # Show legend if needed
        labels = [logits_dirs]
        plt.legend(labels)

        plt.savefig('line_plots.png')



def save(args, hf_model_name, model_type):
    model = AutoModelForCausalLM.from_pretrained(hf_model_name, low_cpu_mem_usage=True)
    save_pretrained_split(model, args.save)


def run(args, hf_model_name, model_cls):
    torch.manual_seed(15213)

    full_prompt_text = args.prompt
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
    print(encoded_text, "len:", encoded_text.input_ids.shape[-1])


    forward_func = None
    if args.do_sample:
        compile_batch_size = args.batch_size*args.num_return_sequences*args.beam
    else:
        compile_batch_size = args.batch_size*args.beam

    # wrap whole thing with try and finally as we want to collect artifacts in the end
    try:
        if args.device == "neuron":
            suffix = f"{neuronxcc.__version__}_{model_cls.__name__}_{hf_model_name.replace('/', '_')}_b{compile_batch_size}_np{args.n_positions}_amp{args.amp}_tp{args.tp_degree}_ul{args.unroll}" if args.suffix is None else args.suffix
            dump_path = f"neuronx_dump_{suffix}"
            snapshot_path = f"neuronx_snapshot_{suffix}"
            if args.snapshot or args.pack_artifacts:
                os.environ["NEURONX_DUMP_TO"] = dump_path
                os.environ["HLO_SNAPSHOT_PATH"] = snapshot_path
                print(f"Set snapshot path {snapshot_path}, dump_path: {dump_path}")

            neuron_config = NeuronConfig(group_query_attention=args.gqa)

            print(f'running {model_cls.__name__}.from_pretrained')
            if model_cls == GPT2ForSamplingWithContextBroadcasting or model_cls == LlamaForSampling or model_cls == BloomForSampling or model_cls == OPTForSampling:
                suffix += f"_ctx{args.context_length_estimate}"
                neuron_model = model_cls.from_pretrained(args.load, batch_size=compile_batch_size, amp=args.amp,
                                                tp_degree=args.tp_degree, n_positions=args.n_positions,
                                                unroll=args.unroll, context_length_estimate=args.context_length_estimate, neuron_config=neuron_config)
            else:
                neuron_model = model_cls.from_pretrained(args.load, batch_size=compile_batch_size, amp=args.amp,
                                                tp_degree=args.tp_degree, n_positions=args.n_positions,
                                                unroll=args.unroll)
            print('running model.to_neuron')
            neuron_model.to_neuron()

            config = AutoConfig.from_pretrained(args.load)
            if args.beam > 1:
                print("Setting up reorder_cache for beam operation")
                neuron_model.setup_reorder_cache()
            model = HuggingFaceGenerationModelAdapter(config, neuron_model)
        else:
            suffix = f"{hf_model_name}_{model_cls.__name__}_cpu"
            print(f'running {model_cls.__name__}.from_pretrained')
            model = AutoModelForCausalLM.from_pretrained(hf_model_name, low_cpu_mem_usage=True)


        if args.simple_sample:
            assert args.device == "neuron", "cannot runing simple sample with non-neuron device"
            print("running simple_sample")
            class OutputClassForSimpleSample:
                def __init__(self, sequences, scores=None):
                    self.sequences = sequences
                    self.scores = scores
            with torch.inference_mode():
                max_length = args.max_length if args.max_length is not None else args.n_positions
                forward_func = lambda : neuron_model.sample(encoded_text.input_ids, max_length,
                    top_k=args.top_k)
                sequences = forward_func()

                scores = None
                if args.output_scores or args.dump_logits:
                    sequences, scores = sequences

                outputs = OutputClassForSimpleSample(sequences, scores)
        else:


            generation_config = {
                "num_beams": args.beam,
                "do_sample": args.do_sample,
                "no_repeat_ngram_size": args.no_repeat_ngram_size,
                "top_k": args.top_k,
                "top_p": args.top_p,
                "temperature": args.temperature,
                "num_return_sequences": args.num_return_sequences,
                "return_dict_in_generate": True,
                "output_scores": args.output_scores,
                "repetition_penalty": args.repetition_penalty,
            }

            if args.max_length is None and args.max_new_tokens is None:
                generation_config["max_length"] = args.n_positions
            if args.max_length is not None:
                generation_config["max_length"] = args.max_length
            if args.max_new_tokens is not None:
                generation_config["max_new_tokens"] = args.max_new_tokens


            if args.dump_logits:
                generation_config['output_scores'] = True

            if args.profile:
                neuron_model.profile(f"profile_{suffix}")

            print("running HuggingFace Generation API, generation_config:\n", generation_config)

            with torch.inference_mode():
                forward_func = lambda : model.generate(
                    input_ids=encoded_text.input_ids,
                    attention_mask=encoded_text.attention_mask,
                    **generation_config,
                )
                outputs = forward_func()

        if args.dump_logits:
            dump_logits_dir = f"logits_dump_{suffix}_simple{args.simple_sample}"
            print(f"Dumping logits into {dump_logits_dir}")
            dump_logits_limit = args.dump_logits_limit
            if args.dump_logits_limit == -1: # dump all for analysis
                dump_logits_limit = len(outputs.scores)
            assert dump_logits_limit <= len(outputs.scores), f"dump_logits_limit {args.dump_logits_limit} exceeds length of generated tokens {len(outputs.scores)}"
            for i in range(dump_logits_limit):
                dump(dump_logits_dir, f"{i}.pt", outputs.scores[i])

        print('generated_sequence=', outputs.sequences)
        outputs_sentences = [tokenizer.decode(gen_seq) for gen_seq in outputs.sequences]
        print(outputs_sentences)

    finally:

        if args.pack_artifacts:
            compiler_artifacts_dir = f"compiler_artifacts_{suffix}"
            print(f"Save artifacts at: {compiler_artifacts_dir}")
            os.makedirs(compiler_artifacts_dir, exist_ok=True)
            if os.path.exists(dump_path):
                shutil.copytree(dump_path, os.path.join(compiler_artifacts_dir, dump_path), dirs_exist_ok=True)
            if os.path.exists(snapshot_path):
                shutil.copytree(snapshot_path, os.path.join(compiler_artifacts_dir, snapshot_path), dirs_exist_ok=True)
            with open(os.path.join(compiler_artifacts_dir, "command.txt"), 'w') as f:
                f.write(f"{' '.join(sys.argv)}\n")

            if args.to_s3 is not None:
                upload_folder_to_s3(compiler_artifacts_dir, args.to_s3)

    if args.benchmark:
        benchmark(args, neuron_model, forward_func, output_length=outputs.sequences.shape[-1], context_length=encoded_text.input_ids.shape[-1])

if __name__ == "__main__":
    main()