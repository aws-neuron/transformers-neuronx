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

import os
import shutil
import json
import torch
import math
import argparse
import time


HELP_DOC_STR = """
    This is a ckpt converter to generate random weight ckpt for any N layer based on existed single layer ckpt.

    It utilizes the `empty` save and loading by only preserve the shape info of weights.
    For 400B Llama (around 100 layers)
        - Disk occupation: 13MB
        - convert time: 60s

    Usage of ckpt_converter:

    step1: generate a single layer ckpt and save it

        model = AutoModelForCausalLM.from_config(single_layer_config)
        save_pretrained_split(model, single_layer_directory)

    step2: use the converter to convert ckpt to arbiratry layer

        converter single_layer_directory n_layer_directory n

    Note: only validated for Llama architecture
"""

def convert(input_dir, output_dir, num_layers, init_std=1.0):
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(input_dir, "config.json")) as f:
        config_json = json.load(f)

    config_json["num_hidden_layers"] = num_layers

    with open(os.path.join(output_dir, "config.json"), 'w') as f:
        json.dump(config_json, f, indent=2)

    shutil.copy(os.path.join(input_dir, "generation_config.json"), os.path.join(output_dir, "generation_config.json"))


    src_bin_dir = os.path.join(input_dir, "pytorch_model.bin")

    dst_bin_dir = os.path.join(output_dir, "pytorch_model.bin")

    os.makedirs(dst_bin_dir, exist_ok=True)

    with open(os.path.join(src_bin_dir, "key_to_filename.json")) as f:
        src_key_to_filename_json = json.load(f)


    non_layer_weights = {}
    per_layer_weights = {}

    all_shape_infos = {}

    for k, v in src_key_to_filename_json.items():
        src_file = os.path.join(src_bin_dir, v)

        w = torch.load(src_file)

        all_shape_infos[k] = w.shape

        if k.startswith("model.layers"):
            per_layer_weights[k] = v
        else:
            non_layer_weights[k] = v

    print(non_layer_weights, per_layer_weights)

    param_counter = 0

    dst_key_to_filename_json = {}


    total_param_bytes = 0

    def add_param(key, src_key):
        nonlocal param_counter
        nonlocal total_param_bytes

        dst_file = f"p{param_counter}.{key}.empty_json"
        dst_key_to_filename_json[key] = dst_file


        tensor_shape = all_shape_infos[src_key]
        empty_json = {
            'torch_dtype': "float32",
            'shape': tensor_shape,
            'init_std': init_std,
        }

        with open(os.path.join(dst_bin_dir, dst_file), 'w') as f:
            json.dump(empty_json, f, indent=2)

        param_counter += 1

        total_param_bytes += math.prod(tensor_shape)

    add_param("model.embed_tokens.weight", "model.embed_tokens.weight")


    for i in range(num_layers):
        for w_k, w_f in per_layer_weights.items():
            suffix = w_k.replace("model.layers.0.", "")
            k = f"model.layers.{i}.{suffix}"
            add_param(k, w_k)


    add_param("model.norm.weight", "model.norm.weight")
    add_param("lm_head.weight", "lm_head.weight")

    with open(os.path.join(dst_bin_dir, "key_to_filename.json"), 'w') as f:
        json.dump(dst_key_to_filename_json, f, indent=2)


    print(f"total params: {total_param_bytes} ({total_param_bytes / (10**9)}B), {total_param_bytes*2 / (10**9)} GB (for fp16)", )


def ckpt_converter():

    parser = argparse.ArgumentParser(description=HELP_DOC_STR,
                                     formatter_class= argparse.RawTextHelpFormatter)
    parser.add_argument('input_dir', type=str)
    parser.add_argument('output_dir', type=str)
    parser.add_argument('num_layers', type=int)
    parser.add_argument('--init_std', type=float, default=1.0)


    args = parser.parse_args()

    start = time.time()
    convert(args.input_dir, args.output_dir, args.num_layers, args.init_std)

    print(f"convert done after {time.time() - start}s")


if __name__ == "__main__":
    ckpt_converter()
