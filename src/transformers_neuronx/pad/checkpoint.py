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
import json
import torch
import numpy as np
from transformers_neuronx.dtypes import to_torch_dtype
from transformers_neuronx.pad.layernorm_padded_cpu import LayerNormCPU
from transformers_neuronx.gpt2.model import GPT2ForSampling
from transformers_neuronx.opt.model import OPTForSampling
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers_neuronx.gpt2.demo import amp_callback as amp_callback_gpt2
from transformers_neuronx.opt.demo import amp_callback as amp_callback_opt
from transformers import GPT2Config, OPTConfig
from transformers_neuronx.module import save_pretrained_split

"""
TPDegreeCheckpointConverter
===============================
A class to pad a checkpoint to make its number of heads is divisible by a given tp-degree.

Issue:
    Currently, transformers_neuronx has the following limitation: 
        n_head % tp_degree == 0

Solution:
    To overcome this limitation, a feature is added to convert a checkpoint 
    to a new checkpoint with extra n_head and pad the tensors such that
        1) Both checkpoints generate same results
        2) The condition n_head % tp_degree == 0 is satisfied.

Usage:
    GPT2
    ----
    >>> from transformers_neuronx.pad.checkpoint import save_pretrained_split_tp_degree
    >>> model = AutoModelForCausalLM.from_pretrained('gpt2')
    >>> save_pretrained_split_tp_degree(model, "./gpt2_split_tp8", tp_degree=8)
    >>> neuron_model = GPT2ForSampling.from_pretrained("./gpt2_split_tp8", tp_degree=8)
    >>> neuron_model.to_neuron()

    OPT
    ----
    >>> from transformers_neuronx.pad.checkpoint import save_pretrained_split_tp_degree
    >>> model = AutoModelForCausalLM.from_pretrained('facebook/opt-125m')
    >>> save_pretrained_split_tp_degree(model, "./opt_125m_split_tp8", tp_degree=8)
    >>> neuron_model = OPTForSampling.from_pretrained("./opt_125m_split_tp8", tp_degree=8)
    >>> neuron_model.to_neuron()
"""

def rule_copy_to_padded(src_param, tgt_param, src_hid_dim, tgt_hid_dim):
    """ Default rule for padding and conversion of source parameters tensor to target tensor

    Args:
        src_param (torch.tensor): source parameters
        tgt_param (torch.tensor): target parameters
        src_hid_dim (int): hidden dimension of source 
        tgt_hid_dim (int): hidden dimension of target 

    Returns:
        torch.tensor: modified target parameters
    """
    shape_diffs = [tgt_shape // tgt_hid_dim for tgt_shape, src_shape in zip(tgt_param.shape, src_param.shape) if tgt_shape != src_shape]
    if len(shape_diffs) > 0:
        max_n_partitions = max(shape_diffs)
        tgt_param[:] = 0
        for i in range(max_n_partitions):
            src_slices = []
            tgt_slices = []
            for tgt_shape, src_shape in zip(tgt_param.shape, src_param.shape):
                if tgt_shape == src_shape or tgt_shape // tgt_hid_dim == 1:
                    src_slice = slice(0, src_shape)
                    tgt_slice = slice(0, src_shape)
                else:
                    src_slice = slice(i * src_hid_dim, (i + 1) * src_hid_dim)
                    tgt_slice = slice(i * tgt_hid_dim, i * tgt_hid_dim + src_hid_dim)
                src_slices.append(src_slice)
                tgt_slices.append(tgt_slice)
            tgt_param[tgt_slices] = src_param[src_slices]
    else:
        tgt_param[:] = src_param
    return tgt_param

def rule_mlp(src_param, tgt_param, src_hid_dim, tgt_hid_dim):
    """ Specific rule for mlp conversion

    Args:
        src_param (torch.tensor): source parameters
        tgt_param (torch.tensor): target parameters
        src_hid_dim (int): hidden dimension of source 
        tgt_hid_dim (int): hidden dimension of target 

    Returns:
        torch.tensor: modified target parameters
    """
    slices = tuple(slice(0, x) for x in src_param.shape)
    tgt_param[:] = 0
    tgt_param[slices] = src_param
    return tgt_param


def get_number_of_extra_heads(n_head, tp_degree):
    """ Get number of extra heads needed to pad

    Args:
        n_head (int): number of heads in source/initial model
        tp_degree (int): Tensor parallel degree

    Returns:
        int: extra heads
    """
    if n_head % tp_degree == 0:
        extra_heads = 0
    else:
        extra_heads = tp_degree - n_head % tp_degree
    return extra_heads

class TPDegreeCheckpointConverter:
    def __init__(self, model_cpu, save_directory, tp_degree, amp="bf16", check_accuracy=False):
        """ Base class to perform checkpoint conversion

        Args:
            model_cpu (nn.Module): source model which we need to pad
            save_directory (string): folder name to save the final checkpoint
            tp_degree (int): tensor parallel degree
        """
        self.tp_degree = tp_degree
        self.tolerance = 1e-2
        self.amp = amp
        self.amp_torch = to_torch_dtype(amp)
        self.src_config = model_cpu.config
        self.pass_fail_messages = ["Fail", "Pass"]
        self.set_model_specific_props()
        model_cpu.eval()
        src_hid_dim, src_n_head = self.get_config_info(self.src_config)
        extra_heads = get_number_of_extra_heads(src_n_head, tp_degree)
        self.save_directory = save_directory
        self.neuron_folder = self.save_directory

        if os.path.exists(self.neuron_folder):
            print(f"Folder {self.neuron_folder} already exists, passing checkpoint generation!")
            return
        
        if extra_heads == 0:
            save_pretrained_split(model_cpu, self.neuron_folder)
            return
        
        os.makedirs(self.save_directory, exist_ok=True)
        tgt_config = self.get_target_config_file(src_hid_dim, src_n_head)
        tgt_config = self.config(**tgt_config)
        tgt_hid_dim, tgt_n_head = self.get_config_info(tgt_config)
        tgt_config = self.model_based_config_change(tgt_config, src_hid_dim, tgt_hid_dim)
        tgt_model_cpu = AutoModelForCausalLM.from_config(tgt_config)
        tgt_model_cpu.eval()
        rules = self.rules
        tgt_model_cpu = self.transfer_model(model_cpu, tgt_model_cpu, src_hid_dim, tgt_hid_dim, rules)
        self.save_neuron_model(tgt_model_cpu)
        if check_accuracy:
            self.check_accuracy_cpu_model(model_cpu, tgt_model_cpu, src_hid_dim, tgt_hid_dim)
            self.check_accuracy_neuron_model(model_cpu, src_hid_dim, tgt_hid_dim)
    
    def set_model_specific_props(self):
        """ Each model-specific child should implement this and assign model specific variables
        """
        raise NotImplementedError("set_model_specific_props is not implemented!")

    def model_based_config_change(self, config, src_hid_dim, tgt_hid_dim):
        """ Each model-specific child can implement extra modifications to config. 
        If not implemented in child, do nothing.
        """
        return config

    def get_target_config_file(self, src_hid_dim, src_n_head):
        """ Calculate needed extra heads and create a corresponding target config 

        Args:
            src_hid_dim (int): source model hidden dim
            src_n_head (int): source model number of heads

        Returns:
            tuple: config and its path
        """        
        tgt_n_head = src_n_head + get_number_of_extra_heads(src_n_head, self.tp_degree)
        tgt_hid_dim = tgt_n_head * src_hid_dim // src_n_head
        tgt_config = {**self.src_config.__dict__}
        properties = {"hid_dim": tgt_hid_dim, "n_head": tgt_n_head, "tgt_src_ratio": tgt_hid_dim / src_hid_dim}
        tgt_config = self.set_config_info(tgt_config, properties)
        return tgt_config

    def set_config_info(self, config, properties):
        """ Set properties of config

        Args:
            config: input config
            properties (dict): dictionary of each config property and its value

        Returns: config
        """
        for field in properties:
            var_name = self.options.get(field, None)
            if var_name is None:
                raise(KeyError, f"field {var_name} does not exist in config!")
            else:
                config[var_name] = properties[field]  
        return config            

    def get_config_info(self, config, fields=["hid_dim", "n_head"]):
        """ Return config values
        """
        outputs = []
        for field in fields:
            var_name = self.options.get(field, None)
            if var_name is None:
                var = None
            else:
                var = getattr(config, var_name)
            outputs.append(var)        
        return outputs

    def transfer_model(self, src, tgt, src_hid_dim, tgt_hid_dim, pattern_to_rule):
        """ 
        Transfer the model from original cpu model to target cpu model with different number of heads

        Args:
            src (torch model): source model where we copy parameters from
            tgt (torch model): target model where we copy parameters to
            src_hid_dim (int): hidden dimension of source model
            tgt_hid_dim (int): hidden dimension of target model
            pattern_to_rule (dict): dictionary of pattern to rule for specific rules
        """
        self.param_names = param_names = list([x[0] for x in tgt.named_parameters()])
        with torch.no_grad():
            for name, tgt_param, src_param in zip(param_names, tgt.parameters(), src.parameters()):
                # print(name, src_param.shape, "->", tgt_param.shape)
                src_param.to(tgt_param.dtype)

                # Specific rules
                found_rule = False
                for pattern, rule in pattern_to_rule.items():
                    if pattern in name:
                        tgt_param = rule(src_param, tgt_param, src_hid_dim, tgt_hid_dim)
                        found_rule = True
                        break
                
                if found_rule:
                    continue

                # Default rules
                tgt_param = rule_copy_to_padded(src_param, tgt_param, src_hid_dim, tgt_hid_dim)
        return tgt
    
    def check_accuracy_cpu_model(self, src_model_cpu, tgt_model_cpu, src_hid_dim, tgt_hid_dim):
        # Check accuracy of padded model on cpu
        raise NotImplementedError("check_accuracy_cpu_model is not implemented!")
    
    def save_neuron_model(self, model):
        # Save neuron model
        raise NotImplementedError("save_neuron_model is not implemented!")
    
    def check_accuracy_neuron_model(self, src_model_cpu, src_hid_dim, tgt_hid_dim):
        # Check accuracy of padded model on neuron
        raise NotImplementedError("check_accuracy_neuron_model is not implemented!")

class GPT2TPDegreeCheckpointConverter(TPDegreeCheckpointConverter):
    """ 
        GPT2 specific converter        
    """
    def set_model_specific_props(self):
        """ Each model should implement this and assign model specific variables
        """        
        self.config = GPT2Config
        self.rules = {"mlp.c_": rule_mlp}
        self.options = {
            "hid_dim": "n_embd",
            "n_head": "n_head",
            "tgt_src_ratio": "tgt_src_ratio"
        }

    def gpt2_get_input_ids(self):
        """ Get input-ids from the corresponding tokenizer for this model """
        tok_gpt = AutoTokenizer.from_pretrained('gpt2')
        batch_prompts = ["There are three cars with different colors: red, blue, and magenta. There is no yellow or green. What car colors do we have?"] 
        gpt_input_ids = torch.as_tensor([tok_gpt.encode(text) for text in batch_prompts]) 
        return gpt_input_ids

    def gpt2_decode_ids(self, generated_sequences):
        tok_gpt = AutoTokenizer.from_pretrained('gpt2')
        generated_sequences = [tok_gpt.decode(seq) for seq in generated_sequences]    
        return generated_sequences

    def gpt2_modify_layer_norm(self, model, factor):
        for i in range(len(model.transformer.h)):
            model.transformer.h[i].ln_1 = LayerNormCPU(model.transformer.h[i].ln_1, factor)
            model.transformer.h[i].ln_2 = LayerNormCPU(model.transformer.h[i].ln_2, factor)
        model.transformer.ln_f = LayerNormCPU(model.transformer.ln_f, factor)
        return model        

    def check_accuracy_cpu_model(self, src_model_cpu, tgt_model_cpu, src_hid_dim, tgt_hid_dim):        
        ratio = tgt_hid_dim / src_hid_dim        
        if tgt_hid_dim != src_hid_dim:            
            tgt_model_cpu = self.gpt2_modify_layer_norm(tgt_model_cpu, ratio)
        input_ids = self.gpt2_get_input_ids()
        tgt_outs = tgt_model_cpu(input_ids)
        src_outs = src_model_cpu(input_ids)
        accurate = np.allclose(src_outs.logits.detach(), tgt_outs.logits.detach(), rtol=self.tolerance)        
        print(f"[CPU accuracy] Check padded checkpoint with original one (both on cpu): {self.pass_fail_messages[accurate]}")
    
    def save_neuron_model(self, model):
        amp_callback_gpt2(model, self.amp_torch)
        save_pretrained_split(model, self.neuron_folder)
    
    def check_accuracy_neuron_model(self, src_model_cpu, src_hid_dim, tgt_hid_dim):
        sequence_length = 128
        os.environ["n_heads_expansion_ratio"] = str(tgt_hid_dim / src_hid_dim)
        gpt2 = GPT2ForSampling.from_pretrained(self.neuron_folder, tp_degree=self.tp_degree, amp=self.amp, batch_size=1, n_positions=sequence_length)
        gpt2.to_neuron()
        input_ids = self.gpt2_get_input_ids()
        output_ids_neuron = gpt2.sample(input_ids, sequence_length=sequence_length, top_k=1)
        output_ids_cpu = src_model_cpu.generate(input_ids, max_length=128)
        accurate = np.allclose(output_ids_neuron, output_ids_cpu, rtol=self.tolerance)
        print(f"[Neuron accuracy] Check neuron result with cpu result: {self.pass_fail_messages[accurate]}")
        if not accurate:
            print("\ninput_ids\n", input_ids)
            print("\noutput_ids_neuron\n", output_ids_neuron)
            print("\noutput_ids_cpu\n", output_ids_cpu)
            print("\nneuron generation\n", self.gpt2_decode_ids(output_ids_neuron))
            print("\ncpu generation\n", self.gpt2_decode_ids(output_ids_cpu))
        
class OPTTPDegreeCheckpointConverter(TPDegreeCheckpointConverter):
    """ 
        OPT specific converter        
    """    
    def set_model_specific_props(self):
        self.config = OPTConfig
        self.rules = {"fc1": rule_mlp, "fc2": rule_mlp}
        self.options = {
            "hid_dim": "hidden_size",
            "n_head": "num_attention_heads",
            "tgt_src_ratio": "tgt_src_ratio"
        }

    def save_neuron_model(self, model):
        amp_callback_opt(model, self.amp_torch)
        save_pretrained_split(model, self.neuron_folder)

    def model_based_config_change(self, config, src_hid_dim, tgt_hid_dim):
        if hasattr(config, "word_embed_proj_dim"):
            config.word_embed_proj_dim = (config.word_embed_proj_dim * tgt_hid_dim) // src_hid_dim
        return config

def save_pretrained_split_tp_degree(model, save_directory, tp_degree):
    model_type = model.config.model_type
    if model_type == "gpt2":
        GPT2TPDegreeCheckpointConverter(model, save_directory, tp_degree)
    elif model_type == "opt":
        OPTTPDegreeCheckpointConverter(model, save_directory, tp_degree)
    else:
        raise NotImplementedError("Checkpoint padding is implemented only for GPT2 and OPT!")