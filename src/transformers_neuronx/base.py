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
import torch
import logging
import hashlib
import warnings
from typing import Optional, Union, List, Dict
from transformers_neuronx import bucket
from transformers_neuronx import utils
from transformers_neuronx import module
from transformers_neuronx import ops
from transformers_neuronx.compiler import ParallelKernel
from transformers_neuronx.constants import LAYOUT_BSH
from transformers_neuronx.config import GenerationConfig
from transformers_neuronx.util.token_tree import validate_token_tree
from concurrent.futures import ProcessPoolExecutor
import json


# Mainly used to expose top level APIs to the model object for serialization
class NeuronModelBase(module.WrappingCheckpointCompatibleModel):
    is_fid = False

    # top level api
    def save(self, directory, sharded_weights=False):
        assert self.serialization_enabled(), 'serialization is not enabled for this model'
        self._save_compiled_artifacts(directory)
        if sharded_weights:
            assert self.neuron_config.on_device_embedding, "on_device_embedding must be True to save and load presharded weights"
            self._save_presharded_weights(directory)
            self.config.is_presharded_checkpoint = True
            with open(os.path.join(directory, 'config.json'), 'w') as f:
                f.write(json.dumps(self.config.__dict__))

    # top level api
    def load(self, directory):
        assert self.serialization_enabled(), 'serialization is not enabled for this model'
        self._compiled_artifacts_directory = directory

    # top level api
    def compile(self, parallel_degree=None):
        kernels = self._get_all_kernels()
        neff_bytes_futures = dict()
        if parallel_degree is None:
            parallel_degree = len(kernels)
        with ProcessPoolExecutor(parallel_degree) as executor:
            for kernel in kernels:
                neff_bytes_futures[hash_hlo(kernel.hlo_module)] = executor.submit(kernel.compile, kernel.num_exec_repetition)
            for kernel in kernels:
                kernel.neff_bytes = neff_bytes_futures[hash_hlo(kernel.hlo_module)].result()

    # top level api
    def setup(self):
        if self.neuron_config and self.neuron_config.is_pp():
            import torch.distributed as dist
            logging.debug(f"Try to init process group, rank id {self.neuron_config.rank_id}, world size {self.neuron_config.pp_stages}")
            dist.init_process_group("gloo", init_method=f"tcp://{os.getenv('CPU_COMM_ID', '127.0.0.1:9999')}",
                    rank=self.neuron_config.rank_id, world_size=self.neuron_config.pp_stages)
        for nbs in self.nbs_objs:
            nbs.setup()

    # TODO: decouple hlo_generation from load weights so compile can be called before it
    def to_neuron(self):
        if hasattr(self, "_using_presharded_weights"):
            self.load_presharded_weights()
        else:
            self.load_weights()
        if hasattr(self, "_compiled_artifacts_directory"):
            self._load_compiled_artifacts(self._compiled_artifacts_directory)
        else:
            self.compile()
        self.setup()
    
    def load_presharded_weights(self):
        assert self.neuron_config.on_device_embedding, "on_device_embedding must be True to save and load presharded weights"
        ops.init()
        ps_dir = self._using_presharded_weights

        for i in range(self.decoder_lm_head.num_layers):
            new_layer = self.decoder_lm_head.new_layer()
            new_layer.load_presharded_weights(ps_dir)
        
        self.decoder_lm_head.load_presharded_weights(ps_dir)
        self.init_rest_of_model()
    
    def _save_presharded_weights(self, directory):
        self.decoder_lm_head.save_presharded_weights(directory)
        for layer in self.decoder_lm_head.layers:
            layer.save_presharded_weights(directory)

    def enable_token_tree_decoder(self, token_tree: Dict[int, List[int]], batch_sizes: Optional[Union[List[int], int]]=None):
        speculation_length, depth = validate_token_tree(token_tree)
        self.enable_speculative_decoder(speculation_length, batch_sizes, token_tree)

    # top level api
    def enable_speculative_decoder(self, speculation_length: Optional[Union[List[int], int]], batch_sizes: Optional[Union[List[int], int]]=None, token_tree=None):
        if isinstance(speculation_length, int):
            speculation_length = [speculation_length]
        if batch_sizes is None:
            batch_sizes = self.decoder_param_set.batch_size
        if isinstance(batch_sizes, int):
            batch_sizes = [batch_sizes]
        for k in speculation_length:
            for batch_size in batch_sizes:
                self.decoder_lm_head_for_speculation[k, batch_size] = \
                    self.decoder_param_set.init_speculative_decoder(unroll=self.unroll, buckets=self.token_buckets, model_obj=self, n_active_tokens=k, batch_size=batch_size, token_tree=token_tree)

    def enable_window_context_decoder(self, window_context_length:Optional[Union[List[int], int]], unroll: Optional[int] = None):
        if isinstance(window_context_length, int):
            window_context_length=[window_context_length]
        self.window_context_buckets = bucket.context_sizes(window_context_length, self.token_buckets)
        if unroll is None:
            unroll = self.decoder_param_set.num_layers
        for k in self.window_context_buckets:
            self.decoder_lm_head_for_window_context[k]=self.decoder_param_set.init_window_context_decoder(unroll=unroll, buckets=self.token_buckets, model_obj=self, n_active_tokens=k)

    def is_compiled(self):
        # First check if the kernels have neffs already
        try:
            if all([kernel.neff_bytes is not None for kernel in self._get_all_kernels()]):
                return True
        # AttributeError means kernels don't even exist yet.
        except AttributeError:
            pass
        return False

    def reorder_cache(self, reorder_ids):
        self.decoder_lm_head.program.reorder_cache(reorder_ids)

    def setup_reorder_cache(self):
        if self.decoder_lm_head.program is not None: # called after to_neuron
            self.decoder_lm_head.program.setup_reorder_cache()
        else:
            self.decoder_lm_head.need_reorder_cache = True

    def _save_compiled_artifacts(self, directory):
        if os.path.isfile(directory):
            raise FileExistsError(
                f'Artifacts should be saved to a directory. '
                f'Found existing file: {directory}'
            )
        os.makedirs(directory, exist_ok=True)
        for i, nbs_obj in enumerate(self.nbs_objs):
            nbs_obj.save_compiler_artifacts(directory)

    def _load_compiled_artifacts(self, directory):
        if not os.path.isdir(directory):
            raise FileNotFoundError(f'Did not find directory: {directory}.')

        for nbs_obj in self.nbs_objs:
            nbs_obj.set_neff_bytes(directory)

    def _get_all_kernels(self):
        all_kernels = []
        for nbs in self.nbs_objs:
            for kernel in nbs.get_all_kernels():
                all_kernels.append(kernel)
        return all_kernels


    # To enable serialization, have the model call this
    # function to register all nbs_obj of your model.
    # The nbs_obj must follow 2 rules:
    #   1. The nbs_obj must inherit from NeuronBaseSerializer.
    #   2. Since this class shouldn't be used directly, a nbs_obj.get_all_kernels()
    #      method should be implemented by the child class, which returns a
    #      list of all kernels which have NEFFs.
    def register_for_serialization(self, nbs_obj):
        # check that requirement 1 and 2 are met.
        assert issubclass(type(nbs_obj), NeuronBaseSerializer), 'The nbs_obj must inheret from NeuronBaseSerializer.'
        assert getattr(nbs_obj, 'get_all_kernels', None) is not None, 'An nbs_obj.get_all_kernels() method should be implemented.'
        temp = getattr(self, 'nbs_objs', [])
        nbs_obj.compiler_artifacts_path = None
        temp.append(nbs_obj)
        self.nbs_objs = temp

    def reset(self):
        self.decoder_lm_head.reset()

    def decode(self, hidden, *args):
        """A helper function for decoding (token-generation)
        This function simply called decoder_lm_head. 
        It is defined as a function to enable wrapping with a decorator and measuring its runtime.
        """ 
        return self.decoder_lm_head(hidden, *args)

    def context(self, hidden, cache_ids, start_ids, last_token_id, *rest):
        """A helper to process context (prompt)
        1) if there is available context encoding model (infered from self.context_buckets)
            - when context_length >= estimate, slice the context up to estimate,
                and call context encoding model
            - when context_length < estimate, skip and fall back to serial token generation model

            and mark `current` accrodingly

        2) process the left over tokens accroding to `current`
            - if there is no context encoding model, simply do serial token generation for context

        Other arguments that are required by the model are contained in `rest`.
        """
        context_length = hidden.shape[1]
        batch_size = 1 if self.neuron_config.use_1d_query else start_ids.shape[0]

        all_logits = [] # Collect all logits if neuron_config.output_all_logits is True

        if self.is_fid:
            # Fusion-In-Decoder context encoding
            fused_context_length = hidden.shape[1]
            context_length = fused_context_length // self.batch_size

        current = 0

        estimate = bucket.find(self.context_buckets, context_length)

        if estimate is not None:
            hidden_context = hidden
            cache_context = cache_ids

            # Slice context that when it is too large
            if context_length > estimate:
                current = estimate
                hidden_context = hidden[:, :estimate]
                cache_context = cache_ids[:estimate]

            # Cannot use context encoding for a context that is too small. This
            # is because the caller must be aware of the cache-ids/start-ids
            # used.
            elif context_length < estimate:
                raise ValueError(f"context_length ({context_length}) shouldn't be smaller than estimate ({estimate})")

            # Directly pass input to the context network when exactly sized
            else:
                current = estimate

            if current == estimate:
                model = self.decoder_lm_head_for_context[estimate, batch_size]
                if self.neuron_config.log_softmax_scores:
                    logits, scores = model(hidden_context, cache_context, start_ids, last_token_id, *rest)
                else:
                    logits = model(hidden_context, cache_context, start_ids, last_token_id, *rest)
                if self.neuron_config.output_all_logits:
                    all_logits.append(logits[:, :last_token_id + 1, :])

        # process the leftovers context
        while current < context_length:
            # find the optimal "window"
            estimate = None
            if hasattr(self, "window_context_buckets"):
                estimate = bucket.find(self.window_context_buckets, context_length - current)

            # when the leftovers is smaller than estimate, fall back to single token generation
            # TODO: can we pad?
            if estimate is None or context_length - current < estimate:
                for i in range(current, context_length):
                    cache_ids = torch.as_tensor([i], dtype=torch.int32)
                    hidden_slice = hidden[:, i:i+1].contiguous()
                    logits = self.decoder_lm_head(hidden_slice, cache_ids, start_ids, last_token_id, *rest)
                    if self.neuron_config.output_all_logits:
                        all_logits.append(logits)
                break

            hidden_slice = hidden[:, current:current+estimate].contiguous()
            cache_ids = torch.as_tensor([i for i in range(current, current+estimate)], dtype=torch.int32)
            last_token_id = torch.as_tensor([estimate - 1])
            if self.neuron_config.log_softmax_scores:
                logits, scores = self.decoder_lm_head_for_window_context[estimate](hidden_slice, cache_ids, start_ids, last_token_id, *rest)
            else:
                logits = self.decoder_lm_head_for_window_context[estimate](hidden_slice, cache_ids, start_ids, last_token_id, *rest)
            if self.neuron_config.output_all_logits:
                all_logits.append(logits)

            current += estimate

        if all_logits:
            logits = torch.cat(all_logits, dim=1)

        if self.is_fid:
            logits[:] = float('-inf')
            logits[self.bos_token_id] = 1.0

        if self.neuron_config.log_softmax_scores:
            return logits, scores
        return logits

    def _prepare_for_par_ctx_rhs_padding(self, input_ids, cache_ids, start_ids=None, **kwargs):
        """A helper to do rhs padding on prompt for parallel context encoding model
        i.e.
            input_ids = [[111, 222, 333]]
            context_length = 3

            if context bucket size is 4
            we will pad input_ids to [[111, 222, 333, 0]]

            last_token_id = 2 (used for generation to mark the last token is at index 2 instead 3)

        Note:
            - there is no change on start_ids with right padding.
            - cache_ids will be set to [0, 1, 2, 3] in self.forward()
        """
        batch_size, context_length = input_ids.shape

        # if last_token_id not used, simply set to 0
        if self.neuron_config.vectorize_last_token_id:
            last_token_id = torch.zeros(batch_size, dtype=torch.int32)
        else:
            last_token_id = torch.as_tensor([0], dtype=torch.int32)
        if context_length == 1:
            # token generation
            if self.neuron_config.paged_attention:
                max_num_seqs = self.neuron_config.continuous_batching.max_num_seqs
                max_model_len = self.neuron_config.continuous_batching.max_model_len
                block_size = self.neuron_config.continuous_batching.block_size
                max_num_blocks_per_seq = (max_model_len + block_size - 1) // block_size

                input_metadata = kwargs.get("input_metadata")
                last_token_id = input_metadata.block_tables
                last_token_id = utils.pad(last_token_id, 0, max_num_seqs, left=False)
                last_token_id = utils.pad(last_token_id, 1, max_num_blocks_per_seq, left=False)
            return input_ids, cache_ids, last_token_id

        # TODO: check context_buckets for compatibility with OPT
        if cache_ids is not None and cache_ids.flatten()[0].item() > 0:
            # speculative forward: n_active_tokens > 1 and cache_ids start from position > 0
            speculation_buckets = list(set([k for k, batch_size in self.decoder_lm_head_for_speculation.keys()]))
            estimate = bucket.find(speculation_buckets, context_length)
        elif hasattr(self, "context_buckets"):
            estimate = bucket.find(self.context_buckets, context_length)
        else:
            estimate = self.context_length_estimate

        if estimate:
            # when context length is larger than estimate, last_token_id=estimate-1
            if self.neuron_config.vectorize_last_token_id:
                if self.neuron_config.use_1d_query:
                    max_num_seqs = self.neuron_config.continuous_batching.batch_size_for_shared_caches
                    if self.neuron_config.paged_attention:
                        # For context encoding phase of paged attention, we get prompt_lens from input_metadata.
                        input_metadata = kwargs.get("input_metadata")
                        prompt_lens = input_metadata.prompt_lens_tensor
                        context_length = prompt_lens.sum().item()

                        # last_token_id
                        # Note: last_token_id would be used in two places for paged attention support
                        # 1) build block diagonal causal mask
                        # 2) dynamic slice logits for concatenated prompt encoding
                        # It should be safe to pad zeros, for both use cases.
                        max_num_seqs = self.neuron_config.continuous_batching.max_num_seqs
                        last_token_id = utils.pad(prompt_lens, 0, max_num_seqs, left=False)
                    else:
                        prompt_lens = cache_ids.max(dim=1).values + 1
                        context_length = prompt_lens.sum().item()

                        # input_ids and cache_ids
                        new_input_ids = torch.tensor([], dtype=input_ids.dtype)
                        new_cache_ids = torch.tensor([], dtype=cache_ids.dtype)
                        for idx, prompt_len in enumerate(prompt_lens):
                            new_input_ids = torch.concat([new_input_ids, input_ids[idx, :prompt_len]])
                            new_cache_ids = torch.concat([new_cache_ids, cache_ids[idx, :prompt_len]])
                        input_ids = new_input_ids.unsqueeze(0)
                        cache_ids = new_cache_ids.unsqueeze(0)

                        # last_token_id
                        # Note: With 1D query, last_token_id actually takes prompt lengths as input,
                        #       and it's converted from prompt_lens to actual last_token_id in HLO.
                        last_token_id = prompt_lens
                        last_token_id_pad = torch.zeros(max_num_seqs, dtype=last_token_id.dtype)
                        last_token_id_pad[start_ids] = last_token_id
                        last_token_id = last_token_id_pad
                else:
                    last_token_id = cache_ids.max(dim=1).values
            else:
                last_token_id = torch.as_tensor([min(context_length - 1, estimate-1)], dtype=torch.int32)
            if context_length < estimate:
                input_ids = utils.pad(input_ids, 1, estimate, left=False)
                cache_ids = self._pad_cache_ids(cache_ids, batch_size, context_length, estimate)

        return input_ids, cache_ids, last_token_id

    def _pad_cache_ids(self, cache_ids, batch_size, context_length, estimate):
        if self.neuron_config.use_2d_cache_ids:
            if self.neuron_config.use_1d_query:
                assert (cache_ids.ndim == 2) and (cache_ids.shape[0] == 1), \
                    f"cache_ids is expected to be a 1xN matrix, but its shape is {cache_ids.shape}"
                start_idx = cache_ids[0, -1].item() + 1
                end_idx = estimate + start_idx - context_length
                pad_elements = torch.arange(start_idx, end_idx, dtype=torch.long).unsqueeze(0)
                cache_ids_pad = torch.concat([cache_ids, pad_elements], dim=1)
                cache_ids = torch.minimum(cache_ids_pad, torch.tensor(estimate-1, dtype=torch.long))
            else:
                # TODO: fix cache_ids padding for batch speculative decoding
                # for now, use cache_ids without change for speculative_forward
                is_speculative_forward = cache_ids.flatten()[0].item() > 0
                if is_speculative_forward:
                    return cache_ids
                cache_ids = torch.arange(estimate, dtype=torch.int32)
                cache_ids = cache_ids.unsqueeze(0).expand(batch_size, estimate)
        else:
            if cache_ids is None:
                cache_ids = torch.arange(estimate, dtype=torch.int32)
            else:
                # Inputs: cache_ids = [16, 17], estimate = 512
                #
                # Process:
                # start_idx = 18, end_idx = 528 (= 512+16)
                # padded_elements =       [18, 19, ..., 511, 512, 513, ..., 525, 526, 527]
                # cache_ids_pad = [16, 17, 18, 19, ..., 511, 512, 513, ..., 525, 526, 527]
                # cache_ids =     [16, 17, 18, 19, ..., 511, 511, 511, ..., 511, 511, 511]
                start_idx = cache_ids[-1].item() + 1
                end_idx = estimate + start_idx - context_length
                pad_elements = torch.arange(start_idx, end_idx, dtype=torch.int32)
                cache_ids_pad = torch.concat([cache_ids, pad_elements], dim=0)
                cache_ids = torch.minimum(cache_ids_pad, torch.tensor(estimate-1, dtype=torch.int32))
        return cache_ids

    def _prepare_for_continuous_batching(self, input_ids, cache_ids=None, seq_ids=None):
        n_seqs, n_active_tokens = input_ids.shape
        continuous_batching = self.neuron_config and self.neuron_config.continuous_batching

        if seq_ids is None or not continuous_batching:
            # static batching
            return input_ids, cache_ids, seq_ids

        batch_size = self.neuron_config.continuous_batching.batch_size_for_shared_caches

        if n_active_tokens > 1 and cache_ids.flatten()[0].item() == 0:
            # context encoding
            n_active_seqs, n_active_tokens = input_ids.shape
            continuous_batching_n_positions = bucket.find(self.context_buckets, n_active_tokens)
            assert n_active_seqs == cache_ids.shape[0], f"invalid n_active_seqs ({n_active_seqs} vs {cache_ids.shape[0]})"
            assert n_active_tokens <= continuous_batching_n_positions, \
                f"invalid input prompt length ({n_active_tokens} <= {continuous_batching_n_positions})"
            cache_ids_pad = torch.zeros(n_active_seqs, continuous_batching_n_positions, dtype=cache_ids.dtype, device='cpu')
            for seq_id in range(n_active_seqs):
                cache_ids_pad[seq_id, :n_active_tokens] = cache_ids[seq_id, :n_active_tokens]
            if self.neuron_config.use_1d_query:
                # For concatenated prompt encoding, we rely on slot_mapping for KV cache placement.
                if self.neuron_config.paged_attention:
                    n_active_tokens = len(seq_ids)
                    assert input_ids.shape[-1] == n_active_tokens, \
                        f"slot_mapping length ({n_active_tokens}) is expected to match length of input_ids ({input_ids.shape[-1]})."
                    continuous_batching_n_positions = bucket.find(self.context_buckets, n_active_tokens)
                    seq_ids = utils.pad(seq_ids, 0, continuous_batching_n_positions, left=False)
                else:
                    prompt_lens = cache_ids_pad.max(dim=1).values + 1
                    new_seq_ids = torch.tensor([], dtype=seq_ids.dtype)
                    for idx, (prompt_len, seq_id) in enumerate(zip(prompt_lens, seq_ids)):
                        offset = continuous_batching_n_positions * seq_id
                        new_seq_ids = torch.concat([new_seq_ids, cache_ids[idx, :prompt_len] + offset], dim=0)
                    n_active_tokens = len(new_seq_ids)
                    continuous_batching_n_positions = bucket.find(self.context_buckets, n_active_tokens)
                    assert continuous_batching_n_positions >= n_active_tokens, \
                        f"n_active_tokens ({n_active_tokens}) is expected to be less than n_positions " \
                        f"({continuous_batching_n_positions}) for concatenated prompt encoding"

                    # Pad seq_ids to context bucket size
                    start_idx = new_seq_ids[-1].item() + 1
                    end_idx = (continuous_batching_n_positions - n_active_tokens) + start_idx
                    seq_ids = torch.concat([new_seq_ids, torch.arange(start_idx, end_idx)])
            return input_ids, cache_ids_pad, seq_ids

        elif n_active_tokens > 1 and cache_ids.flatten()[0].item() > 0:
            # speculative forward
            n_active_seqs, n_active_tokens = input_ids.shape
            speculative_n_positions = bucket.find(self.context_buckets, n_active_tokens)
            assert n_active_tokens <= speculative_n_positions, \
                f"invalid input prompt length ({n_active_tokens} <= {speculative_n_positions})"
            prompt_buckets = list(set([k for k, batch_size in self.decoder_lm_head_for_speculation.keys()]))
            speculation_bucket = bucket.find(prompt_buckets, n_active_tokens)
            # validate the speculative head was compiled for the given batch size
            speculation_batches = [batch_size for (k, batch_size) in self.decoder_lm_head_for_speculation.keys()]
            assert n_active_seqs in speculation_batches, \
                    f"invalid batch size for speculative forward ({n_active_seqs} not in {speculation_batches})"
            # make cache ids 2d if needed and pad to match speculation bucket
            if len(cache_ids.shape) == 1:
                cache_ids = cache_ids.unsqueeze(0)
            assert cache_ids.shape[0] == n_active_seqs, \
                    f"invalid n_active_seqs ({n_active_seqs} vs {cache_ids.shape[0]}) in speculative forward"
            # pad cache IDs with max(n_positions) - 1
            # unlike context encoding, padding with 0
            # during speculative_forward will contaminate kv-cache history
            cache_ids_pad = torch.full((n_active_seqs, speculation_bucket), max(self.context_buckets) - 1, dtype=cache_ids.dtype, device="cpu")
            for seq_id in range(n_active_seqs):
                cache_ids_pad[seq_id, :n_active_tokens] = cache_ids[seq_id, :n_active_tokens]
            return input_ids, cache_ids_pad, seq_ids

        # token generation
        if self.neuron_config.paged_attention:
            # For decoding with multiple KV cache blocks:
            # - cache_ids are used as context_lens
            # - start_ids are used as slot_mapping
            # - last_token_id is used as block_tables
            full_input_ids = utils.pad(input_ids, 0, batch_size, left=False)
            full_cache_ids = utils.pad(cache_ids, 0, batch_size, left=False)
            full_seq_ids = utils.pad(seq_ids, 0, batch_size, left=False)
            return full_input_ids, full_cache_ids, full_seq_ids

        # token generation - padding for naive continuous batching
        full_input_ids = torch.zeros(batch_size, 1, dtype=input_ids.dtype)
        full_cache_ids = torch.zeros(batch_size, 1, dtype=input_ids.dtype)
        full_seq_ids = torch.arange(batch_size, dtype=torch.int32)

        # vLLM v0.3.3 used to pass 1d seq_ids but starting with
        # v0.4.0 that is no longer the case. To ensure consistent behaviour
        # across versions, we flatten them before unsqueezing them.
        seq_ids_int64 = seq_ids.flatten().unsqueeze(-1).to(torch.int64)
        full_input_ids.scatter_(dim=0, index=seq_ids_int64, src=input_ids)
        full_cache_ids.scatter_(dim=0, index=seq_ids_int64, src=cache_ids)

        return full_input_ids, full_cache_ids, full_seq_ids

    def _preprocess(self, input_ids, start_ids=None, cache_ids=None, **kwargs):
        # enable dynamic batch size feature for continuous batching
        input_ids, cache_ids, new_start_ids = self._prepare_for_continuous_batching(input_ids, cache_ids, start_ids)

        # right pad the input_ids if neccessary
        input_ids, cache_ids, last_token_id = self._prepare_for_par_ctx_rhs_padding(input_ids, cache_ids, start_ids, **kwargs)
        start_ids = new_start_ids

        # note: this context_length is after right padded
        batch_size, context_length = input_ids.shape

        if start_ids is None:
            start_ids = torch.zeros(batch_size, dtype=torch.int32)

        if cache_ids is None:
            cache_ids = torch.arange(context_length, dtype=torch.int32)
            if self.neuron_config.use_2d_cache_ids:
                cache_ids = cache_ids.unsqueeze(0).expand(batch_size, context_length)

        if hasattr(self, "prefixed_length") and self.prefixed_length:
            cache_ids += self.prefixed_length

        return input_ids, cache_ids, start_ids, last_token_id

    def _postprocess(self, logits, start_ids=None, **kwargs):

        if start_ids is None or (self.neuron_config.output_all_logits and logits.shape[1] > 1):
            return logits

        if self.neuron_config.paged_attention:
            input_metadata = kwargs.get("input_metadata")
            is_prompt = input_metadata.is_prompt
            if is_prompt:
                num_prefills = input_metadata.num_prefills
                return logits[:num_prefills, :]
            else:
                context_lens = input_metadata.context_lens
                return logits[:len(context_lens), :]

        running_batch_size, n_embed = logits.shape
        input_batch_size = start_ids.shape[0]
        if running_batch_size == input_batch_size:
            # context encoding (aka prefill)
            # NOTE: logits are returned directly, since dynamic batching is handled in _context_dynamic_batching
            return logits

        # token generation (aka decoding)
        seq_ids = start_ids.flatten()
        if torch.equal(seq_ids, torch.arange(input_batch_size)):
            logits = logits[:input_batch_size]
        else:
            logits = logits[seq_ids.to(torch.long)]

        return logits

    def _cast_logits(self, logits):
        # Cast logits to float32 or the dtype specified in the neuron config
        logits_dtype = torch.float32
        if self.neuron_config:
            if self.neuron_config.cast_logits_dtype is not None:
                logits_dtype = getattr(torch, self.neuron_config.cast_logits_dtype)
        return logits.to(logits_dtype)

    def _context_dynamic_batching(self, hidden, *args):
        is_bsh = self.neuron_config and self.neuron_config.attention_layout == LAYOUT_BSH
        input_batch_size = hidden.shape[0] if is_bsh or self.neuron_config.on_device_embedding else hidden.shape[2]
        assert hasattr(self, "context_batch_sizes"), f"{type(self)} doesn't support dynamic batching."

        running_batch_size = self.context_batch_sizes[-1]
        if input_batch_size > running_batch_size:
            assert input_batch_size % running_batch_size == 0, \
                "input batch size ({input_batch_size}) not divisible by running batch size ({running_batch_size})"
            n_iters = input_batch_size // running_batch_size
            all_logits = []
            cache_ids, start_ids, last_token_id = args[0], args[1], args[2]
            for iter_id in range(n_iters):
                start_idx = iter_id*running_batch_size
                end_idx = (iter_id+1)*running_batch_size
                if is_bsh or self.neuron_config.on_device_embedding:
                    hidden_per_batch = hidden[start_idx:end_idx, ...]
                else:
                    hidden_per_batch = hidden[..., start_idx:end_idx]
                cache_ids_per_batch = cache_ids[start_idx:end_idx, :]
                start_ids_per_batch = start_ids[start_idx:end_idx]
                last_token_id_per_batch = last_token_id[start_idx:end_idx]
                logits_per_batch = self.context(hidden_per_batch, cache_ids_per_batch,
                                                start_ids_per_batch, last_token_id_per_batch)
                all_logits.append(logits_per_batch)
            if self.neuron_config.on_device_generation:
                logits = torch.cat(all_logits, dim=0)
            else:
                logits = torch.cat(all_logits, dim=-1)
        else:
            assert input_batch_size == running_batch_size, \
                "input batch size ({input_batch_size}) not equal to running batch size ({running_batch_size})"
            logits = self.context(hidden, *args)
        return logits

    def _forward(self, hidden, *args):
        _, context_length, *_ = hidden.shape

        if context_length > 1:
            continuous_batching = self.neuron_config and self.neuron_config.continuous_batching
            if continuous_batching:
                logits = self._context_dynamic_batching(hidden, *args)
            else:
                logits = self.context(hidden, *args)
        else:
            logits = self.decode(hidden, *args)

        if self.neuron_config.on_device_generation:
            return logits

        logits = self._cast_logits(logits)
        if self.neuron_config.output_all_logits and context_length > 1:
            logits = logits.permute(2, 1, 0)
        else:
            logits = logits[:self.config.vocab_size, -1, :]
            logits = logits.transpose(0, 1)
        return logits


    def pp_forward(self, *args, **kwargs):
        """
        forward wrapper for pipeline parallel
        """
        import torch.distributed as dist
        # if host, run normal forward
        if self.neuron_config.rank_id == 0:
            broad_cast_objects = [args, kwargs]
            dist.broadcast_object_list(broad_cast_objects, src=0, device=torch.device("cpu"))
            res = self(*args, **kwargs)
            return res
        else:
            # if non-host, fall back to a for loop
            # all it does is to receive hidden from previous pp stage
            # if it is last pp stage, send back logits to first pp stage
            # to continue sampling loop
            while True:
                broad_cast_objects = [None, None]
                # we need to receive information from host
                # to determine which decoder to run and which NEFF of decoder to run
                # i.e. run token generation with some batch_size and bucket_id
                #
                # it is now naturally handled in forward call
                dist.broadcast_object_list(broad_cast_objects, src=0, device=torch.device("cpu"))
                args, kwargs = broad_cast_objects
                self(*args, **kwargs)

    def serialization_enabled(self):
        return getattr(self, 'nbs_objs', None) is not None

    def profile(self, profile_dir, ntff_count_limit):
        kernels = self._get_all_kernels()

        for kernel in kernels:
            if isinstance(kernel, ParallelKernel):
                kernel.profile(profile_dir, ntff_count_limit)

    def update_generation_config(self, generation_config: GenerationConfig):
        self.decoder_lm_head.update_generation_config(generation_config)


# Base class for all "Serializable Objects"
class NeuronBaseSerializer:

    def save_compiler_artifacts(self, path):
        for kernel in self.get_all_kernels():
            hlo_hash = hash_hlo(kernel.hlo_module)
            with open(os.path.join(path, hlo_hash), 'wb') as f:
                assert kernel.neff_bytes is not None, "cannot save a model which has not been successfully compiled"
                f.write(kernel.neff_bytes)

    def set_neff_bytes(self, directory):
        for kernel in self.get_all_kernels():
            hlo_hash = hash_hlo(kernel.hlo_module)
            try:
                with open(os.path.join(directory, hlo_hash), 'rb') as f:
                    kernel.neff_bytes = f.read()
            except FileNotFoundError:
                raise FileNotFoundError(('Could not find a matching NEFF for your HLO in this directory. '
                                          'Ensure that the model you are trying to load is the same type and '
                                          'has the same parameters as the one you saved or call "save" on '
                                          'this model to reserialize it.'))

    def get_all_kernels(self):
        raise NotImplementedError(
            f'Class {type(self)} deriving from NeuronBaseSerializer must implement get_all_kernels'
        )

def hash_hlo(hlo_module):
    hash_gen = hashlib.sha256()
    message = hlo_module.SerializeToString()
    hash_gen.update(message)
    hash = str(hash_gen.hexdigest())[:20]
    return hash + '.neff'

