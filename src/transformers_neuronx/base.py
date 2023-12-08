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
from typing import Optional, Union, List
import hashlib
from transformers_neuronx import bucket
from transformers_neuronx import utils
from transformers_neuronx import module
from transformers_neuronx.constants import LAYOUT_BSH
from concurrent.futures import ProcessPoolExecutor


# Mainly used to expose top level APIs to the model object for serialization
class NeuronModelBase(module.WrappingCheckpointCompatibleModel):
    is_fid = False

    # top level api
    def save(self, directory):
        assert self.serialization_enabled(), 'serialization is not enabled for this model'
        self._save_compiled_artifacts(directory)

    # top level api
    def load(self, directory):
        assert self.serialization_enabled(), 'serialization is not enabled for this model'
        self._load_compiled_artifacts(directory)

    # top level api
    def compile(self, parallel_degree=None):
        kernels = self._get_all_kernels()
        neff_bytes_futures = dict()
        if parallel_degree is None:
            parallel_degree = len(kernels)
        with ProcessPoolExecutor(parallel_degree) as executor:
            for kernel in kernels:
                neff_bytes_futures[hash_hlo(kernel.hlo_module)] = executor.submit(kernel.compile)
            for kernel in kernels:
                kernel.neff_bytes = neff_bytes_futures[hash_hlo(kernel.hlo_module)].result()

    # top level api
    def setup(self):
        for nbs in self.nbs_objs:
            nbs.setup()

    # TODO: decouple hlo_generation from load weights so compile can be called before it
    def to_neuron(self):
        self.load_weights()
        self.compile()
        self.setup()

    # top level api
    def enable_speculative_decoder(self,speculation_length:Optional[Union[List[int], int]]):
        if isinstance(speculation_length, int):
            speculation_length=[speculation_length]
        for k in speculation_length:
            self.decoder_lm_head_for_speculation[k]=self.decoder_param_set.init_speculative_decoder(unroll=self.unroll, buckets=self.token_buckets, model_obj=self, n_active_tokens=k)

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

    def context(self, hidden, cache_ids, start_ids, last_token_id, *rest, neuron_config=None):
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
        batch_size, = start_ids.shape

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
                logits = model(hidden_context, cache_context, start_ids, last_token_id, *rest, neuron_config=neuron_config)

        for i in range(current, context_length):
            cache_ids = torch.as_tensor([i], dtype=torch.int32)
            hidden_slice = hidden[:, i:i+1].contiguous()
            logits = self.decoder_lm_head(hidden_slice, cache_ids, start_ids, last_token_id, *rest, neuron_config=neuron_config)

        if self.is_fid:
            logits[:] = float('-inf')
            logits[self.bos_token_id] = 1.0

        return logits

    def _prepare_for_par_ctx_rhs_padding(self, input_ids):
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
        last_token_id = torch.as_tensor(0, dtype=torch.int32)
        if context_length == 1:
            return input_ids, last_token_id

        # TODO: check context_buckets for compatibility with OPT
        if hasattr(self, "context_buckets"):
            estimate = bucket.find(self.context_buckets, context_length)
        else:
            estimate = self.context_length_estimate

        if estimate:
            # when context length is larger than estimate, last_token_id=estimate-1
            last_token_id = torch.as_tensor(min(context_length - 1, estimate-1), dtype=torch.int32)
            if context_length < estimate:
                input_ids = utils.pad(input_ids, 1, estimate, left=False)

        return input_ids, last_token_id

    def _prepare_for_continuous_batching(self, input_ids, cache_ids=None, seq_ids=None):
        n_seqs, n_active_tokens = input_ids.shape
        continuous_batching = self.neuron_config and self.neuron_config.continuous_batching

        if seq_ids is None or not continuous_batching:
            # static batching
            return input_ids, cache_ids, seq_ids

        # continuous batching
        batch_size = self.neuron_config.continuous_batching.batch_size_for_shared_caches
        if n_active_tokens > 1 and cache_ids.flatten()[0].item() == 0:
            # context encoding
            n_active_seqs, n_active_tokens = input_ids.shape
            n_positions = self.context_buckets[-1]
            assert n_active_seqs == cache_ids.shape[0], f"invalid n_active_seqs ({n_active_seqs} vs {cache_ids.shape[0]})"
            assert n_active_tokens <= n_positions, f"invalid input prompt length ({n_active_tokens} <= {n_positions})"
            cache_ids_pad = torch.zeros(n_active_seqs, n_positions, dtype=cache_ids.dtype, device='cpu')
            for seq_id in range(n_active_seqs):
                cache_ids_pad[seq_id, :n_active_tokens] = cache_ids[seq_id, :n_active_tokens]
            return input_ids, cache_ids_pad, seq_ids

        # token generation
        full_input_ids = torch.zeros(batch_size, 1, dtype=input_ids.dtype, device="cpu")
        full_cache_ids = torch.zeros(batch_size, 1, dtype=cache_ids.dtype, device="cpu")
        for idx, seq_id in enumerate(seq_ids.flatten()):
            seq_id = seq_id.item()
            full_input_ids[seq_id, :] = input_ids[idx, :]
            full_cache_ids[seq_id, :] = cache_ids[idx, :]

        return full_input_ids, full_cache_ids, seq_ids

    def _preprocess(self, input_ids, start_ids=None, cache_ids=None):
        # enable dynamic batch size feature for continuous batching
        input_ids, cache_ids, start_ids = self._prepare_for_continuous_batching(input_ids, cache_ids, start_ids)

        # right pad the input_ids if neccessary
        input_ids, last_token_id = self._prepare_for_par_ctx_rhs_padding(input_ids)

        # note: this context_length is after right padded
        batch_size, context_length = input_ids.shape

        if start_ids is None:
            start_ids = torch.zeros(batch_size, dtype=torch.int32)

        if cache_ids is None:
            cache_ids = torch.arange(context_length, dtype=torch.int32)

        if hasattr(self, "prefixed_length") and self.prefixed_length:
            cache_ids += self.prefixed_length

        return input_ids, cache_ids, start_ids, last_token_id

    def _postprocess(self, logits, start_ids=None):
        if start_ids is None:
            return logits

        running_batch_size, n_embed = logits.shape
        input_batch_size = start_ids.shape[0]
        if running_batch_size == input_batch_size:
            # context encoding (aka prefill)
            # NOTE: logits are returned directly, since dynamic batching is handled in _context_dynamic_batching
            return logits

        # token generation (aka decoding)
        seq_ids = start_ids.flatten().tolist()
        assert input_batch_size == len(seq_ids), f"expected seq_ids to be {input_batch_size} in length, but seq_ids={seq_ids}"
        new_logits = torch.zeros(input_batch_size, n_embed, dtype=logits.dtype, device=logits.device)
        for idx, seq_id in enumerate(seq_ids):
            new_logits[idx, :] = logits[seq_id, :]

        return new_logits

    def _cast_logits(self, logits):
        # Cast logits to float32 or the dtype specified in the neuron config
        logits_dtype = torch.float32
        if self.neuron_config:
            logits_dtype = getattr(torch, self.neuron_config.cast_logits_dtype)
        return logits.to(logits_dtype)

    def _context_dynamic_batching(self, hidden, *args, neuron_config=None):
        # Taking HSB layout
        _, context_length, input_batch_size = hidden.shape
        assert hasattr(self, "context_batch_sizes"), f"{type(self)} doesn't support dynamic batching."

        running_batch_size = self.context_batch_sizes[-1]
        if input_batch_size > running_batch_size:
            assert input_batch_size % running_batch_size == 0, \
                "input batch size ({input_batch_size}) not divisible by running batch size ({running_batch_size})"
            n_iters = input_batch_size // running_batch_size
            all_logits = []
            cache_ids, start_ids = args[0], args[1]
            for iter_id in range(n_iters):
                # Assuming HSB layout
                start_idx = iter_id*running_batch_size
                end_idx = (iter_id+1)*running_batch_size
                hidden_per_batch = hidden[:, :, start_idx:end_idx]
                cache_ids_per_batch = cache_ids[start_idx:end_idx, :]
                start_ids_per_batch = start_ids[start_idx:end_idx]
                last_token_id = cache_ids_per_batch.max()
                logits_per_batch = self.context(hidden_per_batch, cache_ids_per_batch,
                                                start_ids_per_batch, last_token_id, neuron_config=neuron_config)
                all_logits.append(logits_per_batch)
            logits = torch.cat(all_logits, dim=2)
        else:
            assert input_batch_size == running_batch_size, \
                "input batch size ({input_batch_size}) not equal to running batch size ({running_batch_size})"
            logits = self.context(hidden, *args, neuron_config=neuron_config)
        return logits

    def _forward(self, hidden, *args, neuron_config=None):
        # Taking HSB layout
        _, context_length, *_ = hidden.shape
        if not self.neuron_config.on_device_embedding:
            hidden = hidden.transpose(0, -1).contiguous()

        if context_length > 1:
            continuous_batching = self.neuron_config and self.neuron_config.continuous_batching
            if continuous_batching:
                logits = self._context_dynamic_batching(hidden, *args, neuron_config=neuron_config)
            else:
                logits = self.context(hidden, *args, neuron_config=neuron_config)
        else:
            logits = self.decoder_lm_head(hidden, *args, neuron_config=neuron_config)

        logits = self._cast_logits(logits)
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
            res = self.forward(*args, **kwargs)
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
                self.forward(*args, **kwargs)

    def serialization_enabled(self):
        return getattr(self, 'nbs_objs', None) is not None

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
