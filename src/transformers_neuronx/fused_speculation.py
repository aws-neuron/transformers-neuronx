import itertools
from typing import Optional, List
 
import torch
 
from transformers_neuronx import compiler
from transformers_neuronx import utils
from transformers_neuronx import hlo
from transformers_neuronx import decoder
from transformers_neuronx import program
from transformers_neuronx import base
from transformers_neuronx.config import GenerationConfig
 
 
class FusedSpeculativeDecoder(torch.nn.Module):
    """
    A speculative decoder which fuses the compute of the draft & target models.
 
    This is based on the original DeepMind paper.
    Reference: https://arxiv.org/pdf/2302.01318.pdf
 
    Compared with the regular `SpeculativeGenerator`, the purpose of this
    implementation is to avoid the typical overheads associated with CPU
    sampling. This can be especially impactful when using very fast draft
    models.
 
    Unlike the CPU sampling implementation, this version of the speculative
    decoder *always* executes k + 1 draft iterations in order to populate
    the draft model KV cache invariant to the number of token rejections. This
    means that this implementation may perform worse compared to sampling
    performed on CPU if the `draft` model is relatively slow and if the `k` value
    is small. This is because the CPU implementation has the ability to skip
    the final draft execution when there is at least 1 rejection.
 
    Arguments:
        draft: A fast and less accurate model to perform `k` speculations with.
        target: A slower and more accurate model which consumes speculated
            tokens.
        k: The number of tokens to speculate with the `draft` model.
        eos_token_id: The identifier for the end of sentence token. When
            provided, early stopping will be enabled. Otherwise the sample loop
            will continue until the maximum sequence length is reached.
        pad_token_id: The identifier which is used to pad batches of uneven
            sequence lengths. This token should be excluded from the
            generation results. This should be explicitly specified when
            using a `streamer`.
        buckets: An optional number of buckets to compile the fused speculative
            model for. By default, the number of buckets used will be derived
            from the `target` model.
        output_scores: Flag that indicates whether to construct the fused model
            so that it will return the target model scores during sampling.
        deterministic_threshold: Flag that allows this value to be used as the token
            acceptance threshold instead of a random uniform. This is used for
            debug/testing.
    """
 
    def __init__(
            self,
            draft: base.NeuronModelBase,
            target: base.NeuronModelBase,
            k: int = 2,
            pad_token_id: int = 0,
            eos_token_id: Optional[int] = None,
            buckets: Optional[List[int]] = None,
            output_scores: Optional[bool] = False,
            deterministic_threshold: Optional[float] = None,
        ) -> None:
        super().__init__()
 
        assert draft.neuron_config.on_device_embedding == True, (
            "The draft model must enable on-device embedding."
        )
        assert draft.neuron_config.on_device_generation is not None, (
            "The draft model must enable on-device sampling."
        )
        assert target.neuron_config.on_device_embedding == True, (
            "The target model must enable on-device embedding."
        )
        assert target.neuron_config.on_device_generation is not None, (
            "The target model must enable on-device sampling."
        )
        assert target.config.vocab_size == draft.config.vocab_size, (
            "The target model and draft model must have the same vocab size."
        )
        # TODO: See if there is a way we can enable different tp degrees
        assert target.decoder_lm_head.tp_degree == draft.decoder_lm_head.tp_degree, (
            "The target model and draft model must have the same tp degree."
        )
        assert isinstance(k, int), (
            "The k value must be an integer."
        )
        if draft.batch_sizes != [1]:
            assert draft.neuron_config.padding_side == "right", (
                "The draft model must set padding_side as right for batch size > 1."
            )
        if target.batch_sizes != [1]:
            assert target.neuron_config.padding_side == "right", (
                "The target model must set padding_side as right for batch size > 1."
            )

        # FIXME: Add more validation to ensure draft/target compatibility
 
        # User-provided attributes
        self.draft = draft
        self.target = target
        self.k = k
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.output_scores = output_scores
        self.deterministic_threshold = deterministic_threshold

        # Derived attributes
        self.neuron_config = self.target.neuron_config
        self.neuron_config.is_sequence_parallel = False
        self.tp_degree = self.target.decoder_lm_head.tp_degree
        if buckets is None:
            buckets = self.target.decoder_lm_head.n_positions_list
        self.buckets = buckets
        self.batch_sizes = self.target.batch_sizes
        self.max_position = buckets[-1]
        self.vocab_size = self.target.config.vocab_size
 
        # Internal attributes
        self.speculator = None
 
    def hlo(self, batch_size=1, n_positions=128):
 
        draft = self.draft.decoder_lm_head
        target = self.target.decoder_lm_head
 
        draft.builder.n_positions = n_positions
        target.builder.n_positions = n_positions
        draft.builder.n_active_tokens = 1
        target.builder.n_active_tokens = self.k + 1
        draft.builder.neuron_config.is_sequence_parallel = False
        target.builder.neuron_config.is_sequence_parallel = False
 
        num_inputs = 0
        num_outputs = 0
 
        def speculator(scribe):
            amp, *_ = utils.parse_amp(draft.amp)
            dtype = getattr(scribe, amp)
            s32 = scribe.s32
 
            # Allocate global inputs
            (tokens, cache_ids, *rest), inputs_sdim = draft.inputs_builder(
                scribe, dtype, 1, batch_size
            )
            nonlocal num_inputs
            num_inputs = len(inputs_sdim)
 
            # Allocate Parameters for weight/cache tensors
            param_builder = decoder.DecoderParameterBuilder(scribe, num_inputs)
            in_caches, layers_weights, pre_layer_params, lm_head_params, generation_params = draft._hlo_parameters(n_positions, batch_size, param_builder)
            caches = in_caches
 
            # Create lists to aggregate outputs (used for token acceptance)
            draft_logits = list()
            draft_probs = list()
            draft_indices = list()
            draft_tokens = [tokens]
            prior_ids = [cache_ids]
 
            for _ in range(self.k):
                tensors = cache_ids, *rest
                logits, caches = draft._hlo_unroll(tokens, tensors, caches, layers_weights, pre_layer_params, lm_head_params)
                tokens, probs, indices = draft._hlo_generation(logits, generation_params, return_probs=True)
 
                # Update cache to next position
                cache_ids = hlo.add(cache_ids, 1)
 
                # Aggregate individual outputs
                draft_logits.append(logits)
                draft_probs.append(probs)
                draft_indices.append(indices)
                draft_tokens.append(hlo.cast(tokens, s32))
                prior_ids.append(cache_ids)

            draft_probs = hlo.concatenate(draft_probs, 1)
            draft_indices = hlo.concatenate(draft_indices, 1)
 
            # Execute final iteration in case all tokens are accepted.
            tensors = cache_ids, *rest
            _, caches = draft._hlo_unroll(tokens, tensors, caches, layers_weights, pre_layer_params, lm_head_params)
            draft._hlo_cache_aliases(in_caches, caches)
 
            draft_out_caches = itertools.chain(*caches)
 
            # Concatenate all draft outputs
            tokens = hlo.concatenate(draft_tokens, 1)
            draft_scores = hlo.concatenate(draft_logits, 1)
            if target.neuron_config.padding_side == 'right':
                # Use 2D cache ID; currently we only support BSH
                cache_ids = hlo.concatenate(prior_ids, 1)
            else:
                cache_ids = hlo.concatenate(prior_ids, 0)
 
            # Execute target model
            in_caches, layers_weights, pre_layer_params, lm_head_params, generation_params = target._hlo_parameters(n_positions, batch_size, param_builder)
            tensors = cache_ids, *rest
            target_scores, caches = target._hlo_unroll(tokens, tensors, in_caches, layers_weights, pre_layer_params, lm_head_params)
            target._hlo_cache_aliases(in_caches, caches)
            target_out_caches = itertools.chain(*caches)

            # Adjust the target probabilities in case of rejection
            target_probs, target_indices = target._hlo_generation(target_scores, generation_params, early_return=True)

            adjusted_target_probs = hlo.speculative_adjust_distribution(
                    draft_probs,
                    draft_indices,
                    target_probs,
                    target_indices,
                    self.k)

            generation_config = target.neuron_config.on_device_generation
            target_ids = hlo.multinomial(adjusted_target_probs, dim=2, deterministic=generation_config.deterministic)
            target_ids = hlo.gather(target_indices, 2, target_ids)
            target_ids = hlo.squeeze(target_ids, 2)
            target_ids = hlo.cast(target_ids, target_ids.scribe.s32)

            # NOTE: Exclude the input token during the speculative token selection
            draft_ids = hlo.concatenate(draft_tokens[1:], 1)
            
            sliced_target_indices = hlo.slice_along(target_indices, 1, limit=self.k)
            sliced_target_probs = hlo.slice_along(target_probs, 1, limit=self.k) 

            next_tokens, index, *mask = hlo.speculative_token_selection(
                draft_ids, target_ids, 
                draft_indices, draft_probs, 
                sliced_target_indices, sliced_target_probs,
                tp_degree=self.tp_degree,
                pad_token_id=self.pad_token_id,
                deterministic_threshold=self.deterministic_threshold, output_mask=self.output_scores
            )

            index = hlo.reshape(index, (batch_size, 1))
            next_token_id = hlo.gather(next_tokens, 1, index)

            # Retrieve the target model output scores for the selected tokens
            target_output_scores = None
            if self.output_scores:
                mask, = mask
                target_scores = hlo.all_gather(target_scores, dim=0, tp_degree=self.tp_degree) # Collect scores from all ranks
                target_scores = hlo.permute(target_scores, (2, 1, 0)) # (vocab, k + 1, batch_size) -> (batch_size, k + 1, vocab)
                mask = hlo.broadcast(mask, target_scores.sizes, [0, 1]) # (batch_size, k + 1) -> (batch_size, k + 1, vocab)
                target_output_scores = hlo.masked_select(mask, target_scores, float('-inf'))
 
            # Determine counts of tokens per batch line
            counts = hlo.add(index, 1)
 
            # Format output
            outputs = [
                next_tokens,
                counts,
                next_token_id,
            ]
            if self.output_scores:
                outputs.append(target_output_scores)
            nonlocal num_outputs
            num_outputs = len(outputs)
            outputs = outputs + [*draft_out_caches, *target_out_caches]
            outputs = [out for out in outputs if out is not None]
            root_shapes = [shape.dtype[shape.sizes] for shape in outputs]
            return scribe.tuple(*root_shapes).Tuple(*outputs)
 
        hlo_module = compiler.compile_py_func(speculator)
        return (
            hlo_module,
            num_inputs,
            num_outputs,
        )
 
    def to_neuron(self, workers=None):

        sizes = list(itertools.product(self.batch_sizes, self.buckets))
 
        hlo_modules = list()
        for (batch_size, sequence_length) in sizes:
            hlo_module, num_inputs, num_outputs = self.hlo(batch_size, sequence_length)
            hlo_modules.append(hlo_module)
        kv_replication = 0
        if self.neuron_config.shard_over_sequence:
            assert hasattr(self.target.decoder_lm_head, 'kv_replication'), "attribute kv_replication need to be set in decoder"
            target_replication = self.target.decoder_lm_head.kv_replication
            draft_replication = self.draft.decoder_lm_head.kv_replication \
                if hasattr(self.draft.decoder_lm_head, 'kv_replication') else 0
            kv_replication = max(draft_replication, target_replication)
        selector = program.FusedSpeculativeSelector(sizes, self.k, kv_shard=kv_replication)
        self.speculator = program.BucketedParallelProgram(
            hlo_modules,
            selector=selector,
            num_inputs=num_inputs,
            num_outputs=num_outputs,
            neuron_config=self.neuron_config,
            tp_degree=self.tp_degree,
            tags=[
                f'fused-speculator-seqlen{sequence_length}-batch{batch_size}'
                for batch_size, sequence_length in sizes
            ],
        )
        draft_params = self.draft.decoder_lm_head.valid_parameters(sequence_length, batch_size)
        target_params = self.target.decoder_lm_head.valid_parameters(sequence_length, batch_size)
        self.speculator.build(workers)
        self.speculator.setup([*draft_params, *target_params])


    def update_generation_config(self, generation_config: GenerationConfig):
        self.draft.update_generation_config(generation_config)
        self.target.update_generation_config(generation_config)


    def sample(
        self,
        input_ids: torch.Tensor,
        start_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        sequence_length: Optional[int] = None,
        streamer: Optional['transformers.generation.streamers.BaseStreamer'] = None,
    ):
        """
        Sample tokens using the fully fused speculative graph.
 
        Args:
            input_ids: The tokenized input identifiers.
            start_ids: The offset from the beginning of each input in a batch.
                For batched speculation, this is sequence ids.
            attention_mask: A mask which indicates which input tokens
                should be attended to.
            sequence_length: The total length of inputs + outputs.
            streamer: A streamer callback object for generated tokens. During
                execution, the sampling loop will stream full batches to the
                `streamer` object with padding included. The value will be the
                `pad_token_id` provided at construction. The streamer should
                handle special tokens to eliminate these identifiers.
 
        Returns:
            tokens: The generated sequences.
            scores: (optional) The target model scores if output_scores is set.

        Examples:
            draft = NeuronAutoModelForCausalLM.from_pretrained(...)
            target = NeuronAutoModelForCausalLM.from_pretrained(...)

            draft.to_neuron()
            target.to_neuron()

            fsd = FusedSpeculativeDecoder(draft, target, 5)
            fsd.to_neuron()

            fsd.sample(input_ids, sequence_length=256)
        """

        if sequence_length is None:
            sequence_length = self.max_position - self.k # type: int
        # FIXME: Loosen this restriction to sequence_length <= max_position
        assert sequence_length <= self.max_position - self.k
 
        batch_size, start = input_ids.shape
        if start_ids is None:
            start_ids = torch.arange(batch_size)

        if batch_size>1:
            assert attention_mask is not None, (
                "The attention mask needs to be provided for speculation where batch_size>1"
            )
            cache_ids = torch.arange(start).reshape(1, start).expand(batch_size, start).mul(attention_mask)
        else:
            cache_ids = torch.arange(start, dtype=torch.int32)
            if self.target.neuron_config.use_2d_cache_ids:
                cache_ids = cache_ids.unsqueeze(0).expand(batch_size, start)
 
        # The streamer should send back the input tokens to conform to
        # huggingface behavior.
        # Reference: https://github.com/huggingface/transformers/blob/v4.39.0/src/transformers/generation/utils.py#L1410-L1411
        if streamer:
            streamer.put(input_ids)
 
        # Context encoding
        # FIXME: populate scores with the context encoding logits
        token_id = self.target(input_ids, cache_ids=cache_ids, start_ids=start_ids)
        if streamer:
            streamer.put(token_id)
        self.draft(input_ids, cache_ids=cache_ids, start_ids=start_ids)
 
        # Preallocate state tensors
        sequences = torch.full((batch_size, sequence_length + self.k + 1), self.pad_token_id, dtype=torch.int32)
        sequences[:, :start] = input_ids
        if batch_size > 1:
            cache_ids = torch.count_nonzero(attention_mask,dim=1).view(-1, 1)
            positions = torch.count_nonzero(attention_mask,dim=1).view(-1, 1) + torch.arange(self.k + 1, dtype=torch.int64).repeat((batch_size, 1))
            sequences.scatter_(1, torch.count_nonzero(attention_mask,dim=1).view(-1, 1), token_id)
            positions = positions + 1
        else:
            cache_ids = torch.full((batch_size, 1), start, dtype=torch.int32)
            positions = torch.arange(start + 1, start + self.k + 2, dtype=torch.int64).repeat((batch_size, 1))
            sequences.scatter_(1, torch.full((batch_size, 1), start, dtype=torch.int64), token_id)
 
        if self.output_scores:
            # We cut off the first `start` tokens when returning scores
            scores = torch.full((list(sequences.shape) + [self.vocab_size]), self.pad_token_id, dtype=torch.float32)
 
        # A tensor which keeps track of which sequences are done
        done = torch.full((batch_size, 1), False)
 
        # A tensor which keeps track of the sequence ends
        ends = torch.full((batch_size,), sequence_length, dtype=torch.int32)
 
        # Tensors to track local batch positions/masking for early stop
        batch_positions = torch.arange(self.k + 1).unsqueeze(0)
        batch_mask = torch.full((batch_size, self.k + 1), False)
 
        # Minor optimization: Convert token types to tensors to avoid casting
        eos_token = None
        if self.eos_token_id is not None:
            eos_token = torch.tensor(self.eos_token_id, dtype=torch.int32)
        sequence_length = torch.tensor(sequence_length, dtype=torch.int32)
 
        while True:
            if self.target.neuron_config.use_2d_cache_ids:
                tokens, counts, token_id, *score = self.speculator.execute(token_id, cache_ids, start_ids, return_ranks=1)
            else:
                tokens, counts, token_id, *score = self.speculator.execute(token_id, cache_ids.squeeze(1), start_ids, return_ranks=1)
 
            if eos_token is not None:
 
                # Do a minimal check for the eos_token to keep the happy-path fast
                finished = torch.nonzero(torch.eq(tokens, eos_token))
                if finished.numel():
                    sequence, position = finished[:, 0], finished[:, 1]
                    ends[sequence] = cache_ids[sequence] + position.int() + 2
                    done[sequence] = True
                    batch_mask[sequence] = torch.greater(batch_positions, position.view(-1, 1))
 
                # Always fill tokens after the eos_token with the pad_token.
                # This needs to be done prior to the streamer call to avoid
                # streaming back outputs beyond the stop token.
                tokens.masked_fill_(batch_mask, self.pad_token_id)
 
                # If a stop token was just found, set future batches to be padded
                if finished.numel():
                    batch_mask.logical_or_(done)
 
            if streamer:
                streamer.put(tokens)
 
            sequences.scatter_(1, positions, tokens)
            if self.output_scores:
                score, = score
                # TODO: Look into the perf benefit of performing concats instead
                scores_positions = torch.broadcast_to(torch.unsqueeze(positions, -1), list(positions.shape) + [self.vocab_size])
                scores.scatter_(1, scores_positions, score)
            positions += counts
            cache_ids += counts
 
            # Clamp the cache_ids to the sequence length so that any batch lines
            # that have not reached the end can continue. For batch lines that
            # are complete this populates garbage data into KV cache tail beyond
            # the sequence_length. Clamp position ids as well to populate garbage
            # tokens beyond sequence_length.
            cache_ids = torch.clamp(cache_ids, max=sequence_length)
            positions = torch.clamp(positions, max=sequence_length)
            done.logical_or_(torch.eq(cache_ids, sequence_length))
 
            if done.all().item():
                break
 
        if streamer:
            streamer.end()
 
        if self.output_scores:
            return sequences[:, :ends.max().item()], scores[:, start:ends.max().item(), :]
        return sequences[:, :ends.max().item()]
    # add this to the base CR


    def speculative_iteration(
            self,
            input_ids: torch.Tensor,
            cache_ids: torch.Tensor,
            start_ids: torch.Tensor,
    ) -> (torch.Tensor, torch.Tensor):

        """
        fused speculative iteration for continous batching.
        Note that this method only do one speculative iteration.
        user need to define the sampling loop for the whole sequence generation.

        Args:
            input_ids:
                The input token ids passed to the model to generate
                next predicted tokens (sequence_length - len(input_ids)).
            cache_ids:
                The positions in the KV cache that should be updated.
                shape=(batch, seqlen) for continous batching
            start_ids:
                The offset from the beginning of each input in a batch.

        Returns:
            tokens (list of [tensor of shape (accepted_token_length)]):
        """

        batch_size, context_len = input_ids.shape
        cache_ids = self.prepare_cache_ids(cache_ids, batch_size, context_len)
        min_values_per_row, _ = torch.min(cache_ids, dim=-1)

        # in continuous batching, we need to identify new request and do context decoding
        do_context_decoding = (min_values_per_row == 0).any()
        if do_context_decoding:
            target_next_id = self.target(input_ids, cache_ids, start_ids).reshape(batch_size, -1)
            self.draft(input_ids, cache_ids, start_ids)
            return target_next_id, torch.tensor([[1]] * batch_size)

        seq_ids = start_ids.flatten().to(torch.long)
        # TODO: enable multiple bucket in batch dimension
        graph_batch_size = self.batch_sizes[0]

        full_input_ids, cache_ids_pad, seq_ids_pad = [
            self.preprocess_input(tensor=t, seq_ids=seq_ids, target_batch_size=graph_batch_size)
            for t in [input_ids, cache_ids, seq_ids] 
        ]

        seq_ids_pad = seq_ids_pad.view(-1)
        tokens, counts, token_id, *score = self.speculator.execute(full_input_ids, cache_ids_pad, seq_ids_pad,
                                                                   return_ranks=1)
        output_tokens, output_counts = [
            self.postprocess_output(tensor=t, seq_ids=seq_ids)
            for t in [tokens, counts]
        ]

        return output_tokens, output_counts

    
    def preprocess_input(self, tensor: torch.Tensor, seq_ids: torch.Tensor, target_batch_size: int):
        """
        Sorts tensor in batch dimensions based on sequence ids.
        Zero pads tensor in batch dimensions when running batch size is smaller than target batch size.

        Example:
            tensor [[1],[2],[3]], seq_ids: [0, 1, 2] target_batch_size:3 -> [[1], [2], [3]]
            tensor [[1],[2],[3]], seq_ids: [2, 0, 1] target_batch_size:3 -> [[2], [3], [1]]
            tensor [[2],[3]], seq_ids: [2, 1] target_batch_size:3 -> [[0], [3], [2]]

        Args:
            tensor:
                The input tensor to be processed. first dimension is batch. dim_size >=1
            seq_ids:
                The positions in the KV cache that should be returned. Expected to be 1-dimensional 
                torch.LongTensor.
             target_batch_size:
                Batch size the full input tensor should have
        """
        original_shape = tensor.shape
        new_shape = (target_batch_size,) + original_shape[1:]
        output_tensor = torch.zeros(new_shape, dtype=tensor.dtype)
        output_tensor[seq_ids] = tensor
        return output_tensor
    

    def postprocess_output(self, tensor: torch.Tensor, seq_ids: torch.LongTensor):
        """
        Removes padding from output tensor by selecting only the sequences passed in original seq_ids. 
        Returns output in order of seq_ids.

        Example:
            tensor [[1],[2],[3]], seq_ids: [0, 1, 2] -> [[1], [2], [3]]
            tensor [[1],[2],[3]], seq_ids: [1, 0] -> [[2], [1]]

         Args:
             tensor:
                The output tensor to be processed. first dimension is batch. dim_size >=1
             seq_ids:
                The positions in the KV cache that should be returned. Expected to be 1-dimensional 
                torch.LongTensor.

        """

        return tensor[seq_ids]


    def prepare_cache_ids(self, cache_ids: torch.Tensor, batch: int, seq_len: int) -> List[torch.Tensor]:
            """
            Args:
                cache_ids:
                    The positions in the KV cache that should be updated.
                batch:
                    The batch size
                seq_len:
                    sequence length

            Returns:
                cache_ids of shape=(batch, seqlen) for continous batching or (seq_len,) for non-continous batching
            """
            if cache_ids is not None:
                return cache_ids
            if self.target.neuron_config and self.target.neuron_config.use_2d_cache_ids:
                return torch.tensor([[j for j in range(seq_len)] for i in range(batch)])
            else:
                return torch.tensor([i for i in range(seq_len)])
