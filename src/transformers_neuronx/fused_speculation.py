import itertools
from typing import Optional, List
 
import torch
 
from transformers_neuronx import compiler
from transformers_neuronx import utils
from transformers_neuronx import hlo
from transformers_neuronx import decoder
from transformers_neuronx import program
from transformers_neuronx import base
 
 
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
 
        # FIXME: Add more validation to ensure draft/target compatibility
 
        # User-provided attributes
        self.draft = draft
        self.target = target
        self.k = k
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.output_scores = output_scores
 
        # Derived attributes
        self.neuron_config = self.target.neuron_config
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
            draft_tokens = [tokens]
            prior_ids = [cache_ids]
 
            for _ in range(self.k):
                tensors = cache_ids, *rest
                logits, caches = draft._hlo_unroll(tokens, tensors, caches, layers_weights, pre_layer_params, lm_head_params)
                tokens = draft._hlo_generation(logits, generation_params)
 
                # Update cache to next position
                cache_ids = hlo.add(cache_ids, 1)
 
                # Aggregate individual outputs
                draft_logits.append(logits)
                draft_tokens.append(hlo.cast(tokens, s32))
                prior_ids.append(cache_ids)
 
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
            target_ids = target._hlo_generation(target_scores, generation_params)
            target_ids = hlo.cast(target_ids, s32)
 
            target_out_caches = itertools.chain(*caches)
 
            # NOTE: Exclude the input token during the speculative token selection
            draft_ids = hlo.concatenate(draft_tokens[1:], 1)
 
            draft_ids = hlo.transpose(draft_ids, 0, 1)
            target_ids = hlo.transpose(target_ids, 0, 1)
 
            next_tokens, index, *mask = hlo.speculative_token_selection(
                draft_ids, target_ids, draft_scores, target_scores,
                tp_degree=self.tp_degree,
                pad_token_id=self.pad_token_id,
                deterministic=True, output_mask=self.output_scores
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
 
        selector = program.FusedSpeculativeSelector(sizes, self.k)
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
 
    def sample(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        sequence_length: Optional[int] = None,
        streamer: Optional['transformers.generation.streamers.BaseStreamer'] = None,
    ):
        """
        Sample tokens using the fully fused speculative graph.
 
        Args:
            input_ids: The tokenized input identifiers.
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
        # FIXME: Handle attention_mask. Currently, attention_mask is not used.
 
        if sequence_length is None:
            sequence_length = self.max_position - self.k # type: int
        # FIXME: Loosen this restriction to sequence_length <= max_position
        assert sequence_length <= self.max_position - self.k
 
        batch_size, start = input_ids.shape
        if batch_size > 1:
            raise NotImplementedError("Current speculative sampling supported only with batch size = 1.")
 
        # The streamer should send back the input tokens to conform to
        # huggingface behavior.
        # Reference: https://github.com/huggingface/transformers/blob/v4.39.0/src/transformers/generation/utils.py#L1410-L1411
        if streamer:
            streamer.put(input_ids)
 
        # Context encoding
        # FIXME: populate scores with the context encoding logits
        cache_ids = torch.arange(start, dtype=torch.int32)
        if self.target.neuron_config.use_2d_cache_ids:
            cache_ids = cache_ids.unsqueeze(0).expand(batch_size, start)
        token_id = self.target(input_ids, cache_ids=cache_ids)
        if streamer:
            streamer.put(token_id)
        self.draft(input_ids, cache_ids=cache_ids)
 
        # Preallocate state tensors
        sequences = torch.full((batch_size, sequence_length + self.k + 1), self.pad_token_id, dtype=torch.int32)
        cache_ids = torch.full((batch_size, 1), start, dtype=torch.int32)
        positions = torch.arange(start + 1, start + self.k + 2, dtype=torch.int64).repeat((batch_size, 1))
        sequences[:, :start] = input_ids
        sequences[:, start:start + 1] = token_id
 
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
                tokens, counts, token_id, *score = self.speculator.execute(token_id, cache_ids, return_ranks=1)
            else:
                tokens, counts, token_id, *score = self.speculator.execute(token_id, cache_ids.squeeze(1), return_ranks=1)
 
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
            # the sequence_length.
            cache_ids = torch.clamp(cache_ids, max=sequence_length)
            done.logical_or_(torch.eq(cache_ids, sequence_length))
 
            if done.all().item():
                break
 
        if streamer:
            streamer.end()
 
        if self.output_scores:
            return sequences[:, :ends.max().item()], scores[:, start:ends.max().item(), :]
        return sequences[:, :ends.max().item()]
