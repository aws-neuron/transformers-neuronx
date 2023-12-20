import torch
import transformers
from typing import Optional, Tuple, Union

from transformers_neuronx import base
from transformers_neuronx import sampling


class TokenAcceptor:
    """
    Abstract base class for token acceptor that is used in Speculative Sampling loop,
    to check which draft tokens should be accepted.

    Note-1: The length of target scores will be one greater than the length of draft scores since the target model
    will additionally generate a next score for the final draft token.
    Note-2: Batch size > 1 is not yet supported for Speculative Decoding.
    """

    def __call__(
        self,
        draft_ids: torch.Tensor,
        draft_scores: torch.Tensor,
        target_scores: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            draft_input_ids (`torch.Tensor` of shape `(speculated_token_length)`):
                Tokens ids from the draft model.

            draft_scores (`torch.Tensor` of shape `(speculated_token_length, vocab)`):
                Prediction scores of the draft model.

            target_scores (`torch.Tensor` of shape `(speculated_token_length + 1, vocab)`):
                Token scores from the target model.

        Return:
            accepted_token_ids (`torch.Tensor` of shape `(accepted_speculated_token)`):
                The accepted draft predicted token ids. Length of accepted_speculated_tokens <= speculated_token_length.

        """
        raise NotImplementedError(f"{self.__class__} is an abstract class. Only"
                                  f" classes inheriting this class can be called.")


class DraftProvider:
    """
    Abstract base class for Draft provider to speculate `k` tokens,
    where (k+1) is the number of parallel tokens target model can decode in one forward pass.

    Note: k value is configured at the target model compilation time.
    """

    def __call__(
            self,
            input_ids: torch.Tensor,
            k: int,
            cache_ids: Optional[torch.Tensor] = None,
            start_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input_ids (`torch.Tensor` of shape `(batch_size, sequence_length)`):
                Either context input ids (prompt) or the next input id from where it needs to start generating tokens.

            k (int):
                The number of speculative tokens to generate.

            cache_ids (`torch.Tensor` of shape `(sequence_length)`):
                The positions in the KV cache that should be updated. KV Cache positions start from 0.

                For example: If your input ids are torch.tensor([108 120 130 140]) and predicted next_token_id is 150,
                to generate a next token, cache_ids is set as 4 and input_ids set to torch.tensor([150]).

                The behavior of the model should be chosen based on the `cache_ids`:
                1. `cache_ids == None`:       [Draft] Context encoding
                2. `cache_ids.shape[1] == 1`: [Draft] Speculated K tokens

            start_ids (`torch.Tensor` of shape `(batch)`):
                The offset from the beginning of each input in a batch.

        Returns:
            tokens (`torch.Tensor` of shape `(batch_size, k)`):
                The next token prediction(s) where k = number of speculated tokens.

            scores (`torch.Tensor` of shape `(batch_size, k, config.vocab)`):
                 The next token score(s)`.

            Note: The returned values should be chosen based on the `cache_ids`:
             1. `cache_ids == None`:       [Draft] 1 Token & Score
             2. `cache_ids.shape[1] == 1`: [Draft] `k` Tokens & Scores

        """
        raise NotImplementedError(f"{self.__class__} is an abstract class. Only"
                                  f" classes inheriting this class can be called.")


class ReferenceTokenAcceptor(TokenAcceptor):
    """
    Reference Implementation of TokenAcceptor defined as per the DeepMind paper: https://arxiv.org/pdf/2302.01318.pdf
    """
    def __call__(
            self,
            draft_ids: torch.Tensor,
            draft_scores: torch.Tensor,
            target_scores: torch.Tensor,
    ) -> torch.Tensor:
        draft_token_len, draft_vocab = draft_scores.shape
        target_token_len, target_vocab = target_scores.shape
        assert draft_vocab == target_vocab  # vocab size should be same
        assert draft_token_len+1 == target_token_len  # target should includes additional token predicted

        accepted_tokens = torch.as_tensor([], dtype=torch.int32)
        accepted_token_count = 0
        all_accepted = True

        k = draft_token_len  # number of speculated tokens
        draft_probabilities = torch.softmax(draft_scores, dim=-1)
        target_probabilities = torch.softmax(target_scores, dim=-1)

        draft_ids = draft_ids[0]

        for i in range(k):
            draft_token_id = draft_ids[i]

            target_prob_i = target_probabilities[i][draft_token_id]
            draft_prob_i = draft_probabilities[i][draft_token_id]

            # Accepted the token if:
            # case-1: Target probability >= Draft probability
            # case-2: Random sampling where sampled value < (Target/Draft probability).
            random = torch.rand((1,)).item()
            if random < min(1, target_prob_i / draft_prob_i):
                # Accepted
                accepted_tokens = torch.cat([accepted_tokens, torch.as_tensor([draft_token_id], dtype=torch.int32)])
                accepted_token_count += 1

            else:  # Rejection
                # Get the non-overlap target probabilities from draft,
                # as overlap tokens are already validated as part of acceptance logic.
                # So, sample only from non-overlap distribution from target probability space.
                prob_diff = target_probabilities[i] - draft_probabilities[i]
                prob_diff = torch.where(prob_diff > 0, prob_diff, 0.)
                accepted_tokens = torch.cat([accepted_tokens, torch.as_tensor([torch.multinomial(prob_diff, num_samples=1, replacement=True)])])
                accepted_token_count += 1
                all_accepted = False
                break

        # Step 4: if all draft tokens were accepted, sample a final token
        if all_accepted:
            accepted_tokens = torch.cat([accepted_tokens, torch.as_tensor([torch.multinomial(target_probabilities[-1], num_samples=1, replacement=True)])])
            accepted_token_count += 1

        return accepted_tokens.view(1, -1)


class DefaultTokenAcceptor(TokenAcceptor):
    """
    Optimized TokenAcceptor based on original DeepMind paper: https://arxiv.org/pdf/2302.01318.pdf
    """
    def __call__(
            self,
            draft_ids: torch.Tensor,
            draft_scores: torch.Tensor,
            target_scores: torch.Tensor,
    ) -> torch.Tensor:
        draft_token_len, draft_vocab = draft_scores.shape
        target_token_len, target_vocab = target_scores.shape
        assert draft_vocab == target_vocab  # vocab size should be same
        assert draft_token_len + 1 == target_token_len  # target should includes additional token predicted

        draft_probabilities = torch.softmax(draft_scores, dim=-1)
        target_probabilities = torch.softmax(target_scores, dim=-1)
        index = draft_ids.view(-1, 1)
        target_probs = torch.gather(target_probabilities[:-1], 1, index)
        draft_probs = torch.gather(draft_probabilities, 1, index)

        random = torch.rand(draft_probs.shape)
        ratio = torch.clamp(target_probs / draft_probs, max=1.0)
        accepted = torch.less(random, ratio)

        # Minimum will return the first occurance of 0 or False (i.e. rejection)
        minimum = torch.min(accepted.view(torch.uint8), dim=0)
        value = minimum.values.item()
        index = minimum.indices.item()

        def sample(probs):
            return torch.multinomial(probs, num_samples=1, replacement=True)

        if value != 0: # If we didn't get a rejection this means all drafts were accepted
            next_token = sample(target_probabilities[-1:])
            return torch.cat((draft_ids, next_token), dim=1)
        else:
            prob_diff = target_probabilities[index:index + 1] - draft_probabilities[index: index + 1]
            prob_diff = torch.clamp(prob_diff, min=0.0)
            next_token = sample(prob_diff)
            return torch.cat((draft_ids[:, :index], next_token), dim=1)


class DraftModelForSpeculation(DraftProvider):
    """
    Standard Implementation of Draft model provider that auto-regressively speculates k tokens.
    """

    def __init__(self, model, **kwargs) -> None:
        self.model = model
        self.kwargs = kwargs

    def _context_block(self, input_ids, start_ids):
        """
        Run context encoding network of the given model.

        Args:
            input_ids: The initial input tokens passed to the model
            start_ids: The offset from the beginning of each input in a batch.

        Returns:
            token: predicted next token
            score: predicted token score
        """
        next_token_scores = self.model(input_ids, None, start_ids)
        inputs = torch.argmax(next_token_scores, dim=1, keepdim=True)
        return inputs, next_token_scores

    def __call__(
            self,
            input_ids: torch.Tensor,
            k: int,
            cache_ids: Optional[torch.Tensor] = None,
            start_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform standard auto-regressive token generation using the draft model, to speculate k-tokens.

        Args:
            input_ids: Either context, next token, or draft tokens. shape=(batch, seqlen)
            k: The number of speculative tokens
            cache_ids: The positions in the KV cache that should be updated. shape=(seqlen,)
            start_ids: The offset from the beginning of each input in a batch. shape=(batch,)

        Returns:
            tokens: The next token prediction(s)
            probabilities: The next token probability(s)
        """
        start_len = 0
        if cache_ids:
            start_len = cache_ids.item()

        if start_len == 0:  # run context network as cache_id location starts from 0.
            return self._context_block(input_ids, start_ids)

        next_token_scores = self.model(input_ids, cache_ids, start_ids)

        scores = []
        tokens = []

        # Speculate k tokens in auto regressive mode.
        for cur_len in range(start_len, start_len + k):
            next_len = cur_len + 1
            topk_values, topk_indices = torch.topk(next_token_scores, k=1)
            probs = torch.nn.functional.softmax(topk_values, dim=-1)
            inputs_in_topk = torch.multinomial(probs, num_samples=1, replacement=True)
            inputs = torch.gather(topk_indices, 1, inputs_in_topk)

            scores.append(next_token_scores)
            tokens.append(inputs)

            if next_len >= start_len + k:
                break

            cache_ids = torch.as_tensor([next_len], dtype=torch.int32)
            next_token_scores = self.model(inputs, cache_ids, start_ids)

        return (
            torch.cat(tokens, dim=1),
            torch.cat(scores, dim=0)
        )


class SpeculativeGenerator:

    def __init__(
        self,
        draft: Union[DraftProvider, base.NeuronModelBase],
        target: base.NeuronModelBase,
        k: int = 4,
        acceptor: Optional[TokenAcceptor] = None
    ):
        """
        Implementation of DeepMind paper (https://arxiv.org/pdf/2302.01318.pdf) speculative sampling loop.

        In this implementation, Draft provider speculates (k-tokens, k-next_scores).
        The last predicted token by the target model + the k tokens predicted by the draft model are passed to the
        speculative decoder in the target model to get k+1 token scores.
        The TokenAcceptor then takes k-next_scores from the draft model and
        k+1 token scores from the target model to return the accepted tokens.

        Above loop is repeated either till end of sequence generation or early stop where eos_token_id is detected.

        Args:
            draft:
                DraftProvider model that provides the speculated k tokens.
            target:
                Target model that is derived from NeuronModelBase
            k:
                The number of parallel tokens Target model can accept in one forward pass.
            acceptor:
                TokenAcceptor that accepts the draft predicted tokens based on draft and target scores.
                This is default to DeepMind paper Token acceptor implementation.
        """
        if isinstance(draft, base.NeuronModelBase):
            draft = DraftModelForSpeculation(draft)
        self.draft = draft
        self.target = target
        self.k = k
        self.acceptor = acceptor or DefaultTokenAcceptor()

    def sample(
        self,
        input_ids: torch.Tensor,
        sequence_length: int,
        start_ids: Optional[torch.Tensor] = None,
        eos_token_id: Optional[int] = None,
        streamer: Optional['transformers.generation.streamers.BaseStreamer'] = None,
    ):
        """
        Speculative sampling loop:
        This is where the draft model speculates tokens and target model verifies them
        using an acceptance/rejection criteria. This happens in a loop either till
        end of sequence generation length or early stop is detected (based on eos_token_id).

        Args:
            input_ids:
                The input token ids passed to the model to generate
                next predicted tokens (sequence_length - len(input_ids)).
            sequence_length:
                The total length of inputs + outputs
            start_ids:
                The offset from the beginning of each input in a batch.
            eos_token_id:
                The id for the end of sentence token
            streamer:
                The streamer to be used for streaming generated tokens.

        Returns:
            tokens (tensor of shape (batch, sequence_length)):
                Input and output tokens predicted by the model via Speculative decoding.
        """
        batch_size, start = input_ids.shape
        if batch_size > 1:
            raise NotImplementedError("Current speculative sampling loop supported only with batch size = 1.")

        # run model context network blocks
        _draft_id, _draft_score = self.draft(input_ids, self.k, None, start_ids)
        target_score = self.target(input_ids, None, start_ids)
        target_next_id = sampling.select_tokens(target_score)  # TODO add generation args

        if streamer:
            streamer.put(target_next_id)

        # Set up early stopping
        early_stop = False
        if eos_token_id is not None:
            done_flags = torch.full((batch_size, 1), False)
            eos_token = torch.tensor(eos_token_id, dtype=torch.int32)
            early_stop = True

        tokens: list[torch.Tensor] = [input_ids, target_next_id]

        current = start
        while True:

            if early_stop:
                done_flags |= (target_next_id == eos_token)
                if batch_size > 1:  # Avoid writing tokens to completed sequences
                    target_next_id[done_flags] = eos_token
                if done_flags.all():
                    break

            draft_cache_id = torch.tensor([current], dtype=torch.int32)

            # returns auto-regressive k - 1 speculated tokens (as one token was already predicted by target)
            # draft_ids is of shape: (bs, k-1)
            # draft_next_scores has k-1 scores and of shape: (k-1, vocab)
            draft_ids, draft_next_scores = self.draft(target_next_id, self.k - 1, draft_cache_id, None)

            # Execute target model with draft tokens
            cache_ids = torch.arange(current, current + draft_ids.shape[1] + 1)  # added length of target predicted token
            input_ids = torch.cat([target_next_id, draft_ids], dim=1)
            # Target model fwd pass returns results of shape [k , vocab, bs]
            target_next_scores = self.target.speculative_forward(input_ids=input_ids, cache_ids=cache_ids,
                                                                 start_ids=start_ids, speculation_length=self.k)
            # TODO FixMe: to support batching as current support is only with bs=1
            target_next_scores = target_next_scores.squeeze(dim=-1)

            # Select which tokens will be used
            accepted_tokens = self.acceptor(draft_ids, draft_next_scores, target_next_scores)

            # NOTE: Required for backwards compatibility since the Acceptor did not return batched inputs
            if accepted_tokens.dim() != 2:
                accepted_tokens = accepted_tokens.view(1, -1)

            _, num_accepted = accepted_tokens.shape
            if sequence_length - num_accepted < self.k:
                accepted_tokens = accepted_tokens[:, :sequence_length - len(tokens)]

            for index in range(num_accepted):
                token = accepted_tokens[:, index:index + 1]

                # Update done flags.
                if early_stop:
                    done_flags |= (token == eos_token)
                current = current + 1

                tokens.append(token)

                # Stream generated tokens
                if streamer:
                    streamer.put(token)

                if current >= sequence_length - 1 or (early_stop and done_flags.all()):
                    if streamer:
                        streamer.end()
                    return torch.cat(tokens, dim=-1)

            # accepted_tokens = 1 means, no draft token accepted
            # accepted_tokens = 2 means, 1 draft token accepted
            # accepted_tokens = K means, all draft tokens accepted + target model predicted token
            target_next_id = accepted_tokens[:, -1:]

        if streamer:
            streamer.end()

        return torch.cat(tokens, dim=-1)
