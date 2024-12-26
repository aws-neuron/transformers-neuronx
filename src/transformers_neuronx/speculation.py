import torch
import transformers
from typing import Optional, Tuple, Union, Dict, List

from transformers_neuronx import base
from transformers_neuronx import sampling
from transformers_neuronx.util.token_tree import validate_token_tree


class TokenTreeAcceptor:
    """
    Abstract base class for token tree acceptor that is used in Speculative Sampling loop,
    to check which draft tokens should be accepted.
    Note-1: Batch size > 1 is not yet supported for Speculative Decoding.
    """

    def __call__(
        self,
        draft_ids: torch.Tensor,
        draft_scores: torch.Tensor,
        target_scores: torch.Tensor,
        token_tree: Dict[int, List[int]],
    ) -> (torch.Tensor, List[int]):
        """
        Args:
            draft_input_ids (`torch.Tensor` of shape `(speculated_token_length)`):
                Tokens ids from the draft model.

            draft_scores (`torch.Tensor` of shape `(speculated_token_length, vocab)`):
                Prediction scores of the draft model.

            target_scores (`torch.Tensor` of shape `(speculated_token_length, vocab)`):
                Token scores from the target model.

            token_tree Token tree structure being used for speculation.

        Return:
            accepted_token_ids (`torch.Tensor` of shape `(accepted_speculated_token)`):
                The accepted draft predicted token ids. Length of accepted_speculated_tokens <= speculated_token_length.

            accepted_indices List of node ids that were accepted as tokens.

        """
        raise NotImplementedError(f"{self.__class__} is an abstract class. Only"
                                  f" classes inheriting this class can be called.")

class TokenAcceptor:
    """
    Abstract base class for token acceptor that is used in Speculative Sampling loop,
    to check which draft tokens should be accepted.

    Note-1: The length of target scores will be one greater than the length of draft scores since the target model
    will additionally generate a next score for the final draft token..
    """

    def __call__(
        self,
        draft_ids: torch.Tensor,
        draft_scores: torch.Tensor,
        target_scores: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            draft_input_ids (`torch.Tensor` of shape `(batch_size, speculated_token_length-1)`):
                Tokens ids from the draft model.

            draft_scores (`torch.Tensor` of shape `(batch_size, speculated_token_length-1, vocab)`):
                Prediction scores of the draft model.

            target_scores (`torch.Tensor` of shape `(batch_size, speculated_token_length, vocab)`):
                Token scores from the target model.

        Return:
            accepted_token_ids (`torch.Tensor` of shape `(batch_size, speculated_token_length)`):
                The accepted draft predicted token ids. Length of accepted_speculated_tokens <= speculated_token_length.

        """
        raise NotImplementedError(f"{self.__class__} is an abstract class. Only"
                                  f" classes inheriting this class can be called.")


class DraftProvider:
    """
    Abstract base class for Draft provider to speculate `k-1` tokens,
    where (k) is the number of parallel tokens target model can decode in one forward pass.

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

            cache_ids (`torch.Tensor` of shape `(sequence_length)` or `(batch_size,sequence_length)`):
                The positions in the KV cache that should be updated. KV Cache positions start from 0.

                For example: If your input ids are torch.tensor([108 120 130 140]) and predicted next_token_id is 150,
                to generate a next token, cache_ids is set as 4 and input_ids set to torch.tensor([150]).

                The behavior of the model should be chosen based on the `cache_ids`:
                1. `cache_ids == None`:       [Draft] Context encoding
                2. `cache_ids.shape[1] == 1`: [Draft] Speculated K tokens

            start_ids (`torch.Tensor` of shape `(batch)`):
                In the case of batched speculation, this is now the sequence_ids.

        Returns:
            tokens (`torch.Tensor` of shape `(batch_size, k)`):
                The next token prediction(s) where k = number of speculated tokens.

            scores (`torch.Tensor` of shape `(batch_size, k, config.vocab)`):
                 The next token score(s)`.

        """
        raise NotImplementedError(f"{self.__class__} is an abstract class. Only"
                                  f" classes inheriting this class can be called.")


class ReferenceTokenAcceptor(TokenAcceptor):
    """
    Reference Implementation of TokenAcceptor defined as per the DeepMind paper: https://arxiv.org/pdf/2302.01318.pdf
    This is only for batch_size 1. 
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


class DefaultTokenTreeAcceptor(TokenTreeAcceptor):
    """
    A simple greedy Token Tree Acceptor where we use argmax to pick the token with the largest scores.
    """
    def __call__(
            self,
            draft_ids: torch.Tensor,
            draft_scores: torch.Tensor,
            target_scores: torch.Tensor,
            token_tree: Dict[int, List[int]],
    ):
        draft_token_len, draft_vocab = draft_scores.shape
        target_token_len, target_vocab = target_scores.shape
        assert draft_vocab == target_vocab
        assert draft_token_len == target_token_len
        accepted_tokens = []
        accepted_indices = [0] # Root node always accepted
        def discover_acceptance(node: int):
            if node in token_tree and len(token_tree[node]) != 0:
                # Not leaf
                new_draft_accepted_node = None
                target_token = torch.argmax(target_scores[node:node+1, :],keepdim=True, dim=1)
                for child in token_tree[node]:
                    if target_token == draft_ids[:, child]:
                        new_draft_accepted_node = child
                accepted_tokens.append(target_token)
                if new_draft_accepted_node is not None:
                    accepted_indices.append(new_draft_accepted_node)
                    discover_acceptance(new_draft_accepted_node)
            else:
                # Leaf, generate extra token by target model
                target_token = torch.argmax(target_scores[node:node+1, :],keepdim=True, dim=1)
                accepted_tokens.append(target_token)
        discover_acceptance(0)
        return torch.cat(accepted_tokens, dim=1), accepted_indices


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
        
        batch_size, draft_token_len, draft_vocab = draft_scores.shape
        batch_size, target_token_len, target_vocab = target_scores.shape
        assert draft_vocab == target_vocab  # vocab size should be same
        assert draft_token_len + 1 == target_token_len  # target should include additional token predicted
        accepted_tokens=torch.zeros(batch_size, target_token_len,dtype=torch.int64)

        
        draft_probabilities = torch.softmax(draft_scores, dim=-1)
        target_probabilities = torch.softmax(target_scores, dim=-1)
        index = torch.reshape(draft_ids, (1, draft_token_len, batch_size))
        target_probs = torch.gather(target_probabilities[:,:-1], -1, index)
        draft_probs = torch.gather(draft_probabilities, -1, index)
        target_probs = target_probs.squeeze(0)              # shape: (k, batch_size)
        draft_probs = draft_probs.squeeze(0)                # shape: (k, batch_size)

        random = torch.rand(draft_probs.shape)
        ratio = torch.clamp(target_probs / draft_probs, max=1.0)
        accepted = torch.less(random, ratio)

        # Minimum will return the first occurance of 0 or False (i.e. rejection)
        minimum = torch.min(accepted.view(torch.uint8), dim=0)
        values = minimum.values
        indices = minimum.indices

        def sample(probs):
            return torch.multinomial(probs, num_samples=1, replacement=True)

        
        for batch_dim in range(batch_size):
            
            if values[batch_dim] != 0: # If we didn't get a rejection this means all drafts were accepted
                next_token = sample(target_probabilities[batch_dim, -1:])
                tokens=torch.cat((draft_ids[batch_dim], next_token[0]), dim=-1)
                positions=torch.arange(target_token_len)
                
            else: 
                index=indices[batch_dim]
                prob_diff = target_probabilities[batch_dim, index:index + 1] - draft_probabilities[batch_dim, index: index + 1]
                prob_diff = torch.clamp(prob_diff, min=0.0)
                next_token = sample(prob_diff)
                tokens=torch.cat((draft_ids[batch_dim,:index], next_token[0]), dim=-1)
                positions=torch.arange(tokens.shape[0])
            
            accepted_tokens[batch_dim].scatter_(0,positions,tokens)    

        return accepted_tokens


class DraftModelForSpeculation(DraftProvider):
    """
    Standard Implementation of Draft model provider that auto-regressively speculates k tokens.
    """

    def __init__(self, model, **kwargs) -> None:
        self.model = model
        self.kwargs = kwargs

    def _context_block(self, input_ids, cache_ids, start_ids):
        """
        Run context encoding network of the given model.

        Args:
            input_ids: The initial input tokens passed to the model
            cache_ids: The positions in the KV cache that should be updated.
            start_ids: For batched speculation, this is sequences ids.

        Returns:
            token: predicted next token
            score: predicted token score
        """
        next_token_scores = self.model(input_ids, cache_ids, start_ids)
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
            cache_ids: The positions in the KV cache that should be updated. shape=(seqlen,) or (batch,seqlen)
            start_ids: For batched speculation, this is sequences ids. shape=(batch,)

        Returns:
            tokens: The next token prediction(s)
            probabilities: The next token probability(s)
        """
        start_len = 0
        batch_size, _ = input_ids.shape
        if cache_ids is not None:
            start_len = cache_ids.item() if cache_ids.dim() == 1 else cache_ids[0][0]
       
        if start_len == 0:  # run context network as cache_id location starts from 0.
            return self._context_block(input_ids, cache_ids, start_ids)

        next_token_scores = self.model(input_ids, cache_ids, start_ids)

        scores = []
        tokens = []
        
        # Speculate k tokens in auto regressive mode.
        for cur_len in range(start_len, start_len + k):
            next_len = cur_len + 1
            inputs = torch.argmax(next_token_scores, keepdim=True, dim=1)

            scores.append(next_token_scores)
            tokens.append(inputs)

            if next_len >= start_len + k:
                break

            cache_ids = torch.as_tensor([next_len]*batch_size, dtype=torch.int32)
            if batch_size>1:
                cache_ids=cache_ids.reshape(batch_size,1)
            next_token_scores = self.model(inputs, cache_ids, start_ids)

        
        return (
            torch.cat(tokens, dim=1),
            torch.stack(scores)
        )


class DraftModelForTreeSpeculation(DraftProvider):
    """
    Implementation of Draft model provider that uses tree based speculates to populate the
    whole token tree structure. For every depth of the tree, a speculation call is forwarded.
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

    def _fetch_inputs(self, next_token_scores, previous_inputs, token_tree):
        """
        From the next_token_scores of shape [k, vocab, bs], this fills up the
        tokens for the whole tree structure.
        """
        for i in range(previous_inputs.shape[1]):
            if i in token_tree and len(token_tree[i]) != 0:
                # Non Leaf
                count_of_child = len(token_tree[i])
                vals, indices = torch.topk(next_token_scores[i:i+1, :, :], count_of_child, dim=1)
                for i, child_index in enumerate(token_tree[i]):
                    previous_inputs[:, child_index] = indices[:, i:i+1, :]
        return previous_inputs

    def __call__(
            self,
            input_ids: torch.Tensor,
            k: int,
            depth: int,
            token_tree: Dict[int, List[int]],
            pad_token: int = 1,
            cache_ids: Optional[torch.Tensor] = None,
            start_ids: Optional[torch.Tensor] = None,
            previous_cache_ids: Optional[torch.Tensor] = None,
            reorder_mapping: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform standard auto-regressive token generation using the draft model, to speculate k-tokens
        using the provided token-tree structure.

        Args:
            input_ids: Either context, next token, or draft tokens. shape=(batch, seqlen)
            k: Total nodes in the token tree
            depth: Total levels in the token tree
            token_tree: The token tree used for generating tokens
            pad_token: Token used to pad the draft speculation inputs.
            cache_ids: The positions in the KV cache that should be updated. shape=(seqlen,)
            start_ids: The offset from the beginning of each input in a batch. shape=(batch,)
            previous_cache_ids: The previous cache ids used with the draft speculation.
            reorder_mapping: The reorder mapping corresponding to previous cache ids based on acceptance.

        Returns:
            tokens: The next token prediction(s) of shape [BS, K]
            probabilities: The next token probability(s) of shape [K, VOCAB]
        """
        if cache_ids is None:
            return self._context_block(input_ids, start_ids)

        if pad_token is None:
            pad_token = 1
        inputs = torch.full((1, k), pad_token, dtype=torch.int32)
        inputs[:, 0] = input_ids
        next_token_scores = self.model.tree_speculative_forward(input_ids=inputs, cache_ids=cache_ids,
                                                           start_ids=start_ids, speculation_length=k,
                                                           previous_cache_ids=previous_cache_ids, reorder_mapping=reorder_mapping)

        for _ in range(depth-1):
            inputs = self._fetch_inputs(next_token_scores, inputs, token_tree)
            next_token_scores = self.model.tree_speculative_forward(input_ids=inputs, cache_ids=cache_ids,
                                                           start_ids=start_ids, speculation_length=k,
                                                           previous_cache_ids=None, reorder_mapping=None)

        return (
            inputs,
            next_token_scores.squeeze(dim=-1)
        )


class TreeSpeculativeGenerator:
    def __init__(
        self,
        draft: Union[DraftProvider, base.NeuronModelBase],
        target: base.NeuronModelBase,
        token_tree: Dict[int, List[int]] = None,
        acceptor: Optional[TokenTreeAcceptor] = None
    ):
        """
        Args:
            draft :
                DraftProvider model that provides the speculated token tree.
            target:
                Target model that is derived from NeuronModelBase.
            token_tree:
                Token tree definition used for speculation.
            acceptor:
                TokenTreeAcceptor that accepts the draft predicted tokens based on draft and target scores.
                This is default to greedy implementation.
            
        """
        if isinstance(draft, base.NeuronModelBase):
            draft = DraftModelForTreeSpeculation(draft)
        self.draft = draft
        self.target = target
        self.token_tree = token_tree
        self.k, self.depth = validate_token_tree(token_tree)
        self.acceptor = acceptor or DefaultTokenTreeAcceptor()

    def _generate_reorder_mapping(
        self,
        accepted_indices
    ):
        """
        Generates the reorder_mapping which is used to reorder the KV cache during speculation
        for the previous cache_id window based on the accepted_indices returned by the acceptor.
        """
        result = [x for x in range(self.k)]
        for i, index in enumerate(accepted_indices):
            temp = result[i]
            result[i] = index
            result[index] = temp
        return torch.tensor(result, dtype=torch.int32)
            

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
        This is where the draft model speculates token tree and target model verifies them
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
        _draft_id, _draft_score = self.draft(input_ids, self.k, self.depth, self.token_tree, eos_token_id, None, start_ids)
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
        reorder_mapping = None
        previous_cache_ids = None
        while True:

            if early_stop:
                done_flags |= (target_next_id == eos_token)
                if batch_size > 1:  # Avoid writing tokens to completed sequences
                    target_next_id[done_flags] = eos_token
                if done_flags.all():
                    break

            # Build draft cache
            draft_cache_id = torch.arange(current, current+self.k, dtype=torch.int32)

            # returns auto-regressive k speculated tokens (first token was already predicted by target)
            # draft_ids is of shape: (bs, k)
            # draft_next_scores has k scores and of shape: (k, vocab)
            draft_ids, draft_next_scores = self.draft(target_next_id, self.k, self.depth, self.token_tree, eos_token_id, draft_cache_id, start_ids, previous_cache_ids, reorder_mapping)

            # Execute target model with draft tokens
            cache_ids = torch.arange(current, current + draft_ids.shape[1])
            # Target model fwd pass returns results of shape [k , vocab, bs]
            target_next_scores = self.target.tree_speculative_forward(input_ids=draft_ids, cache_ids=cache_ids,
                                                                 start_ids=start_ids, speculation_length=self.k,
                                                                 previous_cache_ids=previous_cache_ids, reorder_mapping=reorder_mapping)
            previous_cache_ids = cache_ids

            # TODO FixMe: to support batching as current support is only with bs=1
            target_next_scores = target_next_scores.squeeze(dim=-1)

            # Select which tokens will be used
            accepted_tokens, accepted_indices = self.acceptor(draft_ids, draft_next_scores, target_next_scores, self.token_tree)
            
            # NOTE: Required for backwards compatibility since the Acceptor did not return batched inputs
            if accepted_tokens.dim() != 2:
                accepted_tokens = accepted_tokens.view(1, -1)

            # Reorder mapping computation logic here
            reorder_mapping = self._generate_reorder_mapping(accepted_indices)

            _, num_accepted = accepted_tokens.shape
            if sequence_length - num_accepted < self.depth:
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
            # accepted_tokens = K means, all draft tokens accepted 
            #                   along a tree path from root to leaf + target model predicted token
            target_next_id = accepted_tokens[:, -1:]

            # Boundary condition: If we overflow the boundary then step generating more tokens.
            if current >= sequence_length:
                # TODO: Right now there might be an overflow in generated tokens
                # to be more than seq len. We can fix this up by trimming the extra generated tokens.
                break
        if streamer:
            streamer.end()
        
        return torch.cat(tokens, dim=-1)
        

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

        In this implementation, Draft provider speculates (k-1 tokens, k-1 next_scores).
        The last predicted token by the target model + the k-1 tokens predicted by the draft model are passed to the
        speculative decoder in the target model to get k token scores.
        The TokenAcceptor then takes k-1 next_scores from the draft model and
        k token scores from the target model to return the accepted tokens.

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
        pad_token_id: Optional[int] = 0,
        attention_mask: Optional[torch.Tensor] = None,
        streamer: Optional['transformers.generation.streamers.BaseStreamer'] = None,
        output_logits: bool = False,
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
                The id for the end of sentence token. Results in early stop as soon as eos_token is reached.
            pad_token_id:
                The id for the padding token.
            attention_mask:
                A binary mask with the same shape as input_ids that
                indicates which tokens should be attended to.
            streamer:
                The streamer to be used for streaming generated tokens.
            output_logits:
                Whether to include logits in the output. Used for internal testing.

        Returns:
            tokens (tensor of shape (batch, sequence_length)):
                Input and output tokens predicted by the model via Speculative decoding.
        """
        batch_size, context_len = input_ids.shape
        cache_ids=None
        draft_logits, target_logits = [], []  # only used if output_logits == True
        
        if batch_size>1:
            # We need the user to provide attention mask along with padded input_ids for batched cases
            assert attention_mask is not None, (
            "The attention mask needs to be provided for speculation where batch_size>1"
            )
            # Calculating the 2D cache_ids explicitly for batched cases
            cache_ids = torch.arange(context_len).reshape(1, context_len).expand(batch_size, context_len).mul(attention_mask)
            if start_ids is None:
                start_ids= torch.arange(batch_size)
        
        # Context encoding
        _draft_ids, _draft_scores = self.draft(input_ids, self.k, cache_ids, start_ids)
        target_scores = self.target(input_ids, cache_ids, start_ids)
        if output_logits:
            target_logits.append(target_scores.unsqueeze(1))
                
        # The streamer should send back the input tokens to conform to
        # huggingface behavior.
        # Reference: https://github.com/huggingface/transformers/blob/v4.39.0/src/transformers/generation/utils.py#L1410-L1411
        if streamer:
            streamer.put(input_ids)

        target_next_ids = sampling.select_tokens(target_scores)  # TODO add generation args
        if streamer:
            streamer.put(target_next_ids)

        done_flags = torch.full((batch_size, 1), False)
        
        # Set up early stopping
        early_stop = False
        if eos_token_id is not None:
            eos_token = torch.tensor(eos_token_id, dtype=torch.int64)
            early_stop = True

        # Intialize tokens tensor
        tokens=torch.full((batch_size, sequence_length + self.k + 1), pad_token_id, dtype=torch.int64)
        
        # Insert input_ids into tokens tensor
        input_ids=input_ids.to(dtype=torch.int64) #sanity check before scatter
        update_positions=torch.arange(context_len, dtype=torch.int64).repeat((batch_size, 1))
        tokens.scatter_(1,update_positions,input_ids)

        # Calculate end positions and cache_ids
        end_positions=torch.tensor([context_len], dtype=torch.int64)
        if attention_mask is not None:
            end_positions=torch.count_nonzero(attention_mask,dim=1)
        cache_ids=end_positions.clone().detach()

        # Insert target_next_ids into tokens tensor 
        tokens.scatter_(1,end_positions.view(-1,1),target_next_ids)
        end_positions+=1
        
        # Initialize useful tensors
        k_arange_tensor=torch.arange(self.k, dtype=torch.int64).repeat((batch_size, 1))

        # Main sampling loop
        while True:
            
            if early_stop:
                done_flags |= (target_next_ids == eos_token)
                if batch_size > 1:  # Avoid writing tokens to completed sequences
                    target_next_ids[done_flags] = eos_token
            
            if done_flags.all():
                break
            
            # Set draft model 1D or 2D cache_ids
            draft_cache_ids=cache_ids
            if batch_size>1:
                draft_cache_ids=draft_cache_ids.view(-1,1)
            
            # Returns auto-regressive k - 1 speculated tokens (as one token was already predicted by target)
            # draft_ids is of shape: (bs, k-1)
            # draft_next_scores has k-1 scores and of shape: (bs, k-1, vocab)
            draft_ids, draft_next_scores = self.draft(target_next_ids, self.k - 1, draft_cache_ids, start_ids)

            # Set target 1D or 2D cache_ids
            if batch_size==1:
                target_cache_ids = torch.arange(cache_ids[0], cache_ids[0] + draft_ids.shape[1] + 1)  # added length of target predicted token
            else:
                target_cache_ids=k_arange_tensor+cache_ids.view(-1,1)
          
            # Execute target model with draft tokens
            # Target model speculative forward pass returns results of shape (k , vocab, bs)
            target_input_ids = torch.cat([target_next_ids, draft_ids], dim=1)
            target_next_scores = self.target.speculative_forward(input_ids=target_input_ids, cache_ids=target_cache_ids,
                                                                 start_ids=start_ids, speculation_length=self.k)
            
            # A bunch of transposes to get the scores into shape (bs, k, vocab)
            draft_next_scores = torch.transpose(draft_next_scores,0,1)
            target_next_scores = torch.transpose(target_next_scores,0,2)
            target_next_scores = torch.transpose(target_next_scores,1,2)

            # Run token acceptor and returns padded (bs, k) tensor with accepted tokens
            # Tensor is padded with zeros for seqs with accepted tokens < k
            accepted_tokens = self.acceptor(draft_ids, draft_next_scores, target_next_scores)
            if output_logits:
                draft_logits.append(draft_next_scores)
                target_logits.append(target_next_scores)
        
            # Calculate actual acceptance lengths and end_positions
            actual_accepted_tokens_lengths=torch.count_nonzero(accepted_tokens,dim=1)
            update_positions=k_arange_tensor+end_positions.view(-1,1)
            
            # Masking tokens beyond eos_token and replacing extra "accepted" tokens with eos_token for early stop
            if early_stop:
                eos_mask = (accepted_tokens == eos_token_id).cumsum(1).bool()
                eos_mask |= done_flags.view(-1, 1)
                accepted_tokens[eos_mask] = eos_token_id

            # Update sequences
            tokens.scatter_(1, update_positions, accepted_tokens)
            cache_ids+=actual_accepted_tokens_lengths
            end_positions+=actual_accepted_tokens_lengths
            target_next_ids= accepted_tokens[:,actual_accepted_tokens_lengths-1]
            if batch_size>1:
                target_next_ids=target_next_ids[:,0].view(-1,1)
            
            if streamer:
                streamer.put(accepted_tokens)
            
            # Marking done flags
            if early_stop:
                done_flags |= eos_mask.any(dim=1).view(-1,1)
            
            # If we accepted all tokens then we need to insert the last draft
            # token into the KV cache since it was generated but never executed
            if (actual_accepted_tokens_lengths == self.k).any():
                draft_tokens=accepted_tokens[:, -2:-1]
                draft_cache_ids=cache_ids-1
                if batch_size>1:
                    draft_cache_ids=draft_cache_ids.view(-1,1)
                self.draft(draft_tokens, 1, draft_cache_ids, start_ids)

            # Clamp the cache_ids and end positions to the sequence length so that any batch lines
            # that have not reached the end can continue. For batch lines that
            # are complete this populates garbage data into KV cache tail beyond
            # the sequence_length.
            end_positions=torch.clamp(end_positions, max=sequence_length)
            cache_ids = torch.clamp(cache_ids, max=sequence_length)
            done_flags.logical_or_(torch.eq(cache_ids, sequence_length).unsqueeze(1))

        if streamer:
            streamer.end()

        return tokens[:,:sequence_length] if not output_logits else (tokens[:,:sequence_length], draft_logits, target_logits)


