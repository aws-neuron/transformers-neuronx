import torch
import transformers
import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass
 
from transformers_neuronx import base
from transformers_neuronx import sampling
 
 
# -----------------------------------------------------------------------------
# Speculative Decoding - Interfaces
# -----------------------------------------------------------------------------
 
class TokenAcceptor:
    def __call__(
        self,
        draft_ids: torch.Tensor,
        draft_probabilities: torch.Tensor,
        target_probabilities: torch.Tensor,
    ) -> torch.Tensor:
        """
        A function which checks which draft tokens should be accepted.
 
        The target tokens and probabilities will be one larger than the
        draft tokens since the target model will generate a next probability
        for the final draft token.
 
        Args:
            draft_ids: Tokens from the draft model. shape=(batch, k)
            draft_probabilities: Token probabilities from the draft model. shape=(batch, k, vocab)
            target_probabilities: Tokens probabilities from the target model. shape=(batch, k + 1, vocab)
        """
        raise NotImplementedError
 
 
class DraftProvider:
    """
    Draft provider that speculates K tokens.
    """
    def __call__(
            self,
            input_ids: torch.Tensor,
            k: int,
            cache_ids: Optional[torch.Tensor] = None,
            start_ids: Optional[torch.Tensor] = None,
            state: Optional[torch.Generator] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process either context, next token, or draft tokens.
 
        The behavior of the model should be chosen based on the `cache_ids`:
        1. `cache_ids == None`:       [Draft] Context encoding
        2. `cache_ids.shape[1] == k`: [Draft] Speculated K tokens
 
        Similarly the returned values should be chosen based on the `cache_ids`:
        1. `cache_ids == None`:       [Draft] 1 Token & Probability
        2. `cache_ids.shape[1] == k`: [Draft] `k` Tokens & Probabilities
 
        Arguments:
            input_ids: Either context, next token, or draft tokens. shape=(batch, seqlen)
            k: The number of speculative tokens
            cache_ids: The positions in the KV cache that should be updated. shape=(seqlen,)
            start_ids: The offset from the beginning of each input in a batch. shape=(batch,)
            state: A random state for deterministic sampling.
 
        Returns:
            tokens: The next token prediction(s)
            probabilities: The next token probability(s)
        """
        raise NotImplementedError
 
 
# -----------------------------------------------------------------------------
# Data Classes
# -----------------------------------------------------------------------------
# TODO: We should move this to common utility module
#  (should not belong to this class as this could be leveraged in many places)
 
 
@dataclass
class ModelGenerationArgs:
    """
    Model Generation Args - Data class to track all the generation arguments used while model inference.
    """
    temperature: int
    top_k: int
    top_p: int
    sequence_length: int


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
 
def max_fn(x):
    x_max = torch.where(x > 0, x, 0)
    return x_max / x_max.sum()
 
def sample(p):
    return np.random.choice(np.arange(p.shape[-1]), p=p)
 
 
def sample_multinomial(probs, apply_softmax=False):
    if apply_softmax:
        probs = torch.softmax(probs, dim=-1)
    return torch.multinomial(probs, num_samples=1, replacement=True)
 
def sample_greedy(probs, top_k=1):
    return torch.argmax(probs) 

 
# -----------------------------------------------------------------------------
# Default Implementation of Interfaces
# -----------------------------------------------------------------------------

class DefaultTokenAcceptor(TokenAcceptor):
    def __call__(
            self,
            draft_ids: torch.Tensor,
            draft_probabilities: torch.Tensor,
            target_probabilities: torch.Tensor,  # prob start from last accepted token till
    ) -> torch.Tensor:
        """
        A function which checks which draft tokens should be accepted.
 
        The target tokens and probabilities will be one larger than the
        draft tokens since the target model will generate a next probability
        for the final draft token.
 
        Args:
            draft_ids: Tokens from the draft model. shape=(batch, k)
            draft_probabilities: Token probabilities from the draft model. shape=(batch, k, vocab)
            target_probabilities: Tokens probabilities from the target model. shape=(batch, k + 1, vocab)
        """
        all_accepted = True
        accepted_tokens = torch.as_tensor([], dtype=torch.int32)
        accepted_token_count = 0
        K = draft_ids.shape[1]  # get shape
 
        draft_ids = draft_ids[0]
 
        for i in range(K):
            draft_token_id = draft_ids[i]  # token id generated by draft model at ith place
 
            np_random = np.random.random()
 
            target_prob_i = target_probabilities[i][draft_token_id]
 
            draft_prob_i = draft_probabilities[i][draft_token_id]
 
            if np_random < min(1, target_prob_i / draft_prob_i):  # accepted;
                print(f"Info: Token ACCEPTED: {draft_token_id} and shape: {draft_token_id.shape}")
                accepted_tokens = torch.cat([accepted_tokens, torch.as_tensor([draft_token_id], dtype=torch.int32)])
                accepted_token_count += 1
 
            else:  # rejected
                print(f"Info: Token REJECTED: {draft_token_id} and shape: {draft_token_id.shape}")
                prob_diff = target_probabilities[i]-draft_probabilities[i]
                accepted_tokens = torch.cat([accepted_tokens, torch.as_tensor([sample(max_fn(prob_diff))])])                
                accepted_token_count += 1
                all_accepted = False
                break
 
        # Step 4: if all draft tokens were accepted, sample a final token
        if all_accepted:
            print(f"Info: ALL ACCEPTED")
            accepted_tokens = torch.cat([accepted_tokens, torch.as_tensor([sample_greedy(target_probabilities[-1])])])
            # accepted_tokens = torch.cat([accepted_tokens, sample_multinomial(target_probabilities[-1])])
            accepted_token_count += 1
 
        
        return accepted_tokens
 
class DraftModelForSpeculation(DraftProvider): 

    """
    Draft model that auto-regressively that speculates k tokens. 
    """
 
    def __init__(self, model, **kwargs) -> None:
        self.model = model
        self.kwargs = kwargs
 
    def _context_local(self, input_ids, start_ids):   
        """
        Local helper function to run context encoding network

        Arguments: 
        input_ids: The initial input tokens passed to the model
        start_ids: The offset from the beginning of each input in a batch. 

        Returns:
        tokens: The next token prediction
        logits: The next token logits
        """ 

        _, start = input_ids.shape  
        next_token_scores = self.model(input_ids, None, start_ids)
        inputs = torch.argmax(next_token_scores, dim=1, keepdim=True)
        return inputs, next_token_scores
 
    def __call__(
            self,
            input_ids: torch.Tensor,
            k: int,
            cache_ids: Optional[torch.Tensor] = None,
            start_ids: Optional[torch.Tensor] = None,
            state: Optional[torch.Generator] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform standard auto-regressive token generation using the draft model, to speculate k-tokens. 
 
        Arguments:
            input_ids: Either context, next token, or draft tokens. shape=(batch, seqlen)
            k: The number of speculative tokens
            cache_ids: The positions in the KV cache that should be updated. shape=(seqlen,)
            start_ids: The offset from the beginning of each input in a batch. shape=(batch,)
            state: A random state for deterministic sampling.
 
        Returns:
            tokens: The next token prediction(s)
            probabilities: The next token probability(s)
        """
        start_len = 0
        if cache_ids:
            start_len = cache_ids.item()
        
        
        if start_len == 0:
            return self._context_local(input_ids, start_ids)
 
        k_token_ids = torch.as_tensor([], dtype=torch.int32)
        
        next_token_scores = self.model(input_ids, cache_ids, start_ids)
        k_logits = torch.zeros((0, next_token_scores.shape[-1]))
 
        for cur_len in range(start_len, start_len + k):
            next_len = cur_len + 1
            k_logits = torch.cat([k_logits, next_token_scores], dim=0)
            # TODO: Update values from p/k/temp/stop using topk
            topk_values, topk_indices = torch.topk(next_token_scores, k=1)
            probs = torch.nn.functional.softmax(topk_values, dim=-1)
            inputs_in_topk = torch.multinomial(probs, num_samples=1, replacement=True)
            inputs = torch.gather(topk_indices, 1, inputs_in_topk)
            k_token_ids = torch.cat([k_token_ids, inputs], dim=1)
            if next_len > start_len + k:
                break
            cache_ids = torch.as_tensor([next_len], dtype=torch.int32)
            next_token_scores = self.model(inputs, cache_ids, start_ids)
 
        print(f"Info: K-Draft tokens generated: {k_token_ids} and shape of logits: {k_logits.shape}")
        return k_token_ids, k_logits
 
 
 
# -----------------------------------------------------------------------------
# Main speculative sampling loop
# -----------------------------------------------------------------------------
 
 
class SpeculativeGenerator: 
 
    """
    Execute standard speculative sampling

    Arguments:
    draft: DraftProvider model/cache that provides the speculated k tokens
    target: Target model that is derived from NeuronModelBase
    k: The number of tokens we want to speculate
    acceptor: TokenAcceptor that accepts/rejects tokens
    """
 
    def __init__(
        self,
        draft: DraftProvider,
        target: base.NeuronModelBase,
        k: int = 4,
        acceptor: Optional[TokenAcceptor] = None,
        **kwargs,
    ):
        self.draft = draft
        self.target = target
        self.k = k
        self.acceptor = acceptor  
 
    def sample(
        self,
        input_ids: torch.Tensor,
        sequence_length: int,
        start_ids: Optional[torch.Tensor] = None,
        eos_token_id: Optional[int] = None,
        streamer: Optional['transformers.generation.streamers.BaseStreamer'] = None,
    ):
        """
        Speculative sampling loop


        input_ids: The input tokens passed to the model
        sequence_length: The total length of inputs + outputs
        start_ids: The offset from the beginning of each input in a batch. 
        eos_token_id: The id for the end of sentence token
        streamer: The streamer to be used for streaming tokens

        Returns:
        Final input+output tokens post speculative sampling
        """
        batch_size, start = input_ids.shape
       
        # Populate the both KV caches with prompt
        _draft_ids, _draft_probs = self.draft(input_ids, self.k, None, start_ids)
        target_logit = self.target(input_ids, None, start_ids)
 
        # target next token_id from target model via context network
        target_next_id = sampling.select_tokens(target_logit)  # TODO add gen args
        target_probs = torch.softmax(target_logit, dim=-1)
        
        # Set up early stopping
        early_stop = False
        if eos_token_id is not None:
            done_flags = torch.full((batch_size, 1), False)
            eos_token = torch.tensor(eos_token_id, dtype=torch.int32)
            early_stop = True
 
        tokens: list[torch.Tensor] = [input_ids]
        tokens = torch.cat([input_ids[0], torch.as_tensor([target_next_id])])
        
        current = start
        while True:
 
            if early_stop:
                done_flags |= (target_next_id == eos_token)
                if batch_size > 1:  # Avoid writing tokens to completed sequences
                    target_next_id[done_flags] = eos_token
 
            
            if streamer:
                streamer.put(target_next_id)
 
            if early_stop:
                if done_flags.all():
                    break
 
            if current >= sequence_length - 1:
                print(f"Info: Given target_current: {current} > seq_len: {sequence_length}.. \nInference completed")
                break
 
            # Perform traditional token generation on the draft model
            
            draft_cache_id = torch.tensor([current], dtype=torch.int32)
 
            # returns auto-regressive k - 1 speculated tokens (as one token already predicted by target)
            
 
            draft_ids, draft_probs = self.draft(target_next_id, self.k - 1, draft_cache_id, None)
            # draft_probs has K-1 logits
            
            # Execute target model with draft tokens
            cache_ids = torch.arange(current, current + draft_ids.shape[1] + 1)  # added length of target predicted token
            input_ids = torch.cat([target_next_id, draft_ids], dim=1)
            target_new_probs = self.target(input_ids=input_ids, cache_ids=cache_ids, start_ids=start_ids)    
            target_new_probs = target_new_probs.squeeze(dim=-1) 
 
            # Select which tokens will be used
            accepted_tokens = self.acceptor(draft_ids, draft_probs, target_new_probs)
            if sequence_length - len(tokens) < self.k:
                accepted_tokens = accepted_tokens[:sequence_length - len(tokens)]
 
            tokens = torch.cat([tokens, accepted_tokens])
            accepted_tokens_count = len(accepted_tokens)
 
            # accepted_tokens = 1 means, no draft token accepted
            # accepted_tokens = 2 means, 1 draft token accepted
            # accepted_tokens = K means, all draft tokens accepted + target model predicted token
            target_next_id = torch.unsqueeze(torch.tensor([accepted_tokens[-1]]), 0) # accepted_tokens[-1]  # get last token as accepted token
            current += accepted_tokens_count   # update draft token current length

 
 
        if streamer:
            streamer.end()
 
        return tokens