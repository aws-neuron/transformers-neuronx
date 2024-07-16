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
from typing import Optional

import torch
import transformers

from transformers_neuronx.config import GenerationConfig
from transformers_neuronx.stopping_criteria import StoppingCriteriaList


@torch.no_grad()
def simple_sample(model, input_ids, start_ids, sequence_length, eos_token_id=2, top_k=50, streamer=None, output_scores=False, cache_ids=None):
    # populate key/value caches according to the prompt text
    next_token_scores = model(input_ids, None, start_ids)
    return sample_loop(model, input_ids, start_ids, next_token_scores, sequence_length,
                       eos_token_id, top_k, streamer, output_scores=output_scores, cache_ids=cache_ids)


@torch.no_grad()
def sample_tokens(
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        start_ids: Optional[torch.Tensor] = None,
        *,
        sequence_length: int = 128,
        config: Optional[GenerationConfig] = None,
        streamer: Optional['transformers.generation.streamers.BaseStreamer'] = None,
    ):
    """
    A sampling loop for a model that emits selected tokens.

    This sampling loop should be used when the token selection is built into
    the model itself.
    """
    batch_size, start = input_ids.shape

    # Populate the KV cache with prompt
    next_tokens = model(input_ids, None, start_ids)

    cache_ids = torch.arange(start, sequence_length, dtype=torch.int32).split(1)
    tokens = [input_ids]

    # Use a default config if none is provided
    if config is None:
        config = GenerationConfig()

    early_stop = False
    if config.eos_token_id is not None:
        done_flags = torch.full((batch_size, 1), False)
        eos_token = torch.tensor(config.eos_token_id, dtype=torch.int32)
        early_stop = True

    # Generate loop
    for current, cache_id in zip(range(start + 1, sequence_length + 1), cache_ids):

        if early_stop:
            done_flags |= (next_tokens == eos_token)
            if batch_size > 1:  # Avoid writing tokens to completed sequnces
                next_tokens[done_flags] = eos_token

        tokens.append(next_tokens)

        if streamer:
            streamer.put(next_tokens)

        if early_stop:
            if done_flags.all():
                break

        if current >= sequence_length:
            break

        next_tokens = model(next_tokens, cache_id, start_ids)

    if streamer:
        streamer.end()

    return torch.cat(tokens, dim=-1)


def sample_greedy(model, input_ids, start_ids=None, sequence_length=128):
    """
    A sampling loop that selects tokens according to the most probable score.

    This is useful as a reference implementation for on-device greedy sampling.
    """
    _, start = input_ids.shape
    next_token_scores = model(input_ids, None, start_ids)

    tokens = [input_ids]
    for cur_len in range(start, sequence_length):

        # greedy sample
        inputs = torch.argmax(next_token_scores, dim=1, keepdim=True)
        tokens.append(inputs)

        # forward pass to get next token
        cache_ids = torch.as_tensor([cur_len], dtype=torch.int32)
        next_token_scores = model(inputs, cache_ids, start_ids)

    return torch.cat(tokens, dim=-1)


def sample_loop(model, input_ids, start_ids, next_token_scores, sequence_length, eos_token_id=2,
                top_k=50, streamer=None, output_scores=False, neuron_config=None, log_softmax_scores=None, cache_ids=None):
    log_softmax = neuron_config and neuron_config.log_softmax_scores
    tokens = [input_ids]
    _, start = input_ids.shape
    if cache_ids:
        start=cache_ids.item() + 1
    scores = []
    ls_scores = []
    for cur_len in range(start, sequence_length):
        next_len = cur_len + 1

        # don't sample EOS
        next_token_scores[:, eos_token_id] = -float('inf')

        # Remove all tokens with a probability less than the last token of the top-k
        if output_scores:
            scores.append(next_token_scores)
            if log_softmax:
                ls_scores.append(log_softmax_scores)
        topk_values, topk_indices = torch.topk(next_token_scores, top_k)

        # sample
        probs = torch.nn.functional.softmax(topk_values, dim=-1, dtype=torch.float32)
        inputs_in_topk = torch.multinomial(probs, num_samples=1, replacement=True)
        inputs = torch.gather(topk_indices, 1, inputs_in_topk)
        tokens.append(inputs)

        if streamer:
            streamer.put(inputs)

        if next_len >= sequence_length:
            break

        # forward pass to get next token
        cache_ids = torch.as_tensor([cur_len], dtype=torch.int32)
        if log_softmax:
            next_token_scores, log_softmax_scores =  model(inputs, cache_ids, start_ids)
        else:
            next_token_scores = model(inputs, cache_ids, start_ids)

    if streamer:
        streamer.end()

    if output_scores:
        if log_softmax:
            return torch.cat(tokens, dim=-1), scores, ls_scores
        return torch.cat(tokens, dim=-1), scores

    return torch.cat(tokens, dim=-1)


def validate_top_k_top_p_min_tokens_to_keep(top_k, top_p, min_tokens_to_keep):
    if top_k is not None and (not isinstance(top_k, int) or not (top_k > 0)):
        raise ValueError('top_k has to be a strictly positive int.')

    if top_p is not None and (not isinstance(top_p, float) or not (0.0 < top_p <= 1.0)):
        raise ValueError('top_p has to be a strictly positive float that less than or equal to 1.0.')

    if min_tokens_to_keep is not None and (not isinstance(min_tokens_to_keep, int) or min_tokens_to_keep < 0):
        raise ValueError('min_tokens_to_keep has to be a non-negative int.')


def validate_no_repeat_ngram_size(ngram_size):
    if ngram_size and (not isinstance(ngram_size, int) or ngram_size <= 0):
        raise ValueError(f"`ngram_size` has to be a strictly positive integer, but is {ngram_size}")


def top_k_top_p_filtering(scores, top_k, top_p, min_tokens_to_keep=1):
    validate_top_k_top_p_min_tokens_to_keep(top_k, top_p, min_tokens_to_keep)

    input_size = scores.size(dim=-1)

    def safe_size(size):
        return min(max(size, min_tokens_to_keep), input_size)

    def input_value_and_indices():
        return scores, torch.arange(start=0, end=input_size).repeat([scores.size(dim=0), 1])

    def filter_by_top_k():
        return torch.topk(scores, safe_size(top_k))

    def filter_by_top_p(indices=None):
        """
        indices==None indicates that top_k filtering was not performed, perform top_p filtering on the entire scores.
        Otherwise, performs top_p filtering on the result of top_k filtering, and calculating cumulative probabilities
        only on the filtered result from top_k filtering.
        """
        def filter_sorted(sorted_scores):
            cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_scores, dim=-1, dtype=torch.float32), dim=-1)
            mask = cumulative_probs < top_p

            # Per the original paper (https://arxiv.org/pdf/1904.09751.pdf, page 4 bottom), this filter include the minimal 
            # subset of tokens (in descending sorted order of scores) with cumulative probability >= top_p. This means that 
            # we need an extra token, in addition to those filtered by above logic (cumulative_probs < top_p). We do this by 
            # inserting a True column and the head of the mask (therefore shifting the rest of mask to the right by one),
            # and dropping the last column of the mask.
            mask = torch.concat((torch.ones([mask.shape[0], 1], dtype=torch.bool), mask[:, :-1]), axis=-1)
            
            mask[:, :min_tokens_to_keep] = True
            n_to_keep = safe_size(mask.int().sum(dim=-1).max().item())
            sorted_scores = sorted_scores[:, :n_to_keep]
            mask = mask[:, :n_to_keep]

            # Performed top_p on all batches. Need to return all batches' filtered values in one matrix. Therefore,
            # we need to set the values that correspond to unwanted indices -- those that where mask has value False
            # -- to -inf. This way subsequent sampling logic will not pick up these unwanted token indices.
            sorted_scores[~mask] = -float('inf')
            return sorted_scores

        if indices is None:
            # Not filtered by filter_by_top_k
            ret_scores, ret_indices = torch.sort(scores, descending=True)
            ret_scores = filter_sorted(ret_scores)
            return ret_scores, ret_indices[:, :ret_scores.size(dim=-1)]

        # Already filtered by filter_by_top_k, the value sequences represented by indices are already sorted.
        ret_scores = filter_sorted(torch.gather(scores, 1, indices))
        return ret_scores, indices[:, :ret_scores.size(dim=-1)]

    if (top_k is None and top_p is None) or min_tokens_to_keep > input_size:
        # Nothing to filter
        return input_value_and_indices()

    # Only filter by top_k
    if top_k is not None and top_p is None:
        return filter_by_top_k()

    # Only filter by top_p
    if top_k is None and top_p is not None:
        return filter_by_top_p()

    # Filter by top_k followed by top_p
    return filter_by_top_p(filter_by_top_k()[1])


def _get_ngrams(ngram_size: int, prev_input_ids: torch.Tensor, num_hypos: int):
    # Copied from huggingface logits_process.py
    # Initialize an empty list of dictionaries, one for each hypothesis (index) in the range of num_hypos
    generated_ngrams = [{} for _ in range(num_hypos)]
    for idx in range(num_hypos):
        gen_tokens = prev_input_ids[idx].tolist()
        generated_ngram = generated_ngrams[idx]
        # Loop through each n-gram of size ngram_size in the list of tokens (gen_tokens)
        for ngram in zip(*[gen_tokens[i:] for i in range(ngram_size)]):
            prev_ngram_tuple = tuple(ngram[:-1])
            generated_ngram[prev_ngram_tuple] = generated_ngram.get(prev_ngram_tuple, []) + [ngram[-1]]
    return generated_ngrams


def _get_generated_ngrams(banned_ngrams, prev_input_ids, ngram_size, cur_len):
    # Copied from huggingface logits_process.py
    # Before decoding the next token, prevent decoding of ngrams that have already appeared
    start_idx = cur_len + 1 - ngram_size
    ngram_idx = tuple(prev_input_ids[start_idx:cur_len].tolist())
    return banned_ngrams.get(ngram_idx, [])


def _calc_banned_ngram_tokens(ngram_size: int, prev_input_ids: torch.Tensor, num_hypos: int, cur_len: int):
    # Copied from huggingface logits_process.py
    # Copied from fairseq for no_repeat_ngram in beam_search
    if cur_len + 1 < ngram_size:
        # return no banned tokens if we haven't generated no_repeat_ngram_size tokens yet
        return [[] for _ in range(num_hypos)]
    generated_ngrams = _get_ngrams(ngram_size, prev_input_ids, num_hypos)
    banned_tokens = [
        _get_generated_ngrams(generated_ngrams[hypo_idx], prev_input_ids[hypo_idx], ngram_size, cur_len)
        for hypo_idx in range(num_hypos)
    ]
    return banned_tokens


def filter_ngrams(ngram_size, input_ids, next_token_scores, cur_len):
    num_batch_hypotheses = next_token_scores.shape[0]
    banned_batch_tokens = _calc_banned_ngram_tokens(ngram_size, input_ids, num_batch_hypotheses, cur_len)
    for i, banned_tokens in enumerate(banned_batch_tokens):
        next_token_scores[i, banned_tokens] = -float("inf")
    return next_token_scores


def sample_loop_llama(model, input_ids, start_ids, next_token_scores, sequence_length, eos_token_id=2,
                      top_k=50, top_p=1.0, temperature=1.0, streamer=None, stopping_criteria_list=None, ngram_size=None, cache_ids=None):
    validate_top_k_top_p_min_tokens_to_keep(top_k, top_p, None)

    if not isinstance(temperature, float) or not (temperature > 0):
        raise ValueError('temperature has to be a strictly positive float.')

    stopping_criteria_list = stopping_criteria_list if stopping_criteria_list is not None else StoppingCriteriaList()

# Flags, one per sequence in a batch, to indicate if a sequence hit eos_token_id
    done_flags = torch.full((input_ids.size(dim=0), 1), False)
    tokens = [input_ids]
    _, start_length = input_ids.shape
    if cache_ids is None:
        cache_ids = torch.as_tensor([start_length - 1], dtype=torch.int32)
    else:
        start_length = max(cache_ids.flatten()).item() + 1

    for current_length in range(start_length, sequence_length):

        if ngram_size:
            next_token_scores = filter_ngrams(ngram_size, torch.cat(tokens, dim=-1), next_token_scores, current_length)

        if temperature != 1.0:
            next_token_scores /= temperature

        top_values, top_indices = top_k_top_p_filtering(next_token_scores, top_k=top_k, top_p=top_p)

        # sample
        probs = torch.nn.functional.softmax(top_values, dim=-1, dtype=torch.float32)
        inputs_in_topk = torch.multinomial(probs, num_samples=1, replacement=True)
        inputs = torch.gather(top_indices, 1, inputs_in_topk)

        # Update done flags.
        done_flags = torch.logical_or(done_flags, inputs == eos_token_id)
        # Update token id to be eos_token_id if the corresponding done flag is True. For a batch,
        # this means that, while every sequence in the batch has the same length, a sequence that
        # encounters eos_token_id earlier will be filled with eos_token_ids post the first appearance
        # of eos_token_id.

        token = torch.where(done_flags.eq(True), eos_token_id, inputs)
        tokens.append(token)

        if streamer is not None and hasattr(streamer, 'response_with_prefix') and streamer.response_with_prefix:
             streamer.put(torch.cat(tokens, dim=-1))
        elif streamer:
            streamer.put(token)

        next_length = current_length + 1
        if next_length > sequence_length or done_flags.all():
            break

        if stopping_criteria_list(input_ids, probs):
            break

        # forward pass to get next token
        cache_ids = cache_ids + 1
        
        if model.neuron_config and model.neuron_config.use_2d_cache_ids:
            cache_ids = torch.unsqueeze(cache_ids, dim=0)
        
        next_token_scores = model(inputs, cache_ids, start_ids)

    if streamer:
        streamer.end()

    return torch.cat(tokens, dim=-1)


@torch.no_grad()
def sample_llama(model, input_ids, start_ids, sequence_length, eos_token_id=2, top_k=50, top_p=1.0, temperature=1.0,
                 streamer=None, stopping_criteria_list=None, no_repeat_ngram_size=None, cache_ids=None):
    validate_top_k_top_p_min_tokens_to_keep(top_k, top_p, None)
    
    # populate key/value caches according to the prompt text
    next_token_scores = model(input_ids, cache_ids, start_ids)
    if cache_ids is not None and model.neuron_config.use_2d_cache_ids:
        cache_ids = cache_ids.max(dim=1, keepdim=True).values
    if model.context_hook is not None:
        model.context_hook()
    return sample_loop_llama(
        model, input_ids, start_ids, next_token_scores, sequence_length, eos_token_id, top_k, top_p, temperature,
        streamer, stopping_criteria_list, ngram_size=no_repeat_ngram_size, cache_ids=cache_ids
    )

#TODO Leverage Generation Args data class as input args
def select_tokens(next_token_scores, top_k=1, top_p=1.0, temperature=1.0):
    top_values, top_indices = top_k_top_p_filtering(next_token_scores, top_k=top_k, top_p=top_p)
 
    # sample
    probs = torch.nn.functional.softmax(top_values, dim=-1)
    inputs_in_topk = torch.multinomial(probs, num_samples=1, replacement=True)
    inputs = torch.gather(top_indices, 1, inputs_in_topk)
    return inputs

