import torch
from abc import ABC
from .beam_search import BeamScorer, BeamSearchScorer, BeamSearchOutput
from .logits_process import MinLengthLogitsProcessor
from typing import Optional, Union, List
from transformers import LogitsProcessorList
import warnings
import copy

# imported hee in transforemrs  https://github.com/huggingface/transformers/blob/v4.26.1/src/transformers/utils/__init__.py 
def add_start_docstrings(*docstr):
    def docstring_decorator(fn):
        fn.__doc__ = "".join(docstr) + (fn.__doc__ if fn.__doc__ is not None else "")
        return fn

    return docstring_decorator

STOPPING_CRITERIA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.
            Indices can be obtained using [`BertTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.
            [What are input IDs?](../glossary#input-ids)
        scores (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`):
            Prediction scores of a language modeling head. These can be scores for each vocabulary token before SoftMax
            or scores for each vocabulary token after SoftMax.
        kwargs:
            Additional stopping criteria specific kwargs.
    Return:
        `bool`. `False` indicates we should continue, `True` indicates we should stop.
"""
class StoppingCriteria(ABC):
    """Abstract base class for all stopping criteria that can be applied during generation."""

    @add_start_docstrings(STOPPING_CRITERIA_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        raise NotImplementedError("StoppingCriteria needs to be subclassed")

class MaxLengthCriteria(StoppingCriteria):
    """
    This class can be used to stop generation whenever the full generated number of tokens exceeds `max_length`. Keep
    in mind for decoder-only type of transformers, this will include the initial prompted tokens.
    Args:
        max_length (`int`):
            The maximum length that the output sequence can have in number of tokens.
    """

    def __init__(self, max_length: int):
        self.max_length = max_length

    @add_start_docstrings(STOPPING_CRITERIA_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        return input_ids.shape[-1] >= self.max_length


class StoppingCriteriaList(list):
    @add_start_docstrings(STOPPING_CRITERIA_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        return any(criteria(input_ids, scores) for criteria in self)

    @property
    def max_length(self) -> Optional[int]:
        for stopping_criterium in self:
            if isinstance(stopping_criterium, MaxLengthCriteria):
                return stopping_criterium.max_length
            elif isinstance(stopping_criterium, MaxNewTokensCriteria):
                return stopping_criterium.max_length
        return None 

def validate_stopping_criteria(stopping_criteria: StoppingCriteriaList, max_length: int) -> StoppingCriteriaList:
    stopping_max_length = stopping_criteria.max_length
    new_stopping_criteria = copy.deepcopy(stopping_criteria)
    if stopping_max_length is not None and stopping_max_length != max_length:
        warnings.warn("You set different `max_length` for stopping criteria and `max_length` parameter", UserWarning)
    elif stopping_max_length is None:
        new_stopping_criteria.append(MaxLengthCriteria(max_length=max_length))
    return new_stopping_criteria
#########

def beam_search_func(
    self,
    input_ids: torch.LongTensor,
    beam_scorer: BeamScorer,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    max_length: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[Union[int, List[int]]] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_scores: Optional[bool] = None,
    return_dict_in_generate: Optional[bool] = None,
    synced_gpus: Optional[bool] = False,
    **model_kwargs,
) -> Union[BeamSearchOutput, torch.LongTensor]:
    # r"""
    # Generates sequences of token ids for models with a language modeling head using **beam search decoding** and
    # can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.
    # <Tip warning={true}>
    # In most cases, you do not need to call [`~generation.GenerationMixin.beam_search`] directly. Use generate()
    # instead. For an overview of generation strategies and code examples, check the [following
    # guide](./generation_strategies).
    # </Tip>
    # Parameters:
    #     input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
    #         The sequence used as a prompt for the generation.
    #     beam_scorer (`BeamScorer`):
    #         An derived instance of [`BeamScorer`] that defines how beam hypotheses are constructed, stored and
    #         sorted during generation. For more information, the documentation of [`BeamScorer`] should be read.
    #     logits_processor (`LogitsProcessorList`, *optional*):
    #         An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
    #         used to modify the prediction scores of the language modeling head applied at each generation step.
    #     stopping_criteria (`StoppingCriteriaList`, *optional*):
    #         An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
    #         used to tell if the generation loop should stop.
    #     max_length (`int`, *optional*, defaults to 20):
    #         **DEPRECATED**. Use `logits_processor` or `stopping_criteria` directly to cap the number of generated
    #         tokens. The maximum length of the sequence to be generated.
    #     pad_token_id (`int`, *optional*):
    #         The id of the *padding* token.
    #     eos_token_id (`int`, *optional*):
    #         The id of the *end-of-sequence* token.
    #     output_attentions (`bool`, *optional*, defaults to `False`):
    #         Whether or not to return the attentions tensors of all attention layers. See `attentions` under
    #         returned tensors for more details.
    #     output_hidden_states (`bool`, *optional*, defaults to `False`):
    #         Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
    #         for more details.
    #     output_scores (`bool`, *optional*, defaults to `False`):
    #         Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
    #     return_dict_in_generate (`bool`, *optional*, defaults to `False`):
    #         Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
    #     synced_gpus (`bool`, *optional*, defaults to `False`):
    #         Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
    #     model_kwargs:
    #         Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
    #         an encoder-decoder model the kwargs should include `encoder_outputs`.
    # Return:
    #     [`generation.BeamSearchDecoderOnlyOutput`], [`~generation.BeamSearchEncoderDecoderOutput`] or
    #     `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
    #     [`~generation.BeamSearchDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
    #     `return_dict_in_generate=True` or a [`~generation.BeamSearchEncoderDecoderOutput`] if
    #     `model.config.is_encoder_decoder=True`.
    # Examples:
    # ```python
    # >>> from transformers import (
    # ...     AutoTokenizer,
    # ...     AutoModelForSeq2SeqLM,
    # ...     LogitsProcessorList,
    # ...     MinLengthLogitsProcessor,
    # ...     BeamSearchScorer,
    # ... )
    # >>> import torch
    # >>> tokenizer = AutoTokenizer.from_pretrained("t5-base")
    # >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
    # >>> encoder_input_str = "translate English to German: How old are you?"
    # >>> encoder_input_ids = tokenizer(encoder_input_str, return_tensors="pt").input_ids
    # >>> # lets run beam search using 3 beams
    # >>> num_beams = 3
    # >>> # define decoder start token ids
    # >>> input_ids = torch.ones((num_beams, 1), device=model.device, dtype=torch.long)
    # >>> input_ids = input_ids * model.config.decoder_start_token_id
    # >>> # add encoder_outputs to model keyword arguments
    # >>> model_kwargs = {
    # ...     "encoder_outputs": model.get_encoder()(
    # ...         encoder_input_ids.repeat_interleave(num_beams, dim=0), return_dict=True
    # ...     )
    # ... }
    # >>> # instantiate beam scorer
    # >>> beam_scorer = BeamSearchScorer(
    # ...     batch_size=1,
    # ...     num_beams=num_beams,
    # ...     device=model.device,
    # ... )
    # >>> # instantiate logits processors
    # >>> logits_processor = LogitsProcessorList(
    # ...     [
    # ...         MinLengthLogitsProcessor(5, eos_token_id=model.config.eos_token_id),
    # ...     ]
    # ... )
    # >>> outputs = model.beam_search(input_ids, beam_scorer, logits_processor=logits_processor, **model_kwargs)
    # >>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
    # ['Wie alt bist du?']
    # ```"""

    # init values
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    if max_length is not None:
        warnings.warn(
            "`max_length` is deprecated in this function, use"
            " `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
            UserWarning,
        )
        stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
    if len(stopping_criteria) == 0:
        warnings.warn("You don't have defined any stopping_criteria, this will likely loop forever", UserWarning)
    import pdb
    # pdb.set_trace()
    pad_token_id = pad_token_id if pad_token_id is not None else 0 #self.config.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    output_scores = output_scores if output_scores is not None else False # self.config.output_scores
    output_attentions = (
        output_attentions if output_attentions is not None else False # self.config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else False # self.config.output_hidden_states
    )

    return_dict_in_generate = False

    batch_size = len(beam_scorer._beam_hyps)
    num_beams = beam_scorer.num_beams

    batch_beam_size, cur_len = input_ids.shape

    if num_beams * batch_size != batch_beam_size:
        raise ValueError(
            f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
        )

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    beam_indices = (
        tuple(() for _ in range(batch_beam_size)) if (return_dict_in_generate and output_scores) else None
    )
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        )

    # initialise score of first beam with 0 and the rest with -1e9. This makes sure that only tokens
    # of the first beam are considered to avoid sampling the exact same tokens across all beams.
    beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
    beam_scores[:, 1:] = -1e9
    beam_scores = beam_scores.view((batch_size * num_beams,))

    this_peer_finished = False  # used by synced_gpus only

    def prepare_inputs_for_generation(input_ids, past_key_values=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        # only last token for inputs_ids if past is defined in kwargs
        if past_key_values:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            # "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            # "attention_mask": attention_mask,
            # "token_type_ids": token_type_ids,
        }

    while True:
        if synced_gpus:
            # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
            # The following logic allows an early break if all peers finished generating their sequence
            this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
            # send 0.0 if we finished, 1.0 otherwise
            dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
            # did all peers finish? the reduced sum will be 0.0 then
            if this_peer_finished_flag.item() == 0.0:
                break


        # model_inputs = input_ids
        import pdb
        # pdb.set_trace()
        _, start = input_ids.shape # torch.Size([6, 8])
        # position_ids
        # tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.int32)
        position_ids = torch.arange(start, dtype=torch.int32)
        next_token_scores = self(input_ids, position_ids)

        from torch import nn

        next_token_scores = nn.functional.log_softmax(next_token_scores, dim=-1)

        next_token_scores_processed = logits_processor(input_ids, next_token_scores)
        next_token_scores = next_token_scores_processed + beam_scores[:, None].expand_as(next_token_scores)

        # reshape for beam search
        vocab_size = next_token_scores.shape[-1]
        # pdb.set_trace()
        # issue 
        next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size) # *** RuntimeError: view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.

        # Sample 2 next tokens for each beam (so we have some spare tokens and match output of beam search)
        next_token_scores, next_tokens = torch.topk(next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True)
        # Dimension out of range (expected to be in range of [-1, 0], but got 1) 

        ##
        next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor") # issues *** NameError: name 'torch_int_div' is not defined
        next_tokens = next_tokens % vocab_size

        # stateless
        beam_outputs = beam_scorer.process(input_ids, next_token_scores, next_tokens, next_indices, pad_token_id=pad_token_id, eos_token_id=eos_token_id, beam_indices=beam_indices)

        beam_scores = beam_outputs["next_beam_scores"]
        beam_next_tokens = beam_outputs["next_beam_tokens"]
        beam_idx = beam_outputs["next_beam_indices"]

        input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)

        # increase cur_len
        cur_len = cur_len + 1

        if beam_scorer.is_done or stopping_criteria(input_ids, scores):
            if not synced_gpus:
                break
            else:
                this_peer_finished = True

    sequence_outputs = beam_scorer.finalize(
        input_ids,
        beam_scores,
        next_tokens,
        next_indices,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        max_length=stopping_criteria.max_length,
        beam_indices=beam_indices,
    )

    return sequence_outputs["sequences"]

