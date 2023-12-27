import warnings
import torch

from transformers import PreTrainedModel
from transformers.utils import ModelOutput

"""
An adapter class for HuggingFace Generation API compatibility.

It requires model to have a forward interface as:

    forward(input_ids: Tensor(batch_size, seq_len), cache_ids: Tensor(seq_len), 
        start_ids: Optional[Tensor(batch_size)])) -> Tensor(batch_size, vocab_size)
"""
class HuggingFaceGenerationModelAdapter(PreTrainedModel):
    
    def __init__(self, config, model):
        super().__init__(config)
        self.model = model
        self.config = config
        self.cur_len = 0

    def reset_generation(self):
        self.cur_len = 0

    def forward(self, input_ids, cache_ids, start_ids=None, output_hidden_states=False, output_attentions=False,
            attention_mask=None, return_dict=False):
        
        if  output_hidden_states or output_attentions or attention_mask is not None:
            warnings.warn("Warning: These arguments are not used by forward(): \
                (output_hidden_states, output_attentions, attention_mask)")

        out_logits = self.model(input_ids, cache_ids, start_ids)

        out_logits = out_logits[:, None, :]
        if return_dict:
            return ModelOutput(
                [("logits", out_logits), ("past_key_values", tuple())],
            )
        return (out_logits,)
    
    # keep the generation stateless
    def generate(self, *args, **kwargs):
        self.reset_generation()
        return super().generate(*args, **kwargs)

    # implemented for beam search
    # we ignore past as we don't expose k/v_cache
    def _reorder_cache(self, past, beam_idx):
        assert hasattr(self.model, 'reorder_cache') and callable(self.model.reorder_cache), f"{self.model.__class__.__name__} doesn't have reorder_cache implemented for beam search"
        self.model.reorder_cache(beam_idx)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        # convert attention_mask to start_ids
        attention_mask = None
        cache_ids = None
        start_ids = None
        if "attention_mask" in kwargs:
            attention_mask = kwargs["attention_mask"]

        if attention_mask is not None:
            _, start_ids = attention_mask.max(axis=1)


        if self.cur_len > 0:
            input_ids = input_ids[:, -1:]
            cache_ids = torch.as_tensor([self.cur_len], dtype=torch.int32)
        
        # no need to prepare cache_ids for parallel context encoding here as forward will pad input_ids and generate legalized cache_ids
        
        self.cur_len += input_ids.shape[-1] 
        model_inputs = {
            "input_ids": input_ids,
            "cache_ids": cache_ids,
            "start_ids": start_ids,
        }

        return model_inputs
