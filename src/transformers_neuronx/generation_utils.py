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
        # TODO (bowencc): Need to further verify the behavior of attention_mask and position_ids under beam search, comment out for now
        # batch_dim = input_ids.shape[0]
        # batch_size = self.config.batch_size
        # if batch_dim != batch_size:
        #     if batch_dim < batch_size or batch_dim % batch_size != 0:
        #         raise ValueError(f"batch dimension on input_ids ({batch_dim}) is not compatible with model's compiled batch_size ({batch_size})")
        #     input_ids_splits = input_ids.reshape(batch_size, batch_dim//batch_size, -1).transpose(1, 0).split(1, dim=0)
        #     out_logits = []
        #     # iterate per beam
        #     for split in input_ids_splits:
        #         out = self._forward(split.squeeze(0), cache_ids, start_ids)
        #         out_logits.append(out)
        #     out_logits = torch.cat(out_logits, dim=1).reshape(batch_dim, 1, -1)
        # else:
        out_logits = self.model.forward(input_ids, cache_ids, start_ids)
        out_logits = out_logits[:, None, :]
        if return_dict:
            return ModelOutput(
                [("logits", out_logits)]
            )
        return (out_logits,)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        # convert attention_mask to start_ids
        attention_mask = None
        start_ids = None
        if "attention_mask" in kwargs:
            attention_mask = kwargs["attention_mask"]

        if attention_mask is not None:
            _, start_ids = attention_mask.max(axis=1)

        if self.cur_len > 0:
            input_ids = input_ids[:, -1:]
            cache_ids = torch.as_tensor([self.cur_len], dtype=torch.int32)
        else:
            cache_ids = torch.arange(input_ids.shape[-1], dtype=torch.int32)

        self.cur_len += input_ids.shape[-1]
        model_inputs = {
            "input_ids": input_ids,
            "cache_ids": cache_ids,
            "start_ids": start_ids,
        }

        return model_inputs
