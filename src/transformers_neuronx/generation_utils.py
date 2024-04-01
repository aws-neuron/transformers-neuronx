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
        # sdpa attension is currently unsupported. Torch >=2.1.1 sets _attn_implementation to sdpa causing failures.
        if hasattr(config, "_attn_implementation") and config._attn_implementation == "sdpa":
            warnings.warn("Warning: sdpa is unsupported as part of attention implementation. Falling back to eager attention implementation.")
            config._attn_implementation = "eager"
        super().__init__(config)
        self.model = model
        self.config = config
        self.cur_len = torch.zeros(1, dtype=torch.long)

    def reset_generation(self):
        self.cur_len = torch.zeros(1, dtype=torch.long)

    def forward(self, input_ids, cache_ids, start_ids=None, output_hidden_states=False, output_attentions=False,
            attention_mask=None, return_dict=False):

        if  output_hidden_states or output_attentions or attention_mask is not None:
            warnings.warn("Warning: These arguments are not used by forward(): \
                (output_hidden_states, output_attentions, attention_mask)")

        if self.model.neuron_config.is_pp():
            out_logits = self.model.pp_forward(input_ids, cache_ids, start_ids)
        else:
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
            start_ids = start_ids.int()

        if (self.cur_len > 0).any().item():
            input_ids = input_ids[:, -1:]

        if self.model.neuron_config.use_2d_cache_ids:
            # 2D cache_ids
            batch_size, context_length = attention_mask.shape
            if (self.cur_len > 0).any().item():
                # token generation (aka decoding) with 2D cache_ids
                index_map = torch.arange(context_length).unsqueeze(0).expand(batch_size, context_length)
                cache_ids = (index_map * attention_mask).max(dim=1).values.unsqueeze(-1)
                self.cur_len = cache_ids.squeeze(-1)
            else:
                # context encoding (aka prefill) with 2D cache_ids
                cache_ids = torch.arange(context_length) * attention_mask
                self.cur_len = cache_ids.max(dim=1).values
        else:
            if (self.cur_len > 0).any().item():
                # token generation (aka decoding) with 1D cache_ids
                cache_ids = self.cur_len
                self.cur_len = cache_ids + 1
            else:
                # context encoding (aka prefill) with 1D cache_ids
                batch_size, context_length = input_ids.shape
                cache_ids = torch.arange(context_length)
                self.cur_len = torch.tensor([context_length], dtype=torch.long)

        if self.model.neuron_config.continuous_batching:
            if (self.cur_len > 0).any().item():
                start_ids = None
            else:
                start_ids = torch.arange(input_ids.shape[0])

        model_inputs = {
            "input_ids": input_ids,
            "cache_ids": cache_ids,
            "start_ids": start_ids,
        }

        return model_inputs
