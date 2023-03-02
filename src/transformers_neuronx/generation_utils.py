
import torch

class GenerationMixin:
    
    @torch.no_grad()
    def greedy_search(model, input_ids, sequence_length, eos_token_id=2):
        # TODO(bowencc): move this reset to higher level when generate is available 
        model.reset() 

        _, start = input_ids.shape
        position_ids = torch.arange(start, dtype=torch.int32)
        next_token_scores = model(input_ids, position_ids)

        tokens = [input_ids]
        _, start = input_ids.shape
        for cur_len in range(start, sequence_length):
            next_len = cur_len + 1

            # don't sample EOS
            next_token_scores[:, eos_token_id] = -float('inf')

            # Remove all tokens with a probability less than the last token of the top-k
            max_index = torch.argmax(next_token_scores, dim=-1, keepdim=True)
            tokens.append(max_index)
            # forward pass to get next token
            position_ids = torch.as_tensor([cur_len], dtype=torch.int32)
            next_token_scores = model(max_index, position_ids)
        return torch.cat(tokens, dim=-1)