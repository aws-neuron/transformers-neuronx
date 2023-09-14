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
import torch
import os
from transformers_neuronx import decoder
from transformers_neuronx import module
from transformers_neuronx import ops
from transformers_neuronx import sampling
from transformers_neuronx import utils
from transformers_neuronx import bucket
from transformers_neuronx import base
from transformers_neuronx.llama.config import LlamaConfig
from transformers_neuronx.llama.modules import LlamaForCausalLM
from transformers_neuronx.llama.hlo import LlamaForSamplingNoEmbeddingHlo
from transformers_neuronx.llama.model import LlamaForSampling


class AEMLlamaForSampling(LlamaForSampling):

    def __init__(self, config, *, n_positions=2048, batch_size=1, amp='f32', tp_degree=2,
                 context_length_estimate=None, context_unroll=None, unroll=None,
                 neuron_config=None, aem_models=None, **kwargs):
        super().__init__(config, n_positions=n_positions, batch_size=batch_size, amp=amp,
                        tp_degree=tp_degree, context_length_estimate=context_length_estimate,
                        context_unroll=context_unroll, unroll=unroll, neuron_config=neuron_config,
                        **kwargs)
        self.aem_models = aem_models

    def run_aems(hidden):
        # Run AEM LLMs in parallel  <<---------------
        return hidden
    
    def lmhead(hidden):
        # Run LLM HEAD  <<---------------
        return hidden
        
    def forward(self, input_ids, cache_ids=None, start_ids=None):

        batch_size, context_length = input_ids.shape
        if start_ids is None:
            start_ids = torch.zeros(batch_size, dtype=torch.int32)
        if cache_ids is None:
            cache_ids = torch.arange(context_length, dtype=torch.int32)

        hidden = self.chkpt_model.model.embed_tokens(input_ids)
        hidden = hidden.transpose(0, -1).contiguous()

        if context_length > 1:
            hidden = self.context(hidden, cache_ids, start_ids)
        else:
            hidden = self.decoder_lm_head(hidden, cache_ids, start_ids)

        # Apply AEMs to hidden here <<----------------------
        aems = self.run_aems(hidden)
        logits = self.lmhead(hidden)

        # We need to also apply Llama's LLM head as well

        logits = logits.to(torch.float32)
        logits = logits[:self.config.vocab_size, -1, :]
        logits = logits.transpose(0, 1)
        return logits


    def sample(self, input_ids, sequence_length, start_ids=None,
               top_k=50, top_p=1.0, eos_token_override=None, temperature=1.0, streamer=None):

        # To enable optimized context encoding network, we must pad
        # up to the context length estimate or we will not correctly
        # select the final context logits (See: layers/transformer.py).
        # This also means we need to shift the start_ids over to correct
        # for padding.
        offset = 0
        batch_size, context_length = input_ids.shape
        estimate = bucket.find(self.context_buckets, context_length)
        if estimate:
            if context_length < estimate:
                input_ids = utils.pad(input_ids, 1, estimate, left=True)
                offset = estimate - context_length
                if start_ids is None:
                    start_ids = torch.zeros(batch_size, dtype=torch.int32)
                start_ids += offset
                sequence_length += offset
                # Sequence length cannot be greater than n_positions
                sequence_length = min(sequence_length, self.max_positions)

        # Change sample_llama in sampling.py to use AEM LLMs <------------------
        result = sampling.sample_llama(
            self, input_ids, start_ids, sequence_length,
            eos_token_id=self.config.eos_token_id if eos_token_override is None else eos_token_override,
            top_k=top_k, top_p=top_p, temperature=temperature, streamer=streamer
        )

        if offset != 0:
            result = result[:, offset:]
        return result