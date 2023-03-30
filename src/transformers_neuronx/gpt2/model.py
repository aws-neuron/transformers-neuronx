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
import os
import warnings
import torch
from transformers import PreTrainedModel
from transformers.utils import ModelOutput
from transformers_neuronx import dtypes
from transformers_neuronx import hlo
from transformers_neuronx import module
from transformers_neuronx import ops
from transformers_neuronx import parallel
from transformers_neuronx import sampling
from transformers_neuronx import utils
from transformers_neuronx.decoder import DecoderLmHeadForSamplingNoEmbedding
from transformers_neuronx.gpt2.config import GPT2Config, GPT2HuggingFaceConfig
from transformers_neuronx.opt.model import OPTForSamplingNoEmbeddingHlo


class GPT2ForSampling(module.WrappingCheckpointCompatibleModel):

    def __init__(self, config, batch_size=1, amp='f32', tp_degree=2,
                 unroll=None, init_n_active_tokens=None, **kwargs):
        config = GPT2Config(config, batch_size, amp, tp_degree, **kwargs)
        super().__init__(GPT2CheckpointCompatible, config)
        self.config = config
        # Check if input sequence length is allowed given position embedding dimensions
        sequence_length = kwargs.get("n_positions", None)
        if sequence_length:
            max_allowed_sequence_length = config.n_ctx
            if sequence_length > max_allowed_sequence_length:
                raise ValueError(f"Sequence length ({sequence_length}) cannot be larger than position embedding's context size ({max_allowed_sequence_length})!")
        if unroll is None:
            unroll = config.n_layer
        n_positions_list = utils.power_of_two_bucket_sizes(128, config.n_positions)
        attention_head_size = config.n_embd // config.n_head
        self.decoder_lm_head = DecoderLmHeadForSamplingNoEmbedding(
            tp_degree, n_positions_list, 1, batch_size, attention_head_size, amp,
            config.n_layer, unroll,
        )
        hlo_builder = OPTForSamplingNoEmbeddingHlo(tp_degree, config.n_embd, 'gelu_new')
        self.decoder_lm_head.add_inputs_builder(hlo_builder.inputs)
        self.decoder_lm_head.add_layer_builder(hlo_builder.layer)
        self.decoder_lm_head.add_ln_lm_head_builder(hlo_builder.ln_lm_head)

    def _save_compiled_artifacts(self, directory):
        if os.path.isfile(directory):
            raise FileExistsError(
                f'Artifacts should be saved to a directory. '
                f'Found existing file: {directory}'
            )
        os.makedirs(directory, exist_ok=True)
        self.decoder_lm_head.save_compiler_artifacts(os.path.join(directory, 'neuron-program.pkl'))

    def _load_compiled_artifacts(self, directory):
        if not os.path.isdir(directory):
            raise FileNotFoundError(f'Did not find directory: {directory}')
        program_filename = os.path.join(directory, 'neuron-program.pkl')
        if os.path.exists(program_filename):
            self.decoder_lm_head.load_compiler_artifacts_after_build(program_filename)

    def to_neuron(self):
        ops.init()
        self.chkpt_model.transformer.wte.materialize()
        self.chkpt_model.transformer.wpe.materialize()
        n_embd = self.config.n_embd
        for layer in self.chkpt_model.transformer.h:
            layer.materialize()
            attn = layer.attn
            mlp = layer.mlp
            c_attn_weight = attn.c_attn.weight.detach()
            c_attn_bias = attn.c_attn.bias.detach()
            new_layer = self.decoder_lm_head.new_layer()
            new_layer.add_pre_attention_layer_norm(layer.ln_1.weight.detach(),
                                                   layer.ln_1.bias.detach())
            new_layer.add_attention_query(c_attn_weight[:, :n_embd], c_attn_bias[:n_embd])
            new_layer.add_attention_key(c_attn_weight[:, n_embd:n_embd*2],
                                        c_attn_bias[n_embd:n_embd*2])
            new_layer.add_attention_value(c_attn_weight[:, n_embd*2:n_embd*3],
                                          c_attn_bias[n_embd*2:n_embd*3])
            new_layer.add_attention_output(attn.c_proj.weight.detach(), attn.c_proj.bias.detach())
            new_layer.add_pre_mlp_layer_norm(layer.ln_2.weight.detach(), layer.ln_2.bias.detach())
            new_layer.add_mlp_input(mlp.c_fc.weight.detach(), mlp.c_fc.bias.detach())
            new_layer.add_mlp_output(mlp.c_proj.weight.detach(), mlp.c_proj.bias.detach())
            new_layer.to_neuron()
            layer.nullify()
        ln_f = self.chkpt_model.transformer.ln_f
        ln_f.materialize()
        self.decoder_lm_head.add_final_layer_norm(ln_f.weight.detach(), ln_f.bias.detach())
        lm_head = self.chkpt_model.lm_head
        lm_head.materialize()
        self.decoder_lm_head.add_lm_head(lm_head.weight.detach().T)
        lm_head.nullify()
        self.decoder_lm_head.to_neuron()

    def reset(self):
        self.decoder_lm_head.reset()

    def forward(self, input_ids, cache_ids, start_ids=None):
        inputs_embeds = self.chkpt_model.transformer.wte(input_ids)
        position_ids, start_ids = self.decoder_lm_head.embed_positions_ids(cache_ids, start_ids)
        position_embeds = self.chkpt_model.transformer.wpe(position_ids)
        hidden = inputs_embeds + position_embeds
        hidden = hidden.transpose(0, -1)
        logits = self.decoder_lm_head(hidden, cache_ids, start_ids)
        logits = logits.to(torch.float32)
        logits = logits[:self.config.vocab_size]
        logits = logits.transpose(0, -1)
        logits = logits[:, -1, :]
        return logits

    def sample(self, input_ids, sequence_length, start_ids=None, top_k=50):
        return sampling.simple_sample(self, input_ids, start_ids, sequence_length,
                                      eos_token_id=self.config.eos_token_id, top_k=top_k)

# (bowencc): Need to keep PreTrainedModel after module.PretrainedModel as the later
# overrides from_pretrained methods. Cannot use module.WrappingCheckpointCompatibleModel directly 
# since it doesn't pass config in suer().__init__
class GPT2ForHuggingFaceSampling(module.PretrainedModel, PreTrainedModel):
    def __init__(self, config, batch_size=1, amp='f32', tp_degree=2,
                 unroll=None, init_n_active_tokens=None, **kwargs):
        config = GPT2HuggingFaceConfig(config, batch_size, amp, tp_degree, **kwargs)
        super().__init__(config) # will call transformers.PreTrainedModel(confg)
        self.chkpt_model = GPT2CheckpointCompatible(config)
        self.config = config

        # Check if input sequence length is allowed given position embedding dimensions
        sequence_length = kwargs.get("n_positions", None)
        if sequence_length:
            max_allowed_sequence_length = config.n_ctx
            if sequence_length > max_allowed_sequence_length:
                raise ValueError(f"Sequence length ({sequence_length}) cannot be larger than position embedding's context size ({max_allowed_sequence_length})!")

        if unroll is None:
            unroll = config.n_layer
        n_positions_list = utils.power_of_two_bucket_sizes(128, config.n_positions)
        attention_head_size = config.n_embd // config.n_head
        self.decoder_lm_head = DecoderLmHeadForSamplingNoEmbedding(
            tp_degree, n_positions_list, 1, batch_size, attention_head_size, amp,
            config.n_layer, unroll,
        )
        hlo_builder = OPTForSamplingNoEmbeddingHlo(tp_degree, config.n_embd, 'gelu_new')
        self.decoder_lm_head.add_inputs_builder(hlo_builder.inputs)
        self.decoder_lm_head.add_layer_builder(hlo_builder.layer)
        self.decoder_lm_head.add_ln_lm_head_builder(hlo_builder.ln_lm_head)
        self.cur_len = 0

    def _save_compiled_artifacts(self, directory):
        if os.path.isfile(directory):
            raise FileExistsError(
                f'Artifacts should be saved to a directory. '
                f'Found existing file: {directory}'
            )
        os.makedirs(directory, exist_ok=True)
        self.decoder_lm_head.save_compiler_artifacts(os.path.join(directory, 'neuron-program.pkl'))

    def _load_compiled_artifacts(self, directory):
        if not os.path.isdir(directory):
            raise FileNotFoundError(f'Did not find directory: {directory}')
        program_filename = os.path.join(directory, 'neuron-program.pkl')
        if os.path.exists(program_filename):
            self.decoder_lm_head.load_compiler_artifacts_after_build(program_filename)

    def load_state_dict_dir(self, state_dict_dir):
        self.chkpt_model.load_state_dict_dir(state_dict_dir)

    def load_state_dict_low_memory(self, state_dict):
        self.chkpt_model.load_state_dict_low_memory(state_dict)

    def to_neuron(self):
        ops.init()
        self.chkpt_model.transformer.wte.materialize()
        self.chkpt_model.transformer.wpe.materialize()
        n_embd = self.config.n_embd
        for layer in self.chkpt_model.transformer.h:
            layer.materialize()
            attn = layer.attn
            mlp = layer.mlp
            c_attn_weight = attn.c_attn.weight.detach()
            c_attn_bias = attn.c_attn.bias.detach()
            new_layer = self.decoder_lm_head.new_layer()
            new_layer.add_pre_attention_layer_norm(layer.ln_1.weight.detach(),
                                                   layer.ln_1.bias.detach())
            new_layer.add_attention_query(c_attn_weight[:, :n_embd], c_attn_bias[:n_embd])
            new_layer.add_attention_key(c_attn_weight[:, n_embd:n_embd*2],
                                        c_attn_bias[n_embd:n_embd*2])
            new_layer.add_attention_value(c_attn_weight[:, n_embd*2:n_embd*3],
                                          c_attn_bias[n_embd*2:n_embd*3])
            new_layer.add_attention_output(attn.c_proj.weight.detach(), attn.c_proj.bias.detach())
            new_layer.add_pre_mlp_layer_norm(layer.ln_2.weight.detach(), layer.ln_2.bias.detach())
            new_layer.add_mlp_input(mlp.c_fc.weight.detach(), mlp.c_fc.bias.detach())
            new_layer.add_mlp_output(mlp.c_proj.weight.detach(), mlp.c_proj.bias.detach())
            new_layer.to_neuron()
            layer.nullify()
        ln_f = self.chkpt_model.transformer.ln_f
        ln_f.materialize()
        self.decoder_lm_head.add_final_layer_norm(ln_f.weight.detach(), ln_f.bias.detach())
        lm_head = self.chkpt_model.lm_head
        lm_head.materialize()
        self.decoder_lm_head.add_lm_head(lm_head.weight.detach().T)
        lm_head.nullify()
        self.decoder_lm_head.to_neuron()

    def reset(self):
        # self.decoder_lm_head.reset()
        self.cur_len = 0

    def _forward(self, input_ids, cache_ids, start_ids=None):
        inputs_embeds = self.chkpt_model.transformer.wte(input_ids)
        position_ids, start_ids = self.decoder_lm_head.embed_positions_ids(cache_ids, start_ids)
        position_embeds = self.chkpt_model.transformer.wpe(position_ids)
        hidden = inputs_embeds + position_embeds
        hidden = hidden.transpose(0, -1)
        logits = self.decoder_lm_head(hidden, cache_ids, start_ids)
        logits = logits.to(torch.float32)
        logits = logits[:self.config.vocab_size]
        logits = logits.transpose(0, -1)
        return logits

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
        out_logits = self._forward(input_ids, cache_ids, start_ids)
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


class GPT2ForSamplingWithContextBroadcasting(module.WrappingCheckpointCompatibleModel):

    def __init__(self, config, batch_size=1, amp='f32', tp_degree=2, context_length_estimate=None,
                 unroll=None, **kwargs):
        config = GPT2Config(config, batch_size, amp, tp_degree, **kwargs)
        super().__init__(GPT2CheckpointCompatible, config)
        self.config = config
        if unroll is None:
            unroll = config.n_layer
        attention_head_size = config.n_embd // config.n_head
        if context_length_estimate is None:
            context_length_estimate = config.n_positions // 2
        self.context_length_estimate = context_length_estimate
        n_positions_list = [config.n_positions]
        self.decoder_lm_head = DecoderLmHeadForSamplingNoEmbedding(
            tp_degree, n_positions_list, 1, batch_size, attention_head_size, amp,
            config.n_layer, unroll,
        )
        hlo_builder = OPTForSamplingNoEmbeddingHlo(tp_degree, config.n_embd, 'gelu_new')
        self.decoder_lm_head.add_inputs_builder(hlo_builder.inputs)
        self.decoder_lm_head.add_layer_builder(hlo_builder.layer)
        self.decoder_lm_head.add_ln_lm_head_builder(hlo_builder.ln_lm_head)
        self.decoder_lm_head_for_context = None
        self.context_pre_hook = None
        self.context_hook = None

    def to_neuron(self):
        ops.init()
        self.chkpt_model.transformer.wte.materialize()
        self.chkpt_model.transformer.wpe.materialize()
        n_embd = self.config.n_embd
        for layer in self.chkpt_model.transformer.h:
            layer.materialize()
            attn = layer.attn
            mlp = layer.mlp
            c_attn_weight = attn.c_attn.weight.detach()
            c_attn_bias = attn.c_attn.bias.detach()
            new_layer = self.decoder_lm_head.new_layer()
            new_layer.add_pre_attention_layer_norm(layer.ln_1.weight.detach(),
                                                   layer.ln_1.bias.detach())
            new_layer.add_attention_query(c_attn_weight[:, :n_embd], c_attn_bias[:n_embd])
            new_layer.add_attention_key(c_attn_weight[:, n_embd:n_embd*2],
                                        c_attn_bias[n_embd:n_embd*2])
            new_layer.add_attention_value(c_attn_weight[:, n_embd*2:n_embd*3],
                                          c_attn_bias[n_embd*2:n_embd*3])
            new_layer.add_attention_output(attn.c_proj.weight.detach(), attn.c_proj.bias.detach())
            new_layer.add_pre_mlp_layer_norm(layer.ln_2.weight.detach(), layer.ln_2.bias.detach())
            new_layer.add_mlp_input(mlp.c_fc.weight.detach(), mlp.c_fc.bias.detach())
            new_layer.add_mlp_output(mlp.c_proj.weight.detach(), mlp.c_proj.bias.detach())
            new_layer.to_neuron()
            layer.nullify()
        ln_f = self.chkpt_model.transformer.ln_f
        ln_f.materialize()
        self.decoder_lm_head.add_final_layer_norm(ln_f.weight.detach(), ln_f.bias.detach())
        lm_head = self.chkpt_model.lm_head
        lm_head.materialize()
        self.decoder_lm_head.add_lm_head(lm_head.weight.detach().T)
        lm_head.nullify()
        self.decoder_lm_head.to_neuron()
        self.decoder_lm_head_for_context = self.decoder_lm_head.build_weight_shared(
            n_positions_list=[self.context_length_estimate],
            n_active_tokens=self.context_length_estimate,
            batch_size=1,
            unroll=1,
        )

    def reset(self):
        self.decoder_lm_head.reset()
        self.decoder_lm_head_for_context.reset()

    def forward(self, input_ids, cache_ids, start_ids=None):
        return self._forward(self.decoder_lm_head, input_ids, cache_ids, start_ids)

    def forward_for_context(self, input_ids, cache_ids, start_ids=None):
        return self._forward(self.decoder_lm_head_for_context, input_ids, cache_ids, start_ids)

    def _forward(self, decoder_lm_head, input_ids, cache_ids, start_ids):
        batch_size = input_ids.shape[0]
        inputs_embeds = self.chkpt_model.transformer.wte(input_ids)
        position_ids, start_ids = self.decoder_lm_head.embed_positions_ids(cache_ids, start_ids, batch_size=batch_size)
        position_embeds = self.chkpt_model.transformer.wpe(position_ids)
        hidden = inputs_embeds + position_embeds
        hidden = hidden.transpose(0, -1)
        logits = decoder_lm_head(hidden, cache_ids, start_ids)
        logits = logits.to(torch.float32)
        logits = logits[:self.config.vocab_size]
        logits = logits.transpose(0, -1)
        logits = logits[:, -1, :]
        return logits

    @torch.no_grad()
    def sample(self, input_ids, sequence_length, start_ids=None, top_k=50):
        if self.context_pre_hook is not None:
            self.context_pre_hook()
        broadcaster = CacheBroadcaster(self.decoder_lm_head.tp_degree, shard_dim=2, batch_dim=1,
                                       batch_size=self.decoder_lm_head.batch_size)
        self.reset()
        _, start = input_ids.shape

        # populate key/value caches according to the prompt text
        context_length = self.context_length_estimate
        cache_ids = torch.arange(context_length, dtype=torch.int32)
        input_context = input_ids[:, :context_length]
        if start < context_length:
            input_pad = context_length - start
            input_context = torch.nn.functional.pad(input_context, (0, input_pad, 0, 0))
        next_token_scores = self.forward_for_context(input_context, cache_ids, start_ids)
        for source, target in zip(self.decoder_lm_head_for_context.layers, self.decoder_lm_head.layers):
            broadcaster.broadcast(source.attn_k_cache, target.attn_k_cache, context_length)
            broadcaster.broadcast(source.attn_v_cache, target.attn_v_cache, context_length)
        # if input_ids already have batch size, don't copy
        if input_ids.shape[0] != self.decoder_lm_head.batch_size:
            input_ids = input_ids.repeat([self.decoder_lm_head.batch_size, 1])
        for cur_len in range(context_length, start):
            cache_ids = torch.as_tensor([cur_len], dtype=torch.int32)
            next_token_scores = self(input_ids[:, cur_len:cur_len+1], cache_ids)
        if self.context_hook is not None:
            self.context_hook()
        # if next_token_scores already have batch size, don't copy
        if next_token_scores.shape[0] != self.decoder_lm_head.batch_size:
            next_token_scores = next_token_scores.repeat([self.decoder_lm_head.batch_size, 1])
        return sampling.sample_loop(self, input_ids, start_ids, next_token_scores, sequence_length,
                                    eos_token_id=self.config.eos_token_id, top_k=top_k)


class CacheBroadcaster:

    def __init__(self, tp_degree, shard_dim, batch_dim, batch_size):
        self.manipulator = parallel.ParallelTensorManipulator(tp_degree)
        self.shard_dim = shard_dim
        self.batch_dim = batch_dim
        self.batch_size = batch_size

    def broadcast(self, source, target, context_length):
        source = self.manipulator.unshard_along(source, dim=self.shard_dim)
        source[context_length:] = 0.0
        repeats = [1 for _ in source.shape]
        repeats[self.batch_dim] = self.batch_size
        source = source.repeat(repeats)
        source = self.manipulator.shard_along_on_cpu(source, dim=self.shard_dim)
        ops.parallel_write(target, source)


class GPT2CheckpointCompatible(module.PretrainedModel):

    def __init__(self, config):
        super().__init__()
        dtype, _, _ = utils.parse_amp(config.amp)
        dtype = dtypes.to_torch_dtype(dtype)
        self.transformer = GPT2Transformer(config)
        self.lm_head = module.LowMemoryLazyLinear(config.vocab_size, dtype=dtype, bias=False)


class GPT2Transformer(module.LowMemoryModule):

    def __init__(self, config):
        super().__init__()
        self.wte = module.LowMemoryEmbedding(config.vocab_size, config.n_embd)
        self.wpe = module.LowMemoryEmbedding(config.n_ctx, config.n_embd)
        self.h = module.LowMemoryModuleList()
        for _ in range(config.n_layer):
            self.h.append(GPT2Block(config))
        self.ln_f = module.LowMemoryLayerNorm(config.n_embd)


class GPT2Block(module.LowMemoryModule):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = module.LowMemoryLayerNorm(config.n_embd)
        self.attn = GPT2Attention(config)
        self.ln_2 = module.LowMemoryLayerNorm(config.n_embd)
        self.mlp = GPT2MLP(config)


class GPT2Attention(module.LowMemoryModule):

    def __init__(self, config):
        super().__init__()
        n_embd = config.n_embd
        dtype, _, _ = utils.parse_amp(config.amp)
        dtype = dtypes.to_torch_dtype(dtype)
        self.c_attn = module.LowMemoryLazyLinear(n_embd, dtype=dtype)
        self.c_proj = module.LowMemoryLazyLinear(n_embd, dtype=dtype)


class GPT2MLP(module.LowMemoryModule):

    def __init__(self, config):
        super().__init__()
        dtype, _, _ = utils.parse_amp(config.amp)
        dtype = dtypes.to_torch_dtype(dtype)
        self.c_fc = module.LowMemoryLazyLinear(config.n_embd, dtype=dtype)
        self.c_proj = module.LowMemoryLazyLinear(config.intermediate_dim, dtype=dtype)