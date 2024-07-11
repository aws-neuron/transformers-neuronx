import torch

from transformers_neuronx import base
from transformers_neuronx import bucket
from transformers_neuronx import decoder
from transformers_neuronx import ops
from transformers_neuronx import sampling
from transformers_neuronx.config import NeuronConfig
from transformers_neuronx.constants import LAYOUT_HSB
from transformers_neuronx.starcoder2.config import Starcoder2Config
from transformers_neuronx.starcoder2.hlo import Starcoder2ForSamplingNoEmbeddingHlo
from transformers_neuronx.starcoder2.modules import Starcoder2ForCausalLM


class Starcoder2ForSampling(base.NeuronModelBase):

    def __init__(self, config, *, n_positions=2048, batch_size=1, amp='f32', tp_degree=2,
                 context_length_estimate=None, context_unroll=None, unroll=None,
                 neuron_config=None, prefixed_length=0, **kwargs):
        config = Starcoder2Config(config, n_positions, batch_size, amp, tp_degree)
        super().__init__(Starcoder2ForCausalLM, config)
        self.context_pre_hook = None
        self.context_hook = None
        self.config = config
        self.neuron_config = neuron_config if neuron_config else NeuronConfig()
        if self.neuron_config.on_device_generation:
            self.neuron_config.on_device_generation.vocab_size = self.config.vocab_size

        self.layers_after_partition = self.neuron_config.auto_layer_partition(config.num_hidden_layers)
        self.prefixed_length = prefixed_length

        if context_unroll is None:
            context_unroll = len(self.layers_after_partition)
        self.context_unroll = context_unroll

        if unroll is None:
            unroll = len(self.layers_after_partition)
        self.unroll = unroll

        self.token_buckets = bucket.token_sizes(n_positions)
        self.context_buckets = bucket.context_sizes(context_length_estimate, self.token_buckets)
        self.window_context_buckets = []
        if prefixed_length:
            if prefixed_length not in self.context_buckets:
                self.context_buckets.append(prefixed_length)
                self.context_buckets = sorted(self.context_buckets)

        self.batch_sizes = bucket.batch_sizes(batch_size)
        self.context_batch_sizes = [
            1] if self.neuron_config and self.neuron_config.continuous_batching else self.batch_sizes
        hlo_builder = Starcoder2ForSamplingNoEmbeddingHlo(config, neuron_config=self.neuron_config)
        self.decoder_param_set = decoder.DecoderLmHeadForSamplingNoEmbedding(
            tp_degree=tp_degree, n_positions_list=self.token_buckets, n_active_tokens=1, batch_size=self.batch_sizes,
            attention_head_size=config.attention_head_size, amp=amp,
            num_layers=len(self.layers_after_partition), n_head=config.num_attention_heads,
            n_kv_head=config.num_key_value_heads,
            unroll=unroll, neuron_config=self.neuron_config, allow_pad=True,
            builder=hlo_builder
        )
        self.decoder_lm_head = self.decoder_param_set.init_token_decoder(unroll=self.unroll, buckets=self.token_buckets,
                                                                         model_obj=self)
        self.decoder_lm_head_for_context = self.decoder_param_set.init_context_decoder(unroll=self.context_unroll,
                                                                                       buckets=self.context_buckets,
                                                                                       model_obj=self)
        self.decoder_lm_head_for_speculation = {}
        self.decoder_lm_head_for_window_context = {}

    def load_weights(self):
        # Materialize the embedding to CPU
        self.chkpt_model.model.embed_tokens.materialize()

        ops.init()

        for layer_id, layer in enumerate(self.chkpt_model.model.layers):
            if layer_id not in self.layers_after_partition:
                continue
            layer.materialize()
            attn = layer.self_attn
            mlp = layer.mlp
            new_layer = self.decoder_lm_head.new_layer()
            new_layer.add_pre_attention_layer_norm(layer.input_layernorm.weight.detach(),
                                                   layer.input_layernorm.bias.detach())
            new_layer.add_attention_query(attn.q_proj.weight.detach().T, attn.q_proj.bias.detach())
            new_layer.add_attention_key(attn.k_proj.weight.detach().T, attn.k_proj.bias.detach())
            new_layer.add_attention_value(attn.v_proj.weight.detach().T, attn.v_proj.bias.detach())
            if self.neuron_config and self.neuron_config.attn_output_transposed:
                new_layer.add_attention_output(attn.o_proj.weight.T.detach(), attn.o_proj.bias.detach(), sharding=0,
                                               transposed=True)
            else:
                new_layer.add_attention_output(attn.o_proj.weight.detach(), attn.o_proj.bias.detach(), sharding=1,
                                               transposed=False)
            new_layer.add_pre_mlp_layer_norm(layer.post_attention_layernorm.weight.detach(),
                                             layer.post_attention_layernorm.bias.detach())
            new_layer.add_mlp_input(mlp.c_fc.weight.T.detach(), mlp.c_fc.bias.detach())
            new_layer.add_mlp_output(mlp.c_proj.weight.T.detach(), mlp.c_proj.bias.detach())

            new_layer.to_neuron()
            layer.nullify()
            if self.neuron_config.shard_over_sequence:
                self.decoder_lm_head.add_pre_layer_parameter(torch.arange(self.config.tp_degree), sharding=0)
        ln_f = self.chkpt_model.model.norm
        ln_f.materialize()
        self.decoder_lm_head.add_final_layer_norm(ln_f.weight.detach(), None)

        lm_head = self.chkpt_model.lm_head
        lm_head.materialize()
        self.decoder_lm_head.add_lm_head(lm_head.weight.detach().T)
        if self.neuron_config.on_device_embedding:
            self.decoder_lm_head.add_pre_layer_parameter(self.chkpt_model.model.embed_tokens.weight, sharding=1,
                                                         allow_pad=True)
        lm_head.nullify()

        self.decoder_lm_head.to_neuron()
        # Pipeline parallel deosn't support executor right now
        if not self.neuron_config.is_pp():
            self.decoder_lm_head.use_executor = True

        if self.context_buckets:
            for context_length_estimate in self.context_buckets:
                for batch_size in self.context_batch_sizes:
                    model = self.decoder_lm_head.build_weight_shared(share_caches=True,
                                                                     new=self.decoder_lm_head_for_context[
                                                                         context_length_estimate, batch_size])
                    # PERF: No latency improvement seen in multi-layer models from executor
                    # Pipeline parallel deosn't support executor right now
                    if self.context_unroll == self.config.num_hidden_layers and not self.neuron_config.is_pp():
                        model.use_executor = True
                    self.decoder_lm_head_for_context[context_length_estimate, batch_size] = model

        if self.decoder_lm_head_for_speculation:
            for i, k in enumerate(self.decoder_lm_head_for_speculation):
                model = self.decoder_lm_head.build_weight_shared(share_caches=True,
                                                                 new=self.decoder_lm_head_for_speculation[k])
                self.decoder_lm_head_for_speculation[k] = model

        if self.decoder_lm_head_for_window_context:
            for i, k in enumerate(self.decoder_lm_head_for_window_context):
                model = self.decoder_lm_head.build_weight_shared(share_caches=True,
                                                                 new=self.decoder_lm_head_for_window_context[k])
                self.decoder_lm_head_for_window_context[k] = model

    def set_prefixed(self, input_ids):
        self.prefixed_input_ids = input_ids[:, :self.prefixed_length]
        prefixed_length = self.prefixed_length
        self.prefixed_length = 0
        self.forward(self.prefixed_input_ids)
        self.prefixed_length = prefixed_length

    def forward(self, input_ids, cache_ids=None, start_ids=None):
        inputs, *rst = self._preprocess(input_ids, start_ids=start_ids, cache_ids=cache_ids)
        if not self.neuron_config.on_device_embedding:
            inputs = self.chkpt_model.model.embed_tokens(inputs)
            if self.neuron_config.attention_layout == LAYOUT_HSB:
                inputs = inputs.transpose(0, -1).contiguous()
        logits = self._forward(inputs, *rst)
        logits = self._postprocess(logits, start_ids=start_ids)
        return logits

    def speculative_forward(self, input_ids, cache_ids=None, start_ids=None, speculation_length=None):
        if self.neuron_config and self.neuron_config.continuous_batching:
            inputs, *args = self._preprocess(input_ids, start_ids=start_ids, cache_ids=cache_ids)
        else:
            batch_size, *_ = input_ids.shape
            if start_ids is None:
                start_ids = torch.zeros(batch_size, dtype=torch.int32)
            if cache_ids is None:
                batch_size, context_length = input_ids.shape
                cache_ids = torch.arange(context_length, dtype=torch.int32)
                if self.neuron_config.use_2d_cache_ids:
                    cache_ids = cache_ids.unsqueeze(0).expand(batch_size, context_length)

            inputs, *args = input_ids, cache_ids, start_ids

        batch_size, seq_len = input_ids.shape
        if speculation_length is None:
            model = self.decoder_lm_head
        elif speculation_length not in self.decoder_lm_head_for_speculation.keys():
            # auto-infer speculation bucket, if needed
            speculation_buckets = [k for (k, batch_size) in self.decoder_lm_head_for_speculation.keys()]
            speculation_length = bucket.find(speculation_buckets, seq_len)
            model = self.decoder_lm_head_for_speculation[speculation_length, batch_size]
            if input_ids.shape[-1] > speculation_length:
                input_ids = input_ids[:, :speculation_length]
        else:
            model = self.decoder_lm_head_for_speculation[speculation_length, batch_size]

        if not self.neuron_config.on_device_embedding:
            inputs = self.chkpt_model.model.embed_tokens(inputs)
            if self.neuron_config.attention_layout == LAYOUT_HSB:
                inputs = inputs.transpose(0, -1).contiguous()
        with torch.inference_mode():
            logits = model(inputs, *args)
        logits = self._cast_logits(logits)
        logits = logits[:self.config.vocab_size, -speculation_length:, :]
        logits = logits.transpose(0, 1)

        return logits

    def sample(self, input_ids, sequence_length, cache_ids=None, start_ids=None,
               top_k=50, top_p=1.0, eos_token_override=None, temperature=1.0, streamer=None,
               stopping_criteria_list=None, no_repeat_ngram_size=None, **kwargs):

        if self.neuron_config.on_device_generation:
            return sampling.sample_tokens(self, input_ids, start_ids, sequence_length=sequence_length,
                                          config=self.neuron_config.on_device_generation, streamer=streamer)

        if self.context_pre_hook is not None:
            self.context_pre_hook()
        batch_size, context_length = input_ids.shape
        if batch_size not in self.batch_sizes:
            raise ValueError(
                f"Model not compiled for batch_size : {batch_size}. Acceptable batch_size is one of the following {self.batch_sizes}")
        prefixed_length = self.prefixed_length
        if context_length < prefixed_length:
            self.prefixed_length = 0
        else:
            input_ids = input_ids[:, prefixed_length:]
            context_length -= prefixed_length
            sequence_length -= prefixed_length

        result = sampling.sample_llama(
            self, input_ids, start_ids, sequence_length,
            eos_token_id=self.config.eos_token_id if eos_token_override is None else eos_token_override,
            top_k=top_k, top_p=top_p, temperature=temperature, streamer=streamer,
            stopping_criteria_list=stopping_criteria_list, no_repeat_ngram_size=no_repeat_ngram_size,
            cache_ids=cache_ids,
        )

        return result
