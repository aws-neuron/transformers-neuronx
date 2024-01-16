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
from transformers_neuronx import dtypes
from transformers_neuronx import module
from transformers_neuronx import utils


class BloomForCausalLM(module.PretrainedModel):

    def __init__(self, config):
        super().__init__()
        dtype, _, _ = utils.parse_amp(config.amp)
        dtype = dtypes.to_torch_dtype(dtype)
        self.transformer = BloomModel(config)
        self.lm_head = module.LowMemoryLazyLinear(config.vocab_size, dtype=dtype, bias=False)

    def get_tied_parameters(self):
        return [(self.transformer.word_embeddings.weight, self.lm_head.weight)]

    def get_base_model(self):
        return self.transformer

class BloomModel(module.LowMemoryModule):

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = module.LowMemoryEmbedding(config.vocab_size, config.embed_dim)
        self.word_embeddings_layernorm = module.LowMemoryLayerNorm(config.embed_dim, eps=config.layer_norm_epsilon)
        self.h = module.LowMemoryModuleList([BloomBlock(config) for _ in range(config.n_layer)])
        self.ln_f = module.LowMemoryLayerNorm(config.embed_dim, eps=config.layer_norm_epsilon)


class BloomBlock(module.LowMemoryModule):

    def __init__(self, config):
        super().__init__()
        self.input_layernorm = module.LowMemoryLayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.self_attention = BloomAttention(config)
        self.post_attention_layernorm = module.LowMemoryLayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = BloomMLP(config)


class BloomAttention(module.LowMemoryModule):

    def __init__(self, config):
        super().__init__()
        dtype, _, _ = utils.parse_amp(config.amp)
        dtype = dtypes.to_torch_dtype(dtype)
        self.query_key_value = module.LowMemoryLazyLinear(3 * config.hidden_size, dtype=dtype, bias=True)
        self.dense = module.LowMemoryLazyLinear(config.hidden_size, dtype=dtype)


class BloomMLP(module.LowMemoryModule):

    def __init__(self, config):
        super().__init__()
        dtype, _, _ = utils.parse_amp(config.amp)
        dtype = dtypes.to_torch_dtype(dtype)
        self.dense_h_to_4h = module.LowMemoryLazyLinear(4 * config.hidden_size, dtype=dtype)
        self.dense_4h_to_h = module.LowMemoryLazyLinear(config.embed_dim, dtype=dtype)
