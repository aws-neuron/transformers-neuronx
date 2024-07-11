from transformers_neuronx import dtypes
from transformers_neuronx import module
from transformers_neuronx import utils


class Starcoder2ForCausalLM(module.PretrainedModel):

    def __init__(self, config):
        super().__init__()
        dtype, _, _ = utils.parse_amp(config.amp)
        dtype = dtypes.to_torch_dtype(dtype)
        self.model = Starcoder2Model(config)
        self.lm_head = module.LowMemoryLazyLinear(config.vocab_size, dtype=dtype, bias=config.use_bias)

    def get_tied_parameters(self):
        return [(self.model.embed_tokens.weight, self.lm_head.weight)]

    def get_base_model(self):
        return self.model


class Starcoder2Model(module.LowMemoryModule):

    def __init__(self, config):
        super().__init__()
        self.embed_tokens = module.LowMemoryEmbedding(config.vocab_size, config.hidden_size)
        self.layers = module.LowMemoryModuleList(
            [Starcoder2DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = Starcoder2RMSNorm(config)


class Starcoder2RMSNorm(module.LowMemoryModule):

    def __init__(self, config) -> None:
        super().__init__()
        self.weight = module.UninitializedParameter()
        if config.use_bias:
            self.bias = module.UninitializedParameter()
        else:
            self.bias = None


class Starcoder2DecoderLayer(module.LowMemoryModule):

    def __init__(self, config):
        super().__init__()
        self.self_attn = Starcoder2Attention(config)
        self.mlp = Starcoder2MLP(config)
        self.input_layernorm = Starcoder2RMSNorm(config)
        self.post_attention_layernorm = Starcoder2RMSNorm(config)


class Starcoder2Attention(module.LowMemoryModule):

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        dtype, _, _ = utils.parse_amp(config.amp)
        dtype = dtypes.to_torch_dtype(dtype)
        self.q_proj = module.LowMemoryLazyLinear(self.num_heads * self.head_dim, bias=config.use_bias, dtype=dtype)
        self.k_proj = module.LowMemoryLazyLinear(config.num_key_value_heads * self.head_dim, bias=config.use_bias,
                                                 dtype=dtype)
        self.v_proj = module.LowMemoryLazyLinear(config.num_key_value_heads * self.head_dim, bias=config.use_bias,
                                                 dtype=dtype)
        self.o_proj = module.LowMemoryLazyLinear(self.hidden_size, bias=config.use_bias, dtype=dtype)


class Starcoder2MLP(module.LowMemoryModule):

    def __init__(self, config):
        super().__init__()
        dtype, _, _ = utils.parse_amp(config.amp)
        dtype = dtypes.to_torch_dtype(dtype)
        self.c_fc = module.LowMemoryLazyLinear(config.intermediate_size, bias=config.use_bias, dtype=dtype)
        self.c_proj = module.LowMemoryLazyLinear(config.hidden_size, bias=config.use_bias, dtype=dtype)
