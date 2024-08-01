from transformers_neuronx import utils


class Starcoder2Config:
    def __init__(
            self,
            config,
            n_positions,
            batch_size,
            amp,
            tp_degree,
            **kwargs
    ):
        # Extract configs used for building HLO
        self.intermediate_size = config.intermediate_size
        self.hidden_size = config.hidden_size

        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads if hasattr(config,
                                                                         "num_key_value_heads") else config.num_attention_heads
        self.num_hidden_layers = config.num_hidden_layers
        self.vocab_size = config.vocab_size
        self.hidden_act = config.hidden_act
        self.bos_token_id = config.bos_token_id
        self.eos_token_id = config.eos_token_id
        self.max_position_embeddings = config.max_position_embeddings
        self.rms_norm_eps = config.norm_epsilon
        self.rotary_percentage = getattr(config, "rotary_percentage", 1)
        self.rope_theta = getattr(config, "rope_theta", 10000)
        self.position_interpolation_factor = getattr(config, "position_interpolation_factor", None)
        self.use_bias = getattr(config, "use_bias", True)
        utils.maybe_override_attributes(self, kwargs)

        # Add required Neuron configs
        self.n_positions = n_positions
        self.batch_size = batch_size
        self.amp = amp
        self.tp_degree = tp_degree
