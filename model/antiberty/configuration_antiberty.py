from transformers.utils import logging

from ..configuration_utils import HeadConfig, MaskedLMHeadConfig, PretrainedConfig

logger = logging.get_logger(__name__)


class AntiBERTyConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`RnaBertModel`]. It is used to instantiate a
    RnaBert model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the RnaBert
    [mana438/RNABERT](https://github.com/mana438/RNABERT) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 25):
            Vocabulary size of the RnaBert model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`RnaBertModel`].
        hidden_size (`int`, *optional*, defaults to 120):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 6):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 40):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        hidden_dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (`int`, *optional*, defaults to 440):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.

    Examples:
        >>> from multimolecule import RnaBertModel, RnaBertConfig

        >>> # Initializing a RNABERT multimolecule/rnabert style configuration
        >>> configuration = RnaBertConfig()

        >>> # Initializing a model (with random weights) from the multimolecule/rnabert style configuration
        >>> model = RnaBertModel(configuration)

        >>> # Accessing the model configuration
        >>> configuration = model.config
    """

    model_type = "antiberty"

    def __init__(
        self,
        vocab_size=25,
        hidden_size=512,
        num_hidden_layers=8,
        num_attention_heads=8,
        intermediate_size=2048,
        hidden_act="gelu",
        hidden_dropout=0.1,
        attention_dropout=0.1,
        type_vocab_size=2,
        max_position_embeddings=512,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        position_embedding_type="absolute",
        use_cache=True,
        head=None,
        lm_head=None,
        token_type=None,
        freeze=True,
        _name_or_path=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_name_or_path = _name_or_path
        self.vocab_size = vocab_size
        self.type_vocab_size = type_vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache
        self.output_hidden_states = True
        self.head = HeadConfig(**head if head is not None else {})
        self.lm_head = MaskedLMHeadConfig(**lm_head if lm_head is not None else {})
        self.freeze = freeze
