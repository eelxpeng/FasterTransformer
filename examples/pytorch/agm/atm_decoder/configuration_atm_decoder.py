from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


# This is similar to BartConfig
class ATMT5Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a :class:`~ATMT5Model`. It is used to
    instantiate a ATMT5 model according to the specified arguments, defining the model architecture.

    Configuration objects inherit from :class:`~transformers.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~transformers.PretrainedConfig` for more information.

    Args:
        vocab_size (:obj:`int`, `optional`, defaults to 50265):
            Vocabulary size of the BERT model. Defines the number of different tokens that can be represented by the
            :obj:`inputs_ids` passed when calling :class:`~transformers.BartModel`.
        d_model (:obj:`int`, `optional`, defaults to 1024):
            Dimensionality of the layers and the pooler layer.
        embed_dim (:obj:`int`, `optional`, defaults to None):
            Dimensionality of text embedding. If specified and different than `d_model`, then we insert a
            projection layer after embedding to project activations to have dimension of `d_model`
        encoder_layers (:obj:`int`, `optional`, defaults to 12):
            Number of encoder layers, 6 are used for the `bart-base` model.
        decoder_layers (:obj:`int`, `optional`, defaults to 12):
            Number of decoder layers, 6 are used for the `bart-base` model.
        encoder_attention_heads (:obj:`int`, `optional`, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        decoder_attention_heads (:obj:`int`, `optional`, defaults to 16):
            Number of attention heads for each attention layer in the Transformer decoder.
        decoder_ffn_dim (:obj:`int`, `optional`, defaults to 4096):
            Dimensionality of the "intermediate" (often named feed-forward) layer in decoder.
        encoder_ffn_dim (:obj:`int`, `optional`, defaults to 4096):
            Dimensionality of the "intermediate" (often named feed-forward) layer in decoder.
        activation_function (:obj:`str` or :obj:`function`, `optional`, defaults to :obj:`"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string,
            :obj:`"gelu"`, :obj:`"relu"`, :obj:`"silu"` and :obj:`"gelu_new"` are supported.
        dropout (:obj:`float`, `optional`, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_dropout (:obj:`float`, `optional`, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        activation_dropout (:obj:`float`, `optional`, defaults to 0.0):
            The dropout ratio for activations inside the fully connected layer.
        classifier_dropout (:obj:`float`, `optional`, defaults to 0.0):
            The dropout ratio for classifier.
        max_position_embeddings (:obj:`int`, `optional`, defaults to 1024):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        init_std (:obj:`float`, `optional`, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        add_bias_logits (:obj:`bool`, `optional`, defaults to :obj:`False`):
            This should be completed, specific to marian.
        normalize_before (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Call layernorm before attention ops.
        normalize_embedding (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Call layernorm after embeddings.
        static_position_embeddings (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Don't learn positional embeddings, use sinusoidal.
        add_final_layer_norm (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Why not add another layernorm?
        do_blenderbot_90_layernorm (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Blenderbot-90m checkpoint uses `layernorm_embedding` one line earlier in the decoder.
        scale_embedding (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Scale embeddings by diving by sqrt(d_model).
        eos_token_id (:obj:`int`, `optional`, defaults to 2)
            End of stream token id.
        pad_token_id (:obj:`int`, `optional`, defaults to 1)
            Padding token id.
        bos_token_id (:obj:`int`, `optional`, defaults to 0)
            Beginning of stream token id.
        encoder_layerdrop: (:obj:`float`, `optional`, defaults to 0.0):
            The LayerDrop probability for the encoder. See the `LayerDrop paper <see
            https://arxiv.org/abs/1909.11556>`__ for more details.
        decoder_layerdrop: (:obj:`float`, `optional`, defaults to 0.0):
            The LayerDrop probability for the decoder. See the `LayerDrop paper <see
            https://arxiv.org/abs/1909.11556>`__ for more details.
        extra_pos_embeddings: (:obj:`int`, `optional`, defaults to 2):
            How many extra learned positional embeddings to use. Should be set to :obj:`pad_token_id+1`.
        num_labels: (:obj:`int`, `optional`, defaults to 3):
            The number of labels to use in :class:`~transformers.BartForSequenceClassification`.
        is_encoder_decoder (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether this is an encoder/decoder model.
        force_bos_token_to_be_generated (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not to force BOS token to be generated at step 1 (after ``decoder_start_token_id``), only
            :obj:`True` for `bart-large-cnn`.
        use_cache (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
        share_embeddings (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not share the embeddings between encoder and decoder
        relative_attention_num_buckets (`int`, *optional*, defaults to 32):
            The number of buckets to use for each attention layer.
        relative_attention_max_distance (`int`, *optional*, defaults to 128):
            The maximum distance of the longer sequences for the bucket separation.
        scale_init_for_embeddings (:obj:`bool`, `optional`, defaults to :obj:`False`):
            If set True it scale the initial embedding weights by `1/sqrt(num_hidden_layers)` to improve stability at
            the beginning of pretraining
        use_megatron_softmax (`bool`, `optional`, defaults to False):
            If true, then use Megatron's Softmax kernel instead of `torch.nn.functional.softmax`
        use_activation_offloading (`bool`, `optional`, defaults to False):
            If true, then use DeepSpeed's activation offloading to move activations between GPU & CPU memory
        activation_offloading_max_copy_events (`int`, `optional`, defaults to 9):
            If `use_activation_offloading` is True, then this is the maximum number of concurrent CPU<->GPU copy events
        activation_offloading_max_num_prefetch (`int`, `optional`, defaults to 2):
            If `use_activation_offloading` is True, then this is the maximum number of prefetch events
        position_bias_per_layer (`bool`, `optional`, defaults to True):
            If `position_bias_per_layer` is True, then every layer has its own learned position bias parameter. If false,
            then learn one position bias parameter for entire model (as original T5 implementation does).
        scale_attention (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to scale attention scores as suggested in attention all you need paper
        add_lm_head_bias (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether to add bias to LM head (only applicable to CE4, for others there is no biases)
         use_flash_attention (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Weather or not to use memory-efficient Flash attention from xformers library.
    """
    model_type = "bart"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
            self,
            activation_dropout=0.0,
            extra_pos_embeddings=2,
            activation_function="gelu",
            vocab_size=50265,
            d_model=1024,
            embed_dim=None,
            encoder_ffn_dim=4096,
            encoder_layers=12,
            encoder_attention_heads=16,
            decoder_ffn_dim=4096,
            decoder_layers=12,
            decoder_attention_heads=16,
            encoder_layerdrop=0.0,
            decoder_layerdrop=0.0,
            attention_dropout=0.0,
            dropout=0.1,
            max_position_embeddings=1024,
            init_std=0.02,
            classifier_dropout=0.0,
            num_labels=3,
            is_encoder_decoder=False,
            normalize_before=False,
            add_final_layer_norm=False,
            do_blenderbot_90_layernorm=False,
            scale_embedding=False,
            normalize_embedding=True,
            static_position_embeddings=False,
            add_bias_logits=False,
            force_bos_token_to_be_generated=False,
            use_cache=False,
            pad_token_id=1,
            bos_token_id=0,
            eos_token_id=2,
            fp32_cast_query_key=None,
            share_embeddings=True,
            relative_attention_num_buckets=32,
            relative_attention_max_distance=128,
            scale_init_for_embeddings=True,
            layer_norm_eps=1e-12,
            use_megatron_softmax=False,
            use_activation_offloading=False,
            activation_offloading_max_copy_events=9,
            activation_offloading_max_num_prefetch=2,
            position_bias_per_layer=True,
            scale_attention=True,
            add_lm_head_bias=False,
            use_flash_attention=False,
            use_alibi_position_embedding=False,
            use_relative_attention_position_embedding=True,
            max_alibi_positions=2048,
            mup_scale=True,
            **common_kwargs
    ):
        r"""
        :class:`~ATMT5` is the configuration class for `ATMT5Model`.

        """
        if "hidden_size" in common_kwargs:
            raise ValueError("hidden size is called d_model")
        super().__init__(
            num_labels=num_labels,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            is_encoder_decoder=is_encoder_decoder,
            **common_kwargs,
        )
        self.vocab_size = vocab_size
        self.d_model = d_model  # encoder_embed_dim and decoder_embed_dim
        self.embed_dim = embed_dim
        self.encoder_ffn_dim = encoder_ffn_dim
        self.encoder_layers = self.num_hidden_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.encoder_layerdrop = encoder_layerdrop
        self.decoder_layerdrop = decoder_layerdrop
        self.decoder_ffn_dim = decoder_ffn_dim
        self.decoder_layers = decoder_layers
        self.decoder_attention_heads = decoder_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.init_std = init_std  # Normal(0, this parameter)
        self.activation_function = activation_function

        # Params introduced for Mbart
        self.scale_embedding = scale_embedding  # scale factor will be sqrt(d_model) if True
        self.normalize_embedding = normalize_embedding  # True for mbart, False otherwise
        self.normalize_before = normalize_before  # combo of fairseq's encoder_ and decoder_normalize_before
        self.add_final_layer_norm = add_final_layer_norm

        # Params introduced for Marian
        self.add_bias_logits = add_bias_logits
        self.static_position_embeddings = static_position_embeddings

        # 3 Types of Dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.dropout = dropout

        # Classifier stuff
        self.classifier_dropout = classifier_dropout

        # pos embedding offset
        self.extra_pos_embeddings = extra_pos_embeddings
        # bart has a hack that offsets positional embeddings by 2, other models don't do this

        self.force_bos_token_to_be_generated = force_bos_token_to_be_generated

        self.do_blenderbot_90_layernorm = do_blenderbot_90_layernorm

        self.use_cache = use_cache
        self.fp32_cast_query_key = fp32_cast_query_key

        self.share_embeddings = share_embeddings

        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.relative_attention_max_distance = relative_attention_max_distance

        self.scale_init_for_embeddings = scale_init_for_embeddings

        self.layer_norm_eps = layer_norm_eps

        self.use_megatron_softmax = use_megatron_softmax
        self.use_activation_offloading = use_activation_offloading
        self.activation_offloading_max_copy_events = activation_offloading_max_copy_events
        self.activation_offloading_max_num_prefetch = activation_offloading_max_num_prefetch

        self.position_bias_per_layer = position_bias_per_layer
        self.scale_attention = scale_attention

        self.add_lm_head_bias = add_lm_head_bias
        self.use_flash_attention = use_flash_attention
        self.use_relative_attention_position_embedding = use_relative_attention_position_embedding
        self.use_alibi_position_embedding = use_alibi_position_embedding
        self.max_alibi_positions = max_alibi_positions

        self.mup_scale = mup_scale

    @property
    def num_attention_heads(self) -> int:
        return self.encoder_attention_heads

    @property
    def hidden_size(self) -> int:
        return self.d_model

    def is_valid_mbart(self) -> bool:
        """Is the configuration aligned with the MBART paper."""
        if self.normalize_before and self.add_final_layer_norm and self.scale_embedding:
            return True
        if self.normalize_before or self.add_final_layer_norm or self.scale_embedding:
            logger.info("This configuration is a mixture of MBART and BART settings")
        return False
