import contextlib
import math
import random
import warnings
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.activations import ACT2FN
from transformers.file_utils import (
    add_code_sample_docstrings,
    add_end_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

from .configuration_atm_decoder import ATMT5Config

try:
    import xformers.ops as xops
except:
    print("Warning: Could not find xformers")

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "BartConfig"
_TOKENIZER_FOR_DOC = "BartTokenizer"

BART_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/bart-large",
    # See all BART models at https://huggingface.co/models?filter=bart
]


# For alibi pos embeddings
def get_slopes_power_of_2(n):
    start = (2 ** (-2 ** -(math.log2(n) - 3)))
    ratio = start
    return [start * ratio ** i for i in range(n)]


def get_slopes(n):
    if math.log2(n).is_integer():
        return get_slopes_power_of_2(n)
    # In the paper, we only train models that have 2^a heads for some a. This function has
    # some good properties that only occur when the input is a power of 2. To maintain that even
    # when the number of heads is not a power of 2, we use this workaround.
    else:
        closest_power_of_2 = 2 ** math.floor(
            math.log2(n))
        return get_slopes_power_of_2(closest_power_of_2) + \
               get_slopes(2 * closest_power_of_2)[0::2][:n - closest_power_of_2]


def compute_alibi_bias(max_pos, attn_heads):
    # following is a work-around to be able to initialize models in deepspeed context
    slopes = torch.zeros(attn_heads)
    # perform operations in float on cpu so that we can use numpy
    slopes = slopes.to(torch.float).to("cpu") + np.array(get_slopes(attn_heads))

    # In the next line, the part after the * is what constructs the diagonal matrix (right matrix in Figure 3 in the paper).
    # If you run it you'll see that it doesn't exactly print out the same matrix as we have in Figure 3, but one where all rows are identical.
    # This works because the softmax operation is invariant to translation, and our bias functions are always linear.
    alibi = slopes.unsqueeze(1).unsqueeze(1) * \
            torch.arange(max_pos).unsqueeze(0).unsqueeze(0).expand(attn_heads, -1, -1)
    alibi = alibi.view(attn_heads, 1, max_pos)
    return alibi


# Done with alibi pos embeddings utility

def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


def _make_causal_mask(input_ids_shape: torch.Size, dtype: torch.dtype, past_key_values_length: int = 0):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), float("-inf"))
    mask_cond = torch.arange(mask.size(-1))
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    or from `[bsz, tgt_seq_len, tgt_seq_len]` to `[bsz, 1, tgt_seq_len, tgt_seq_len]` in decoder only mode
    """
    if len(mask.size()) == 2:
        bsz, src_len = mask.size()
        tgt_len = tgt_len if tgt_len is not None else src_len

        expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
    elif len(mask.size()) == 3:
        expanded_mask = mask[:, None, :, :].to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.bool(), torch.finfo(dtype).min)


class ATMT5CE4Attention(nn.Module):
    def __init__(self, embed_dim, num_heads, config: ATMT5Config, is_decoder=False, has_relative_attention_bias=False,
                 has_alibi_attention_bias=False, mode=0):
        super().__init__()
        self.is_decoder = is_decoder
        self.has_relative_attention_bias = has_relative_attention_bias
        self.has_alibi_attention_bias = has_alibi_attention_bias
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.relative_attention_max_distance = config.relative_attention_max_distance
        self.d_model = embed_dim
        self.key_value_proj_dim = embed_dim // num_heads
        self.n_heads = num_heads
        self.dropout = config.dropout
        self.inner_dim = self.n_heads * self.key_value_proj_dim

        if config.scale_attention:
            if not config.mup_scale:
                self.scaling = self.key_value_proj_dim ** -0.5
            else:
                self.scaling = 8 / self.key_value_proj_dim
        else:
            self.scaling = 1.0

        self.use_flash_attention = config.use_flash_attention

        # Mesh TensorFlow initialization to avoid scaling before softmax
        self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)

        self.position_bias_per_layer = config.position_bias_per_layer

        if self.has_relative_attention_bias and self.has_alibi_attention_bias:
            raise ValueError("Cannot use both relative attention and alibi attention")

        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets, self.n_heads)
        if self.has_alibi_attention_bias:
            self.alibi = compute_alibi_bias(config.max_alibi_positions, self.n_heads)

        # self.pruned_heads = set()
        self.gradient_checkpointing = False

    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593
        Translate relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on
        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer
        Returns:
            a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
        """
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        # now relative_position is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        relative_position_if_large = max_exact + (
                torch.log(relative_position.float() / max_exact)
                / math.log(max_distance / max_exact)
                * (num_buckets - max_exact)
        ).to(torch.long)
        relative_position_if_large = torch.min(
            relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
        )

        relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets

    def compute_bias(self, query_length, key_length, device=None):
        """Compute binned relative position bias"""
        if device is None:
            device = self.relative_attention_bias.weight.device
        context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        relative_position = memory_position - context_position  # shape (query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        # it is important to call contiguous to speed up the summation
        values = values.permute([2, 0, 1]).unsqueeze(0).contiguous()  # shape (1, num_heads, query_length, key_length)
        return values

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            key_value_states=None,
            position_bias=None,
            past_key_value=None,
            layer_head_mask=None,
            query_length=None,
            use_cache=False,
            output_attentions=False,
    ):
        """
        Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
        """
        # Input is (batch_size, seq_length, dim)
        # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
        # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)
        batch_size, seq_length = hidden_states.shape[:2]

        real_seq_length = seq_length

        if past_key_value is not None:
            assert (
                    len(past_key_value) == 2
            ), f"past_key_value should have 2 past states: keys and values. Got {len(past_key_value)} past states"
            real_seq_length += past_key_value[0].shape[2] if query_length is None else query_length

        key_length = real_seq_length if key_value_states is None else key_value_states.shape[1]

        def shape(states):
            """projection"""
            return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

        def unshape(states):
            """reshape"""
            return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)

        def project(hidden_states, proj_layer, key_value_states, past_key_value):
            """projects hidden states correctly to key/query states"""
            if key_value_states is None:
                # self-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(hidden_states))
            elif past_key_value is None:
                # cross-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(key_value_states))

            if past_key_value is not None:
                if key_value_states is None:
                    # self-attn
                    # (batch_size, n_heads, key_length, dim_per_head)
                    hidden_states = torch.cat([past_key_value, hidden_states], dim=2)
                else:
                    # cross-attn
                    hidden_states = past_key_value
            return hidden_states

        # get query states
        query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)

        # get key/value states
        key_states = project(
            hidden_states, self.k, key_value_states, past_key_value[0] if past_key_value is not None else None
        )
        value_states = project(
            hidden_states, self.v, key_value_states, past_key_value[1] if past_key_value is not None else None
        )

        if self.has_alibi_attention_bias:
            if self.alibi.device != hidden_states.device:
                self.alibi = self.alibi.to(hidden_states.device).to(hidden_states.dtype)
            position_bias = self.alibi[:, :, 0:real_seq_length].unsqueeze(0)
        elif position_bias is None:
            if not self.has_relative_attention_bias:
                position_bias = torch.zeros(
                    (1, self.n_heads, real_seq_length, key_length), device=scores.device, dtype=scores.dtype
                )
                if self.gradient_checkpointing and self.training:
                    position_bias.requires_grad = True
            else:
                position_bias = self.compute_bias(real_seq_length, key_length, device=hidden_states.device)

            # if key and values are already calculated
            # we want only the last query position bias
            if past_key_value is not None:
                position_bias = position_bias[:, :, -hidden_states.size(1):, :]

        if attention_mask is not None:
            position_bias_masked = position_bias + attention_mask  # (batch_size, n_heads, seq_length, key_length)
        else:
            position_bias_masked = position_bias

        if self.use_flash_attention:
            query_states = query_states.permute(0, 2, 1, 3)
            key_states = key_states.permute(0, 2, 1, 3)
            value_states = value_states.permute(0, 2, 1, 3)
            batch_size, n_heads = query_states.size(0), query_states.size(2)
            n_queries, n_keys = query_states.size(1), key_states.size(1)

            # Need to reshape position_bias for flash-attention
            # position_bias_masked = position_bias_masked.broadcast_to(batch_size, n_heads, n_queries, n_keys).reshape(-1, n_queries, n_keys)
            position_bias_masked = position_bias_masked.broadcast_to(batch_size, n_heads, n_queries, n_keys)
            position_bias_masked = position_bias_masked.contiguous()

            attn_output = xops.memory_efficient_attention(query_states, key_states, value_states, scale=self.scaling,
                                                          p=self.dropout, attn_bias=position_bias_masked)
            attn_output = attn_output.view(batch_size, -1, self.inner_dim)
        else:
            # Flash-attention handles its own scaling so we only need to manualy do it when not using flash attention
            query_states *= self.scaling

            # compute scores
            scores = torch.matmul(
                query_states, key_states.transpose(3, 2)
            )  # equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9
            scores += position_bias_masked
            attn_weights = nn.functional.softmax(scores, dim=-1)  # (batch_size, n_heads, seq_length, key_length)

            attn_weights = nn.functional.dropout(
                attn_weights, p=self.dropout, training=self.training
            )  # (batch_size, n_heads, seq_length, key_length)

            # Mask heads if we want to
            if layer_head_mask is not None:
                attn_weights = attn_weights * layer_head_mask

            attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)

        attn_output = self.o(attn_output)

        present_key_value_state = (key_states, value_states) if (self.is_decoder and use_cache) else (None, None)

        outputs = (attn_output,) + (present_key_value_state,)
        if not self.position_bias_per_layer:
            outputs = outputs + (position_bias,)

        if output_attentions:
            outputs = outputs + (attn_weights,)
        return outputs


class ATMT5CE4EncoderLayer(nn.Module):
    def __init__(self, config: ATMT5Config, has_relative_attention_bias=False, has_alibi_attention_bias=False):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = ATMT5CE4Attention(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            config=config,
            has_relative_attention_bias=has_relative_attention_bias,
            has_alibi_attention_bias=has_alibi_attention_bias,
            mode=0
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: torch.Tensor,
            layer_head_mask: torch.Tensor,
            position_bias: torch.Tensor = None,
            output_attentions: bool = False,
    ):
        """
        Args:
            hidden_states (:obj:`torch.FloatTensor`): input to the layer of shape `(seq_len, batch, embed_dim)`
            attention_mask (:obj:`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (:obj:`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(config.encoder_attention_heads,)`.
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
        """
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        attention_output = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
            position_bias=position_bias
        )
        hidden_states = F.dropout(attention_output[0], p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        # move this up to become preLN
        # hidden_states = self.self_attn_layer_norm(hidden_states)

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = F.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        # moving this up to become preLN
        # hidden_states = self.final_layer_norm(hidden_states)

        if torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any():
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,) + attention_output[1:]

        # comment out since it is already checked in attention outputs
        # if output_attentions:
        #     outputs += (attn_weights,)

        return outputs


class ATMT5CE4DecoderLayer(nn.Module):
    def __init__(self, config: ATMT5Config, has_relative_attention_bias=False, has_alibi_attention_bias=False, mode=0):
        """

        :param config:
        :param has_relative_attention_bias:
        :param mode: either 0 for self attention only or 1 for cross-attention only
        """
        super().__init__()
        self.embed_dim = config.d_model

        self.position_bias_per_layer = config.position_bias_per_layer
        if mode == 0:
            self.self_attn = ATMT5CE4Attention(
                embed_dim=self.embed_dim,
                num_heads=config.decoder_attention_heads,
                is_decoder=True,
                config=config,
                has_relative_attention_bias=has_relative_attention_bias,
                has_alibi_attention_bias=has_alibi_attention_bias,
                mode=mode
            )
            self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        elif mode == 1:
            self.encoder_attn = ATMT5CE4Attention(
                self.embed_dim,
                config.decoder_attention_heads,
                is_decoder=True,
                config=config,
                has_relative_attention_bias=has_relative_attention_bias,
                has_alibi_attention_bias=has_alibi_attention_bias,
                mode=mode
            )
            self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)
        self.mode = mode

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            layer_head_mask: Optional[torch.Tensor] = None,
            encoder_layer_head_mask: Optional[torch.Tensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            self_attn_position_bias=None,
            encoder_attn_position_bias=None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = True,
    ):
        """
        Args:
            hidden_states (:obj:`torch.FloatTensor`): input to the layer of shape `(seq_len, batch, embed_dim)`
            attention_mask (:obj:`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            encoder_hidden_states (:obj:`torch.FloatTensor`): cross attention input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_attention_mask (:obj:`torch.FloatTensor`): encoder attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (:obj:`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(config.encoder_attention_heads,)`.
            encoder_layer_head_mask (:obj:`torch.FloatTensor`): mask for encoder attention heads in a given layer of
                size `(config.encoder_attention_heads,)`.
            past_key_value (:obj:`Tuple(torch.FloatTensor)`): cached past key and value projection states
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
        """
        residual = hidden_states
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
        if cross_attn_past_key_value == (None, None):
            cross_attn_past_key_value = None
        if self_attn_past_key_value == (None, None):
            self_attn_past_key_value = None
        if self.mode == 0:
            hidden_states = self.self_attn_layer_norm(hidden_states)
            # print("context pre LN output", hidden_states[0,0,:10])
            # Self Attention
            # decoder uni-directional self-attention cached key/values tuple is at positions 1,2

            # add present self-attn cache to positions 1,2 of present_key_value tuple
            # hidden_states, self_attn_weights, present_key_value
            self_attention_output = self.self_attn(
                hidden_states=hidden_states,
                past_key_value=self_attn_past_key_value,
                attention_mask=attention_mask,
                layer_head_mask=layer_head_mask,
                output_attentions=output_attentions,
                position_bias=self_attn_position_bias,
                use_cache=use_cache
            )
            
            if self.position_bias_per_layer:
                present_key_value = self_attention_output[1]
            else:
                present_key_value, self_attn_position_bias = self_attention_output[1:3]

            hidden_states = F.dropout(self_attention_output[0], p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states
            # moving this up to become preLN
            # hidden_states = self.self_attn_layer_norm(hidden_states)
            if cross_attn_past_key_value is not None:
                cross_attn_present_key_value = cross_attn_past_key_value
            else:
                cross_attn_present_key_value = (None, None)
        elif self.mode == 1:
            # Cross-Attention Block
            cross_attn_present_key_value = None
            cross_attn_weights = None
            if encoder_hidden_states is not None:
                # residual = hidden_states
                if self_attn_past_key_value is not None and use_cache:
                    query_length = self_attn_past_key_value[0].shape[2]
                else:
                    query_length = None
                hidden_states = self.encoder_attn_layer_norm(hidden_states)
                # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple

                # hidden_states, cross_attn_weights, cross_attn_present_key_value
                encoder_attention_output = self.encoder_attn(
                    hidden_states=hidden_states,
                    key_value_states=encoder_hidden_states,
                    attention_mask=encoder_attention_mask,
                    layer_head_mask=encoder_layer_head_mask,
                    past_key_value=cross_attn_past_key_value,
                    output_attentions=output_attentions,
                    position_bias=encoder_attn_position_bias,
                    query_length=query_length,
                    use_cache=use_cache
                )

                if self.position_bias_per_layer:
                    cross_attn_present_key_value = encoder_attention_output[1]
                else:
                    cross_attn_present_key_value, encoder_attn_position_bias = encoder_attention_output[1:3]
                hidden_states = F.dropout(encoder_attention_output[0], p=self.dropout, training=self.training)
                hidden_states = residual + hidden_states
                # moving this up to become preLN
                # hidden_states = self.encoder_attn_layer_norm(hidden_states)

                # add cross-attn to positions 3,4 of present_key_value tuple
            # if self_attn_past_key_value is None:
            #     present_key_value = (None, None)
            # else:
            present_key_value = (None, None)
        present_key_value = present_key_value + cross_attn_present_key_value

        # Fully Connected
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        # print("context ffn after LN output", hidden_states[0,0,:10])
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = F.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        # moving this up to make it preLN
        # hidden_states = self.final_layer_norm(hidden_states)

        # print("context ffn output", hidden_states[0,0,:10])
        outputs = (hidden_states,)
        if not self.position_bias_per_layer:
            outputs = outputs + (self_attn_position_bias,) + (encoder_attn_position_bias,)

        if output_attentions:
            outputs += (self_attention_output[2], encoder_attention_output[2])

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class ATMT5CE4ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
            self,
            input_dim: int,
            inner_dim: int,
            num_classes: int,
            pooler_dropout: float,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states


class ATMT5CE4PretrainedModel(PreTrainedModel):
    config_class = ATMT5Config
    base_model_prefix = "model"

    def _init_weights(self, module):
        std = self.config.init_std
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            if self.config.scale_init_for_embeddings:
                module.weight.data.normal_(mean=0.0, std=std / math.sqrt(self.config.num_hidden_layers))
            else:
                module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    @property
    def dummy_inputs(self):
        pad_token = self.config.pad_token_id
        input_ids = torch.tensor([[0, 6, 10, 4, 2], [0, 8, 12, 2, pad_token]], device=self.device)
        dummy_inputs = {
            "attention_mask": input_ids.ne(pad_token),
            "input_ids": input_ids,
        }
        return dummy_inputs


class PretrainedATMT5CE4Model(ATMT5CE4PretrainedModel):
    def __init_subclass__(self):
        warnings.warn(
            "The class `PretrainedATMT5CE4Model` has been depreciated, please use `ATMT5CE4PretrainedModel` instead.",
            FutureWarning,
        )


BART_START_DOCSTRING = r"""
    This model inherits from :class:`~transformers.PreTrainedModel`. Check the superclass documentation for the generic
    methods the library implements for all its model (such as downloading or saving, resizing the input embeddings,
    pruning heads etc.)

    This model is also a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`__
    subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to
    general usage and behavior.

    Parameters:
        config (:class:`~transformers.BartConfig`):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.
"""

BART_GENERATION_EXAMPLE = r"""
    Summarization example::

        >>> from transformers import BartTokenizer, ATMT5CE4ForConditionalGeneration, BartConfig

        >>> model = ATMT5CE4ForConditionalGeneration.from_pretrained('facebook/bart-large')
        >>> tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

        >>> ARTICLE_TO_SUMMARIZE = "My friends are cool but they eat too many carbs."
        >>> inputs = tokenizer([ARTICLE_TO_SUMMARIZE], max_length=1024, return_tensors='pt')

        >>> # Generate Summary
        >>> summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=5, early_stopping=True)
        >>> print([tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids])

    Mask filling example::

        >>> from transformers import BartTokenizer, ATMT5CE4ForConditionalGeneration
        >>> tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
        >>> TXT = "My friends are <mask> but they eat too many carbs."

        >>> model = ATMT5CE4ForConditionalGeneration.from_pretrained('facebook/bart-large')
        >>> input_ids = tokenizer([TXT], return_tensors='pt')['input_ids']
        >>> logits = model(input_ids).logits

        >>> masked_index = (input_ids[0] == tokenizer.mask_token_id).nonzero().item()
        >>> probs = logits[0, masked_index].softmax(dim=0)
        >>> values, predictions = probs.topk(5)

        >>> tokenizer.decode(predictions).split()
"""

BART_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using :class:`~transformers.BartTokenizer`. See
            :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__` for
            details.

            `What are input IDs? <../glossary.html#input-ids>`__
        attention_mask (:obj:`torch.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            `What are attention masks? <../glossary.html#attention-mask>`__
        decoder_input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, target_sequence_length)`, `optional`):
            Indices of decoder input sequence tokens in the vocabulary.

            Indices can be obtained using :class:`~transformers.BartTokenizer`. See
            :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__` for
            details.

            `What are input IDs? <../glossary.html#input-ids>`__

            ATMT5 uses the :obj:`eos_token_id` as the starting token for :obj:`decoder_input_ids` generation. If
            :obj:`past_key_values` is used, optionally only the last :obj:`decoder_input_ids` have to be input (see
            :obj:`past_key_values`).

            For translation and summarization training, :obj:`decoder_input_ids` should be provided. If no
            :obj:`decoder_input_ids` is provided, the model will create this tensor by shifting the :obj:`input_ids` to
            the right for denoising pre-training following the paper.
        decoder_attention_mask (:obj:`torch.LongTensor` of shape :obj:`(batch_size, target_sequence_length)`, `optional`):
            Default behavior: generate a tensor that ignores pad tokens in :obj:`decoder_input_ids`. Causal mask will
            also be used by default.

            If you want to change padding behavior, you should read :func:`modeling_bart._prepare_decoder_inputs` and
            modify to your needs. See diagram 1 in `the paper <https://arxiv.org/abs/1910.13461>`__ for more
            information on the default strategy.
        head_mask (:obj:`torch.Tensor` of shape :obj:`(num_layers, num_heads)`, `optional`):
            Mask to nullify selected heads of the attention modules in the encoder. Mask values selected in ``[0, 1]``:

            - 1 indicates the head is **not masked**,
            - 0 indicates the heas is **masked**.

        decoder_head_mask (:obj:`torch.Tensor` of shape :obj:`(num_layers, num_heads)`, `optional`):
            Mask to nullify selected heads of the attention modules in the decoder. Mask values selected in ``[0, 1]``:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        encoder_outputs (:obj:`tuple(tuple(torch.FloatTensor)`, `optional`):
            Tuple consists of (:obj:`last_hidden_state`, `optional`: :obj:`hidden_states`, `optional`:
            :obj:`attentions`) :obj:`last_hidden_state` of shape :obj:`(batch_size, sequence_length, hidden_size)`,
            `optional`) is a sequence of hidden-states at the output of the last layer of the encoder. Used in the
            cross-attention of the decoder.
        past_key_values (:obj:`Tuple[Tuple[torch.Tensor]]` of length :obj:`config.n_layers` with each tuple having 2 tuples each of which has 2 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden-states of the attention blocks. Can be used to speed up decoding.

            If :obj:`past_key_values` are used, the user can optionally input only the last :obj:`decoder_input_ids`
            (those that don't have their past key value states given to this model) of shape :obj:`(batch_size, 1)`
            instead of all :obj:`decoder_input_ids`` of shape :obj:`(batch_size, sequence_length)`.
        inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert :obj:`input_ids` indices into associated
            vectors than the model's internal embedding lookup matrix.
        decoder_inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, target_sequence_length, hidden_size)`, `optional`):
            Optionally, instead of passing :obj:`decoder_input_ids` you can choose to directly pass an embedded
            representation. If :obj:`past_key_values` is used, optionally only the last :obj:`decoder_inputs_embeds`
            have to be input (see :obj:`past_key_values`). This is useful if you want more control over how to convert
            :obj:`decoder_input_ids` indices into associated vectors than the model's internal embedding lookup matrix.

            If :obj:`decoder_input_ids` and :obj:`decoder_inputs_embeds` are both unset, :obj:`decoder_inputs_embeds`
            takes the value of :obj:`inputs_embeds`.
        use_cache (:obj:`bool`, `optional`):
            If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
            decoding (see :obj:`past_key_values`).
        output_attentions (:obj:`bool`, `optional`):
            Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under returned
            tensors for more detail.
        output_hidden_states (:obj:`bool`, `optional`):
            Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors for
            more detail.
        return_dict (:obj:`bool`, `optional`):
            Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
"""


class ATMT5CE4Encoder(ATMT5CE4PretrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    :class:`ATMT5CE4EncoderLayer`.

    Args:
        config: BartConfig
        embed_tokens (torch.nn.Embedding): output embedding
    """

    def __init__(self, config: ATMT5Config, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)

        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, embed_dim, self.padding_idx)

        if config.embed_dim is not None:
            self.embed_projection = True
            self.proj = nn.Linear(config.embed_dim, config.hidden_size, bias=False)
        else:
            self.embed_projection = False

        # self.embed_positions = ATMT5LearnedPositionalEmbedding(
        #     config.max_position_embeddings,
        #     embed_dim
        # )
        if (not config.use_relative_attention_position_embedding) and (not config.use_alibi_position_embedding):
            raise ValueError("Must use either relative position embeddings or alibi position embeddings")
        if config.use_relative_attention_position_embedding and config.use_alibi_position_embedding:
            raise ValueError(
                "Must use either relative position embeddings or alibi position embeddings, cannot specify both")

        if config.use_relative_attention_position_embedding:
            if config.position_bias_per_layer:
                self.layers = nn.ModuleList([ATMT5CE4EncoderLayer(config, has_relative_attention_bias=True) for i in
                                             range(config.encoder_layers)])
            else:
                self.layers = nn.ModuleList(
                    [ATMT5CE4EncoderLayer(config, has_relative_attention_bias=bool(i == 0)) for i in
                     range(config.encoder_layers)])
        elif config.use_alibi_position_embedding:
            self.layers = nn.ModuleList([ATMT5CE4EncoderLayer(config, has_alibi_attention_bias=True) for i in
                                         range(config.encoder_layers)])

        # remove this in preLN
        # self.layernorm_embedding = nn.LayerNorm(embed_dim)
        # add this for preLN
        self.layernorm_output = nn.LayerNorm(embed_dim)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            head_mask=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        Args:
            input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using :class:`~transformers.BartTokenizer`. See
                :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__`
                for details.

                `What are input IDs? <../glossary.html#input-ids>`__
            attention_mask (:obj:`torch.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                `What are attention masks? <../glossary.html#attention-mask>`__
            head_mask (:obj:`torch.Tensor` of shape :obj:`(num_layers, num_heads)`, `optional`):
                Mask to nullify selected heads of the attention modules. Mask values selected in ``[0, 1]``:

                - 1 indicates the head is **not masked**,
                - 0 indicates the heas is **masked**.

            inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
                Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded
                representation. This is useful if you want more control over how to convert :obj:`input_ids` indices
                into associated vectors than the model's internal embedding lookup matrix.
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
            output_hidden_states (:obj:`bool`, `optional`):
                Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors
                for more detail.
            return_dict (:obj:`bool`, `optional`):
                Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        if self.embed_projection:
            inputs_embeds = self.proj(inputs_embeds)

        # embed_pos = self.embed_positions(input_shape)

        # hidden_states = inputs_embeds + embed_pos
        hidden_states = inputs_embeds
        # removing this in preLN
        # hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)

        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, inputs_embeds.dtype)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            assert head_mask.size()[0] == (
                len(self.layers)
            ), f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
        position_bias = None

        cm = contextlib.nullcontext()
        with cm:
            for idx, encoder_layer in enumerate(self.layers):
                if output_hidden_states:
                    encoder_states = encoder_states + (hidden_states,)
                # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
                dropout_probability = random.uniform(0, 1)
                if self.training and (dropout_probability < self.layerdrop):  # skip the layer
                    layer_outputs = (None, None)
                else:
                    if getattr(self.config, "gradient_checkpointing", False) and self.training:

                        def create_custom_forward(module):
                            def custom_forward(*inputs):
                                return module(*inputs, output_attentions)

                            return custom_forward

                        layer_outputs = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(encoder_layer),
                            hidden_states,
                            attention_mask,
                            (head_mask[idx] if head_mask is not None else None),
                            position_bias
                        )
                    else:
                        layer_outputs = encoder_layer(
                            hidden_states,
                            attention_mask,
                            layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                            position_bias=position_bias,
                            output_attentions=output_attentions
                        )

                    hidden_states = layer_outputs[0]

                    if not self.config.position_bias_per_layer:
                        position_bias = layer_outputs[2]

                if output_attentions:
                    all_attentions = all_attentions + (layer_outputs[2],)

        # add this for preLN
        hidden_states = self.layernorm_output(hidden_states)
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )


class ATMT5CE4Decoder(ATMT5CE4PretrainedModel):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a :class:`ATMT5CE4DecoderLayer`

    Args:
        config: BartConfig
        embed_tokens (torch.nn.Embedding): output embedding
    """

    def __init__(self, config: ATMT5Config, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0
        self.is_encoder_decoder = config.is_encoder_decoder
        self.bos_token_id = config.bos_token_id

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            if config.embed_dim is None:
                embed_dim = config.d_model
            else:
                embed_dim = config.embed_dim

            self.embed_tokens = nn.Embedding(config.vocab_size, embed_dim, self.padding_idx)

        if config.embed_dim is not None:
            self.embed_projection = True
            self.proj = nn.Linear(config.embed_dim, config.hidden_size, bias=False)
        else:
            self.embed_projection = False

        # self.embed_positions = ATMT5LearnedPositionalEmbedding(
        #     config.max_position_embeddings,
        #     config.d_model
        # )

        if (not config.use_relative_attention_position_embedding) and (not config.use_alibi_position_embedding):
            raise ValueError("Must use either relative position embeddings or alibi position embeddings")
        if config.use_relative_attention_position_embedding and config.use_alibi_position_embedding:
            raise ValueError(
                "Must use either relative position embeddings or alibi position embeddings, cannot specify both")

        if config.use_relative_attention_position_embedding:
            if config.position_bias_per_layer:
                if self.is_encoder_decoder:
                    self.layers = nn.ModuleList(
                        [ATMT5CE4DecoderLayer(config, has_relative_attention_bias=True, mode=(i % 2)) for i in
                         range(config.decoder_layers)])
                else:
                    self.layers = nn.ModuleList(
                        [ATMT5CE4DecoderLayer(config, has_relative_attention_bias=True, mode=0) for i in
                         range(config.decoder_layers)])
            else:
                if self.is_encoder_decoder:
                    self.layers = nn.ModuleList(
                        [ATMT5CE4DecoderLayer(config, has_relative_attention_bias=bool(i == 0), mode=(i % 2)) for i in
                         range(config.decoder_layers)])
                else:
                    self.layers = nn.ModuleList(
                        [ATMT5CE4DecoderLayer(config, has_relative_attention_bias=bool(i == 0), mode=0) for i in
                         range(config.decoder_layers)])
        elif config.use_alibi_position_embedding:
            if self.is_encoder_decoder:
                self.layers = nn.ModuleList(
                    [ATMT5CE4DecoderLayer(config, has_alibi_attention_bias=True, mode=(i % 2)) for i in
                     range(config.decoder_layers)])
            else:
                self.layers = nn.ModuleList(
                    [ATMT5CE4DecoderLayer(config, has_alibi_attention_bias=True, mode=0) for i in
                     range(config.decoder_layers)])

        # remove this in preLN
        # self.layernorm_embedding = nn.LayerNorm(config.d_model)
        # add this for preLN
        self.layernorm_output = nn.LayerNorm(config.d_model)

        self.init_weights()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape, inputs_embeds.dtype, past_key_values_length=past_key_values_length
            ).to(self.device)

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            head_mask=None,
            encoder_head_mask=None,
            past_key_values=None,
            inputs_embeds=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        Args:
            input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using :class:`~transformers.BartTokenizer`. See
                :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__`
                for details.

                `What are input IDs? <../glossary.html#input-ids>`__
            attention_mask (:obj:`torch.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                `What are attention masks? <../glossary.html#attention-mask>`__
            encoder_hidden_states (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, encoder_sequence_length, hidden_size)`, `optional`):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
                of the decoder.
            encoder_attention_mask (:obj:`torch.LongTensor` of shape :obj:`(batch_size, encoder_sequence_length)`, `optional`):
                Mask to avoid performing cross-attention on padding tokens indices of encoder input_ids. Mask values
                selected in ``[0, 1]``:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                `What are attention masks? <../glossary.html#attention-mask>`__
            head_mask (:obj:`torch.Tensor` of shape :obj:`(num_layers, num_heads)`, `optional`):
                Mask to nullify selected heads of the attention modules. Mask values selected in ``[0, 1]``:

                - 1 indicates the head is **not masked**,
                - 0 indicates the heas is **masked**.

            encoder_head_mask (:obj:`torch.Tensor` of shape :obj:`(num_layers, num_heads)`, `optional`):
                Mask to nullify selected heads of the attention modules in encoder to avoid performing cross-attention
                on hidden heads. Mask values selected in ``[0, 1]``:

                - 1 indicates the head is **not masked**,
                - 0 indicates the heas is **masked**.

            past_key_values (:obj:`Tuple[Tuple[torch.Tensor]]` of length :obj:`config.n_layers` with each tuple having 2 tuples each of which has 2 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
                Contains precomputed key and value hidden-states of the attention blocks. Can be used to speed up
                decoding.

                If :obj:`past_key_values` are used, the user can optionally input only the last
                :obj:`decoder_input_ids` (those that don't have their past key value states given to this model) of
                shape :obj:`(batch_size, 1)` instead of all :obj:`decoder_input_ids`` of shape :obj:`(batch_size,
                sequence_length)`.
            inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
                Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded
                representation. This is useful if you want more control over how to convert :obj:`input_ids` indices
                into associated vectors than the model's internal embedding lookup matrix.
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
            output_hidden_states (:obj:`bool`, `optional`):
                Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors
                for more detail.
            return_dict (:obj:`bool`, `optional`):
                Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        # print("input len", input_ids.shape)
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
        # print("input embed", inputs_embeds[0,0,:10])

        if self.embed_projection:
            inputs_embeds = self.proj(inputs_embeds)

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_key_values_length
        )

        # expand encoder attention mask
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            encoder_attention_mask = _expand_mask(encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])

        # embed positions
        # positions = self.embed_positions(input_shape, past_key_values_length)

        # hidden_states = inputs_embeds + positions
        hidden_states = inputs_embeds
        # remove this in preLN
        # hidden_states = self.layernorm_embedding(hidden_states)

        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
        next_decoder_cache = () if use_cache else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            assert head_mask.size()[0] == (
                len(self.layers)
            ), f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
        self_attn_position_bias = None
        encoder_attn_position_bias = None


        cm = contextlib.nullcontext()
        with cm:
            for idx, decoder_layer in enumerate(self.layers):
                # print("===== layer", idx)
                # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
                if output_hidden_states:
                    all_hidden_states += (hidden_states,)
                dropout_probability = random.uniform(0, 1)
                if self.training and (dropout_probability < self.layerdrop):
                    continue

                past_key_value = past_key_values[idx] if past_key_values is not None else None

                if getattr(self.config, "gradient_checkpointing", False) and self.training:

                    if use_cache:
                        logger.warn(
                            "`use_cache=True` is incompatible with `config.gradient_checkpointing=True`. Setting "
                            "`use_cache=False`..."
                        )
                        use_cache = False

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            # None for past_key_value
                            return module(*inputs, output_attentions, use_cache)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(decoder_layer),
                        hidden_states,
                        attention_mask,
                        encoder_hidden_states,
                        encoder_attention_mask,
                        head_mask[idx] if head_mask is not None else None,
                        encoder_head_mask[idx] if encoder_head_mask is not None else None,
                        None,
                        self_attn_position_bias,
                        encoder_attn_position_bias,
                    )
                else:

                    layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        encoder_hidden_states=encoder_hidden_states,
                        encoder_attention_mask=encoder_attention_mask,
                        layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                        encoder_layer_head_mask=(encoder_head_mask[idx] if encoder_head_mask is not None else None),
                        past_key_value=past_key_value,
                        self_attn_position_bias=self_attn_position_bias,
                        encoder_attn_position_bias=encoder_attn_position_bias,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                    )
                hidden_states = layer_outputs[0]

                if not self.config.position_bias_per_layer:
                    self_attn_position_bias, encoder_attn_position_bias = layer_outputs[1:3]

                if use_cache:
                    tmp_cache = layer_outputs[3 if output_attentions else 1]
                    if decoder_layer.mode == 1:
                        # since in this model there is no self-attention, we need to pass self-attention from previous
                        # layer so the cross-attention knows the query length
                        tmp_cache = (next_decoder_cache[-1][0], next_decoder_cache[-1][1], tmp_cache[2], tmp_cache[3])
                    next_decoder_cache += (tmp_cache,)
                    # next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)

                if output_attentions:
                    all_self_attns += (layer_outputs[1],)

                    if encoder_hidden_states is not None:
                        all_cross_attentions += (layer_outputs[2],)

        # add this for preLN
        hidden_states = self.layernorm_output(hidden_states)
        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )


@add_start_docstrings(
    "The bare BART Model outputting raw hidden-states without any specific head on top.",
    BART_START_DOCSTRING,
)
class ATMT5CE4Model(ATMT5CE4PretrainedModel):
    def __init__(self, config: ATMT5Config):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        if config.embed_dim is None:
            embed_dim = config.d_model
        else:
            embed_dim = config.embed_dim

        self.is_encoder_decoder = config.is_encoder_decoder

        self.shared = nn.Embedding(vocab_size, embed_dim, padding_idx)

        if self.is_encoder_decoder:
            self.encoder = ATMT5CE4Encoder(config, self.shared)
        if config.share_embeddings:
            self.decoder = ATMT5CE4Decoder(config, self.shared)
        else:
            self.decoder = ATMT5CE4Decoder(config)

        self.init_weights()

    def get_input_embeddings(self):
        return self.decoder.embed_tokens

    def set_input_embeddings(self, value):
        self.shared = value
        if self.is_encoder_decoder:
            self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    def get_encoder(self):
        if self.is_encoder_decoder:
            return self.encoder
        else:
            return None

    def get_decoder(self):
        return self.decoder

    @add_start_docstrings_to_model_forward(BART_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        #    tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="atm-seq2seq",
        output_type=Seq2SeqModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            encoder_outputs=None,
            past_key_values=None,
            inputs_embeds=None,
            decoder_inputs_embeds=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):

        # different to other models, ATMT5CE4 automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None and self.is_encoder_decoder:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput) and self.is_encoder_decoder:
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )
        if not self.is_encoder_decoder:
            encoder_outputs = [None]
        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            encoder_head_mask=head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        if self.is_encoder_decoder:

            return Seq2SeqModelOutput(
                last_hidden_state=decoder_outputs.last_hidden_state,
                past_key_values=decoder_outputs.past_key_values,
                decoder_hidden_states=decoder_outputs.hidden_states,
                decoder_attentions=decoder_outputs.attentions,
                cross_attentions=decoder_outputs.cross_attentions,
                encoder_last_hidden_state=encoder_outputs.last_hidden_state,
                encoder_hidden_states=encoder_outputs.hidden_states,
                encoder_attentions=encoder_outputs.attentions,
            )
        else:
            return Seq2SeqModelOutput(
                last_hidden_state=decoder_outputs.last_hidden_state,
                past_key_values=decoder_outputs.past_key_values,
                decoder_hidden_states=decoder_outputs.hidden_states,
                decoder_attentions=decoder_outputs.attentions,
                cross_attentions=decoder_outputs.cross_attentions,
                encoder_last_hidden_state=None,
                encoder_hidden_states=None,
                encoder_attentions=None,
            )


class MemEfficientLMHead(nn.Module):
    """Roberta Head for masked language modeling."""

    def __init__(self, config, num_embeddings=None, embedding_weights=None):
        super().__init__()

        if num_embeddings is None:
            num_embeddings = config.vocab_size

        if config.embed_dim is not None:
            self.embed_projection = True
            self.proj = nn.Linear(config.hidden_size, config.embed_dim, bias=False)
            self.decoder = nn.Linear(config.embed_dim, num_embeddings, bias=False)
        else:
            self.embed_projection = False
            self.decoder = nn.Linear(config.hidden_size, num_embeddings, bias=False)

        if config.add_lm_head_bias:
            self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        else:
            self.bias = None

        if embedding_weights is not None:
            self.decoder.weight = embedding_weights

    def forward(self, features, **kwargs):
        # project back to size of vocabulary with bias
        if self.embed_projection:
            features = self.proj(features)
        if self.bias is not None:
            return self.decoder(features) + self.bias
        else:
            return self.decoder(features)

    def _tie_weights(self):
        pass


@add_start_docstrings(
    "The BART Model with a language modeling head. Can be used for summarization.", BART_START_DOCSTRING
)
class ATMT5CE4ForConditionalGeneration(ATMT5CE4PretrainedModel):
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [
        r"final_logits_bias",
        r"encoder\.version",
        r"decoder\.version",
        r"lm_head\.weight",
    ]

    def __init__(self, config: ATMT5Config):
        self.is_encoder_decoder = config.is_encoder_decoder

        self.tie_word_embeddings = config.tie_word_embeddings
        if config.tie_word_embeddings:
            # Set to false to avoid HF calling `set_output_embeddings()`, as we control tying manually below
            config.tie_word_embeddings = False

        super().__init__(config)
        self.model = ATMT5CE4Model(config)

        if self.tie_word_embeddings:
            self.lm_head = MemEfficientLMHead(config, num_embeddings=self.model.shared.num_embeddings,
                                              embedding_weights=self.model.shared.weight)
        else:
            self.lm_head = MemEfficientLMHead(config, num_embeddings=self.model.shared.num_embeddings)

        self.init_weights()

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        return new_embeddings

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    @add_start_docstrings_to_model_forward(BART_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    @add_end_docstrings(BART_GENERATION_EXAMPLE)
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            encoder_outputs=None,
            past_key_values=None,
            inputs_embeds=None,
            decoder_inputs_embeds=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should either be in ``[0, ...,
            config.vocab_size]`` or -100 (see ``input_ids`` docstring). Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``.

        Returns:
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )
        # this is for batch inference for decoder only since padding happends on the left
        if not self.is_encoder_decoder and decoder_attention_mask is None \
                and decoder_input_ids is not None:
            decoder_attention_mask = attention_mask

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        lm_logits = self.lm_head(outputs[0])

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        if self.is_encoder_decoder:
            return Seq2SeqLMOutput(
                loss=masked_lm_loss,
                logits=lm_logits,
                past_key_values=outputs.past_key_values,
                decoder_hidden_states=outputs.decoder_hidden_states,
                decoder_attentions=outputs.decoder_attentions,
                cross_attentions=outputs.cross_attentions,
                encoder_last_hidden_state=outputs.encoder_last_hidden_state,
                encoder_hidden_states=outputs.encoder_hidden_states,
                encoder_attentions=outputs.encoder_attentions,
            )
        else:
            return Seq2SeqLMOutput(
                loss=masked_lm_loss,
                logits=lm_logits,
                past_key_values=outputs.past_key_values,
                decoder_hidden_states=outputs.decoder_hidden_states,
                decoder_attentions=outputs.decoder_attentions,
                cross_attentions=outputs.cross_attentions,
                encoder_last_hidden_state=None,
                encoder_hidden_states=None,
                encoder_attentions=None,
            )

    def prepare_inputs_for_generation(
            self,
            decoder_input_ids,
            past=None,
            attention_mask=None,
            head_mask=None,
            use_cache=None,
            encoder_outputs=None,
            **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past
