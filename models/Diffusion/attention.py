import math
from typing import Tuple, Union

import torch
from einops import rearrange
from torch import Tensor
from torch import nn as nn
from torch.nn import functional as F

from .encoder import PositionalEncoder
from .utils import FeedForwardNetwork


class PrepareForMultiHeadAttention(nn.Module):
    """
    This module is used to prepare input tensors for multi-head attention.
    It applies a linear transformation to the input to ensure
    that it has the correct dimensions for multi-head attention.

    Args:
        d_model (int): The input feature dimension.
        heads (int): The number of attention heads.
        bias (bool, optional): If True, adds a learnable bias to the linear transformation.
            (default: True)

    Attributes:
        d_model (int): The input feature dimension.
        heads (int): The number of attention heads.
        linear (nn.Linear): Linear layer used for dimension transformation.
    """

    def __init__(self, d_model: int, heads: int, *, bias: bool = True) -> None:
        super().__init__()

        self.d_model = d_model
        self.heads = heads

        self.linear = nn.Linear(d_model, d_model, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear(x)

        return self.split_heads(x)

    def split_heads(self, x: Tensor) -> Tensor:
        """
        Split the tensor into multiple attention heads.

        Args:
            x (Tensor): Transformed input tensor with shape (batch size, sequence length, d_model).

        Returns:
            Tensor: Tensor with shape (batch size, number of heads, sequence length, feature dimension per head).
        """

        return rearrange(
            x,
            "b s (h d) -> b h s d",
            b=x.size(0),
            h=self.heads,
            d=(self.d_model // self.heads),
        )


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention allows the model to focus on different parts of the input sequence in parallel, enabling
    improved modeling of long-range dependencies.

    Args:
        d_model (int): The input feature dimension.
        heads (int): The number of attention heads.
        dropout (float, optional): The dropout probability. (default: 0.1)
        bias (bool, optional): If True, adds a learnable bias to linear transformations.
            (default: True)
        return_weights (bool, optional): If True, returns attention weights in addition to
            the output. (default: True)

    Attributes:
        d_model (int): The input feature dimension.
        heads (int): The number of attention heads.
        return_weights (bool): Whether to return attention weights.
    """

    def __init__(
        self,
        d_model: int,
        heads: int,
        *,
        dropout: float = 0.1,
        bias: bool = True,
        return_weights: bool = True,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.heads = heads
        self.return_weights = return_weights

        self.query = PrepareForMultiHeadAttention(d_model, heads, bias=bias)
        self.key = PrepareForMultiHeadAttention(d_model, heads, bias=bias)
        self.value = PrepareForMultiHeadAttention(d_model, heads, bias=True)

        self.output = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, q: Tensor, k: Tensor, v: Tensor, *, mask: Tensor = None
    ) -> Union[Tuple[Tensor, Tensor], Tensor]:
        query = self.query(q)
        key = self.key(k)
        value = self.value(v)

        attention, attention_weights = self.scaled_dot_product_attention(
            query, key, value, mask=mask
        )

        attention = self.dropout(attention)

        attention = rearrange(
            attention,
            "b h s d -> b s (h d)",
            b=q.size(0),
            s=q.size(1),
            h=self.heads,
            d=(self.d_model // self.heads),
        )

        if self.return_weights:
            return self.output(attention), attention_weights
        else:
            return self.output(attention)

    @staticmethod
    def scaled_dot_product_attention(
        q: Tensor, k: Tensor, v: Tensor, *, mask: Tensor = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute scaled dot-product attention.

        Args:
            q (Tensor): Query tensor of shape (batch size, number of heads, sequence length, feature dimension per head).
            k (Tensor): Key tensor of shape (batch size, number of heads, sequence length, feature dimension per head).
            v (Tensor): Value tensor of shape (batch size, number of heads, sequence length, feature dimension per head).
            mask (Tensor, optional): Optional mask tensor for sequence masking.
                (default: None)

        Returns:
            Tuple[Tensor, Tensor]: The output tensor and the attention weights tensor.
        """

        scores = q @ k.transpose(-2, -1)
        dk = k.size(-1)
        scores /= math.sqrt(dk)

        if mask is not None:
            scores += mask * -1e12

        attention_weights = torch.softmax(scores, dim=-1)

        return attention_weights @ v, attention_weights


class AffineTransformLayer(nn.Module):
    """
    The AffineTransformLayer applies a learnable affine transformation to its input data.
    It consists of two linear layers, one for scaling (gamma) and one for shifting (beta).
    These parameters are learned during training.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        channel_first_input (bool, optional): If True, the input tensor has channels in
            the first dimension (C x H x W). If False, the input tensor has channels
            in the last dimension (H x W x C). (default: False)

    Attributes:
        gamma (nn.Linear): Linear layer for scaling (gamma) transformation.
        beta (nn.Linear): Linear layer for shifting (beta) transformation.
        channel_first_input (bool): Indicates the input tensor format.
    """

    def __init__(
        self, in_features: int, out_features: int, channel_first_input: bool = False
    ) -> None:
        super().__init__()

        self.gamma = nn.Linear(in_features, out_features)
        self.beta = nn.Linear(in_features, out_features)
        self.channel_first_input = channel_first_input

        # * `bias_initializer='ones'` in original implementation *
        self.gamma.bias.data.fill_(1.0)

    def forward(self, x: Tensor, sigma: Tensor) -> Tensor:
        gammas = self.gamma(sigma)
        betas = self.beta(sigma)

        if self.channel_first_input:
            return rearrange(
                rearrange(x, "b h w -> b w h") * gammas + betas, "b h w -> b w h"
            )

        return x * gammas + betas


class AttentionBlock(nn.Module):
    """
    The AttentionBlock applies multi-head self-attention mechanisms to the input data and computes attention weights.
    The block consists of multiple layers, including multi-head attention, positional encoding, feedforward layers,
    and affine transformations.

    Args:
        in_features (int): Number of input features.
        d_model (int): Model dimensionality, representing the size of embedding vectors.
        num_heads (int): Number of attention heads.
        drop_rate (float, optional): Dropout rate to apply between layers. (default: 0.1)
        pos_factor (int, optional): Positional encoding factor. (default: 1)
        swap_channel_layer (bool, optional): If True, input channels are swapped during
            processing. (default: True)

    Attributes:
        swap_channel_layer (bool): Indicates whether the channel swapping layer is used.
        text_pos (Tensor): Textual positional encoding.
        stroke_pos (Tensor): Stroke positional encoding.
        dense_layer (nn.Linear): Linear layer for feature dimension adjustment.
        layer_norm (nn.LayerNorm): Layer normalization.
        affine_0 (AffineTransformLayer): Affine transformation layer 0.
        mha_0 (nn.MultiheadAttention): First multi-head self-attention mechanism.
        affine_1 (AffineTransformLayer): Affine transformation layer 1.
        mha_1 (nn.MultiheadAttention): Second multi-head self-attention mechanism.
        affine_2 (AffineTransformLayer): Affine transformation layer 2.
        ff_network (FeedForwardNetwork): Feedforward network.
        dropout (nn.Dropout): Dropout layer.
        affine_3 (AffineTransformLayer): Affine transformation layer 3.
    """

    def __init__(
        self,
        in_features: int,
        d_model: int,
        num_heads: int,
        *,
        drop_rate: float = 0.1,
        pos_factor: int = 1,
        swap_channel_layer: bool = True,
    ) -> None:
        super().__init__()

        self.swap_channel_layer = swap_channel_layer
        self.text_pos = PositionalEncoder(2000, d_model)()
        self.stroke_pos = PositionalEncoder(2000, d_model, pos_factor=pos_factor)()

        self.dense_layer = nn.Linear(in_features, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6, elementwise_affine=False)
        self.affine_0 = AffineTransformLayer(in_features // 12, d_model)

        self.mha_0 = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.affine_1 = AffineTransformLayer(in_features // 12, d_model)

        self.mha_1 = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.affine_2 = AffineTransformLayer(in_features // 12, d_model)

        self.ff_network = FeedForwardNetwork(d_model, d_model, hidden_size=d_model * 2)
        self.dropout = nn.Dropout(drop_rate)
        self.affine_3 = AffineTransformLayer(in_features // 12, d_model)

    def forward(
        self, x: Tensor, text: Tensor, sigma: Tensor, *, mask: Tensor
    ) -> Tuple[Tensor, Tensor]:
        if self.text_pos.device != x.device:
            self.text_pos = self.text_pos.to(x.device)
            self.stroke_pos = self.stroke_pos.to(x.device)

        text = self.dense_layer(F.silu(text))
        text = self.affine_0(self.layer_norm(text), sigma)
        text_pos = text + self.text_pos[:, : text.size(1)]

        if self.swap_channel_layer:
            x = rearrange(x, "b h w -> b w h")

        mask = rearrange(mask, "b 1 1 h -> b h")
        x_pos = x + self.stroke_pos[:, : x.size(1)]
        x_2, attention = self.mha_0(x_pos, text_pos, text, key_padding_mask=mask)
        x_2 = self.layer_norm(self.dropout(x_2))
        x_2 = self.affine_1(x_2, sigma) + x

        x_2_pos = x_2 + self.stroke_pos[:, : x.size(1)]
        x_3, _ = self.mha_1(x_2_pos, x_2_pos, x_2)
        x_3 = self.layer_norm(x_2 + self.dropout(x_3))
        x_3 = self.affine_2(x_3, sigma)

        x_4 = self.ff_network(x_3)
        x_4 = self.dropout(x_4) + x_3
        output = self.affine_3(self.layer_norm(x_4), sigma)

        return output, attention
