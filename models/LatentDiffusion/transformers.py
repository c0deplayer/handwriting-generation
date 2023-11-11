import torch.nn as nn
from einops import rearrange
from torch import Tensor

from .attention import CrossAttention
from .utils import FeedForwardNetwork


class BasicTransformerBlock(nn.Module):
    """
    The BasicTransformerBlock consists of two layers of cross-attention and one feedforward layer,
    each followed by layer normalization. It is a fundamental component of a transformer architecture.

    Args:
        d_model (int): The dimension of model embeddings.
        n_heads (int): The number of self-attention heads.
        d_head (int): The dimension of each self-attention head.
        dropout (float, optional): The dropout probability. Default is 0.0.
        d_cond (int, optional): The dimension of the conditioning context. Default is None.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_head: int,
        *,
        dropout: float = 0.0,
        d_cond: int = None,
    ) -> None:
        super().__init__()

        self.attention_0 = CrossAttention(
            d_model,
            n_heads=n_heads,
            d_head=d_head,
            dropout=dropout,
            use_flash_attention=False,
        )
        self.norm_0 = nn.LayerNorm(d_model)

        self.attention_1 = CrossAttention(
            d_model,
            d_cond=d_cond,
            n_heads=n_heads,
            d_head=d_head,
            dropout=dropout,
            use_flash_attention=False,
        )
        self.norm_1 = nn.LayerNorm(d_model)

        self.ff_net = FeedForwardNetwork(d_model, dropout=dropout)
        self.norm_2 = nn.LayerNorm(d_model)

    def forward(self, x: Tensor, context: Tensor = None) -> Tensor:
        x = self.attention_0(self.norm_0(x)) + x
        x = self.attention_1(self.norm_1(x), context=context) + x
        x = self.ff_net(self.norm_2(x)) + x

        return x


class SpatialTransformer(nn.Module):
    """
    The SpatialTransformer applies a series of basic transformer blocks to the input tensor,
    followed by projection layers. It allows a neural network to learn how to perform spatial transformations
    on the input image in order to enhance the geometric invariance of the model. For example,
    it can crop a region of interest, scale and correct the orientation of an image.

    Args:
        channels (int): The number of input channels.
        n_heads (int): The number of self-attention heads in each transformer block.
        d_head (int): The dimension of each self-attention head.
        n_layers (int, optional): The number of transformer blocks to apply. Default is 1.
        dropout (float, optional): The dropout probability. Default is 0.0.
        d_cond (int, optional): The dimension of the conditioning context. Default is None.
    """

    def __init__(
        self,
        channels: int,
        n_heads: int,
        d_head: int,
        *,
        n_layers: int = 1,
        dropout: float = 0.0,
        d_cond: int = None,
    ) -> None:
        super().__init__()

        self.channels = channels

        self.norm = nn.GroupNorm(
            num_groups=32, num_channels=channels, eps=1e-6, affine=True
        )
        self.proj_in = nn.Conv2d(
            channels, n_heads * d_head, kernel_size=1, stride=1, padding=0
        )
        self.transformers = nn.ModuleList(
            [
                BasicTransformerBlock(
                    n_heads * d_head,
                    n_heads,
                    d_head,
                    dropout=dropout,
                    d_cond=d_cond,
                )
                for _ in range(n_layers)
            ]
        )
        self.proj_out = nn.Conv2d(
            n_heads * d_head, channels, kernel_size=1, stride=1, padding=0
        )

        for p in self.proj_out.parameters():
            p.detach().zero_()

    def forward(self, x: Tensor, context: Tensor = None) -> Tensor:
        _, _, h, w = x.shape
        x_in = x

        x = self.norm(x)
        x = self.proj_in(x)
        x = rearrange(x, "b c h w -> b (h w) c")

        for transformer in self.transformers:
            x = transformer(x, context=context)

        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
        x = self.proj_out(x)

        return x + x_in
