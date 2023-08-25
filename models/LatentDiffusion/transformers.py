import torch.nn as nn
from einops import rearrange
from torch import Tensor

from .attention import CrossAttention
from .model import FeedForwardNetwork


class BasicTransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_head: int,
        *,
        dropout: float = 0.0,
        d_cond: int = None,
    ) -> None:
        """
        _summary_

        Parameters
        ----------
        d_model : int
            _description_
        n_heads : int
            _description_
        d_head : int
            _description_
        dropout : float, optional
            _description_, by default 0.0
        d_cond : int, optional
            _description_, by default None
        """
        super().__init__()

        self.attention_0 = CrossAttention(
            d_model, n_heads=n_heads, d_head=d_head, dropout=dropout
        )
        self.norm_0 = nn.LayerNorm(d_model)

        self.attention_1 = CrossAttention(
            d_model,
            d_cond=d_cond,
            n_heads=n_heads,
            d_head=d_head,
            dropout=dropout,
        )
        self.norm_1 = nn.LayerNorm(d_model)

        self.ff_net = FeedForwardNetwork(d_model, dropout=dropout)
        self.norm_2 = nn.LayerNorm(d_model)

    def forward(self, x: Tensor, context: Tensor = None) -> Tensor:
        """
        _summary_

        Parameters
        ----------
        x : Tensor
            _description_
        context : Tensor, optional
            _description_, by default None

        Returns
        -------
        Tensor
            _description_
        """
        x = self.attention_0(self.norm_0(x)) + x
        x = self.attention_1(self.norm_1(x), context=context) + x
        x = self.ff_net(self.norm_2(x)) + x

        return x


class SpatialTransformer(nn.Module):
    def __init__(
        self,
        channels: int,
        n_heads: int,
        d_head: int,
        n_layers: int = 1,
        dropout: float = 0.0,
        d_cond: int = None,
    ) -> None:
        """
        _summary_

        Parameters
        ----------
        channels : int
            _description_
        n_heads : int
            _description_
        d_head : int
            _description_
        n_layers : int, optional
            _description_, by default 1
        dropout : float, optional
            _description_, by default 0.0
        d_cond : int, optional
            _description_, by default None
        """
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
