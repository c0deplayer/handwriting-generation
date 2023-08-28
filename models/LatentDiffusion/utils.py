import torch
from einops import rearrange
from torch import nn, Tensor

from .activation import GeGLU


class GroupNorm32(nn.GroupNorm):
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.float()).type(x.dtype)


def noise_image(
    x: Tensor, time_step: Tensor, alpha_bar: Tensor
) -> tuple[Tensor, Tensor]:
    sqrt_alpha_bar = rearrange(torch.sqrt(alpha_bar[time_step]), "v -> v 1 1 1")
    sqrt_one_minus_alpha_bar = rearrange(
        torch.sqrt(1 - alpha_bar[time_step]), "v -> v 1 1 1"
    )
    noise = torch.randn_like(x)

    return sqrt_alpha_bar * x + sqrt_one_minus_alpha_bar * noise, noise


class FeedForwardNetwork(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_out: int = None,
        *,
        d_mult: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if d_out is None:
            d_out = d_model

        self.ff_net = nn.Sequential(
            nn.Linear(d_model, d_model * d_mult),
            GeGLU(d_model, d_model * d_mult),
            nn.Dropout(dropout),
            nn.Linear(d_model * d_mult, d_out),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.ff_net(x)
