import torch
from einops import rearrange
from torch import nn, Tensor


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
