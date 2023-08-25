import math

import torch.nn.functional as F
from einops import einsum
from torch import nn, Tensor


class GroupNorm32(nn.GroupNorm):
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.float()).type(x.dtype)


def scaled_dot_product_attention(qkv: Tensor, heads: int) -> Tensor:
    bs, width, length = qkv.size()
    if width % (3 * heads) != 0:
        raise RuntimeError("Width must be divisible by 3 * heads")

    channels = width // (3 * heads)
    q, v, k = qkv.chunk(3, dim=1)
    scale = 1 / math.sqrt(math.sqrt(channels))
    weight = einsum(
        (q * scale).view(bs * heads, channels, length),
        (k * scale).view(bs * heads, channels, length),
        "b c t, b c s -> b t s",
    )
    weight = F.softmax(weight.float(), dim=-1, dtype=weight.dtype)

    return einsum(
        weight,
        v.reshape(bs * heads, channels, length),
        "b t s, b c s -> b c t",
    ).reshape(bs, -1, length)
