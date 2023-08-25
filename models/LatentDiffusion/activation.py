"""
Originally copied from here, but modified a little bit for my taste.
https://github.com/pfnet-research/deep-table/blob/master/deep_table/nn/layers/activation.py
"""

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class GeGLU(nn.Module):
    """
    References
    ----------
    Shazeer et al., "GLU Variants Improve Transformer," 2020.
        https://arxiv.org/abs/2002.05202
    """

    def __init__(self, d_in: int, d_out: int) -> None:
        super().__init__()

        self.proj = nn.Linear(d_in, d_out * 2)

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-1] % 2 == 0:
            raise RuntimeError("")

        return self.geglu(x)

    def geglu(self, x: Tensor) -> Tensor:
        x, gate = self.proj(x).chunk(2, dim=-1)

        return x * F.gelu(gate)
