"""
Originally copied from here, but modified a little bit.
https://github.com/pfnet-research/deep-table/blob/master/deep_table/nn/layers/activation.py
"""

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class GeGLU(nn.Module):
    """
    The GeGLU activation is a variant of the Gated Linear Unit (GLU) and is used in neural networks to
    model complex relationships between features.

    References:
    - Shazeer et al., "GLU Variants Improve Transformer," 2020.
      [Paper](https://arxiv.org/abs/2002.05202)

    Args:
        d_in (int): Number of input features.
        d_out (int): Number of output features (twice the input dimension).

    Raises:
        RuntimeError: If the last dimension of the input tensor is not an even number.
    """

    def __init__(self, d_in: int, d_out: int) -> None:
        super().__init__()

        self.proj = nn.Linear(d_in, d_out * 2)

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-1] % 2:
            raise RuntimeError(
                f"The last dimension ({x.shape[-1]}) is not an even number"
            )

        return self.geglu(x)

    def geglu(self, x: Tensor) -> Tensor:
        x, gate = self.proj(x).chunk(2, dim=-1)

        return x * F.gelu(gate)
