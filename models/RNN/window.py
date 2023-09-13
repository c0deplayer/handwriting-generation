from typing import Tuple

import torch
import torch.nn as nn
from einops import reduce, rearrange
from torch import Tensor


class GaussianWindow(nn.Module):
    """
    _summary_
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        """
        _summary_

        Parameters
        ----------
        in_features : int
            _description_
        out_features : int
            _description_
        """

        super().__init__()

        self.abk = nn.Linear(in_features, out_features * 3)

    def forward(
        self, x: Tensor, text: Tensor, prev_kappa: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        _summary_

        Parameters
        ----------
        x : Tensor
            _description_
        text : Tensor
            _description_
        prev_kappa : Tensor
            _description_

        Returns
        -------
        Tuple[Tensor, Tensor, Tensor]
            _description_
        """

        num_chars = text.size(1)
        x = rearrange(x, "b 1 v -> b v")

        abk = torch.exp(self.abk(x))
        alpha, beta, kappa = abk.chunk(3, dim=-1)
        new_kappa = prev_kappa + kappa

        alpha = rearrange(alpha, "h w -> h w 1")
        beta = rearrange(beta, "h w -> h w 1")
        kappa = rearrange(new_kappa, "h w -> h w 1")
        u = torch.linspace(0, end=num_chars - 1, steps=num_chars, device=x.device)

        densities = alpha * torch.exp(-beta * (kappa - u) ** 2)
        phi = reduce(densities, "b h w -> b () w", "sum")
        window = phi @ text

        return phi, new_kappa, window
