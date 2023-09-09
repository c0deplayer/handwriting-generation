from typing import Tuple

import torch
import torch.nn as nn
from einops import reduce, repeat, rearrange
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

        self.alpha = nn.Linear(in_features, out_features)
        self.beta = nn.Linear(in_features, out_features)
        self.kappa = nn.Linear(in_features, out_features)

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

        alpha = torch.exp(self.alpha(x))
        beta = torch.exp(self.beta(x))
        new_kappa = prev_kappa + torch.exp(self.kappa(x))

        alpha = repeat(alpha, "h w -> h w new_axis", new_axis=num_chars)
        beta = repeat(beta, "h w -> h w new_axis", new_axis=num_chars)
        kappa = repeat(new_kappa, "h w -> h w new_axis", new_axis=num_chars)
        u = torch.linspace(0, end=num_chars - 1, steps=num_chars, device=x.device)

        densities = alpha * torch.exp(-beta * (kappa - u) ** 2)
        phi = reduce(densities, "b h w -> b () w", "sum")
        window = torch.bmm(phi, text)

        return phi, new_kappa, window
