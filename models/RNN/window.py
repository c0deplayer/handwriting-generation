import torch
import torch.nn as nn
from einops import reduce, repeat
from torch import Tensor


class GaussianWindow(nn.Module):
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
        self, batch: tuple[Tensor, Tensor, Tensor], *, device: torch.device
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        _summary_

        Parameters
        ----------
        batch : tuple[Tensor, Tensor, Tensor]
            _description_
        device : torch.device
            _description_

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            _description_
        """

        strokes, text, prev_kappa = batch
        num_chars = text.size(1)
        strokes = strokes[:, 0]

        alpha = torch.exp(self.alpha(strokes))
        beta = torch.exp(self.beta(strokes))
        new_kappa = prev_kappa + torch.exp(self.kappa(strokes))

        alpha = repeat(alpha, "h w -> h w new_axis", new_axis=num_chars)
        beta = repeat(beta, "h w -> h w new_axis", new_axis=num_chars)
        kappa = repeat(new_kappa, "h w -> h w new_axis", new_axis=num_chars)
        u = torch.arange(num_chars, device=device)

        densities = alpha * torch.exp(-beta * (kappa - u) ** 2)
        phi = reduce(densities, "b h w -> b () w", "sum")
        window = phi @ text

        return phi, new_kappa, window
