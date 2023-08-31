from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor


class MixtureDensityNetwork(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, *, bias: float = None
    ) -> None:
        """
        _summary_

        Parameters
        ----------
        in_features: int
            _description_
        out_features : int
            _description_
        bias : float, optional
            _description_, by default None
        """

        super().__init__()

        self.pi = nn.Linear(in_features, out_features)
        self.mu = nn.Linear(in_features, out_features * 2)
        self.sigma = nn.Linear(in_features, out_features * 2)
        self.rho = nn.Linear(in_features, out_features)
        self.eos = nn.Linear(in_features, 1)

        self.bias = bias if bias is not None else 0.0

    def forward(self, x: Tensor) -> Tuple[Tensor, ...]:
        """
        _summary_

        Parameters
        ----------
        x : Tensor
            _description_

        Returns
        -------
        Tuple[Tensor, ...]
            _description_
        """

        pi_hat, sigma_hat = self.pi(x), self.sigma(x)

        pi = torch.softmax(pi_hat * (1 + self.bias), dim=-1)
        sigma = torch.exp(sigma_hat - self.bias)
        mu = self.mu(x)
        rho = torch.tanh(self.rho(x))
        eos = torch.sigmoid(self.eos(x))

        return pi, mu, sigma, rho, eos
