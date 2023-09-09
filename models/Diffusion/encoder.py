import math

import torch
from einops import rearrange
from torch import nn as nn, Tensor


class PositionalEncoder(nn.Module):
    """
    _summary_
    """
    
    def __init__(self, position: int, d_model: int, *, pos_factor: int = 1) -> None:
        """
        _summary_

        Parameters
        ----------
        position : int
            _description_
        d_model : int
            _description_
        pos_factor : int, optional
            _description_, by default 1
        """
        super().__init__()

        self.d_model = d_model
        self.pos_factor = pos_factor

        self.register_buffer(
            "positional_encodings",
            self.get_positional_encoding(position, d_model),
            False,
        )

    def forward(self) -> Tensor:
        """
        _summary_

        Returns
        -------
        Tensor
            _description_
        """
        
        return self.positional_encodings.detach().requires_grad_(False)

    def get_positional_encoding(self, position: int, d_model: int) -> Tensor:
        """
        _summary_

        Parameters
        ----------
        position : int
            _description_
        d_model : int
            _description_

        Returns
        -------
        Tensor
            _description_
        """
        
        position = rearrange(torch.arange(position), "p -> p 1")
        two_i = rearrange(torch.arange(d_model), "p -> 1 p")

        div_term = torch.exp(two_i * -(math.log(10000.0) / d_model))
        angle_rates = position * div_term * self.pos_factor

        angle_rates[:, 0::2] = torch.sin(angle_rates[:, 0::2])
        angle_rates[:, 1::2] = torch.cos(angle_rates[:, 1::2])

        return rearrange(angle_rates, "b d -> 1 b d")
