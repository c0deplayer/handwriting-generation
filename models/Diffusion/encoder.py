import math

import torch
from einops import rearrange
from torch import Tensor
from torch import nn as nn


class PositionalEncoder(nn.Module):
    """
    The PositionalEncoder generates positional encodings for use in self-attention mechanisms.
    It provides a way to incorporate position information into the model without the need for recurrence.

    Args:
        position (int): The maximum position to encode.
        d_model (int): The dimension of the model.
        pos_factor (int, optional): A factor to scale the positional encodings. (default: 1)

    Attributes:
        positional_encodings (Tensor): The computed positional encodings.
    """

    def __init__(self, position: int, d_model: int, *, pos_factor: int = 1) -> None:
        super().__init__()

        self.d_model = d_model
        self.pos_factor = pos_factor

        self.register_buffer(
            "positional_encodings",
            self.get_positional_encoding(position, d_model),
            False,
        )

    def forward(self) -> Tensor:
        return self.positional_encodings.detach().requires_grad_(False)

    def get_positional_encoding(self, position: int, d_model: int) -> Tensor:
        """
        Generate positional encodings.

        Args:
            position (int): The maximum position to encode.
            d_model (int): The dimension of the model.

        Returns:
            Tensor: The positional encodings tensor.
        """

        position = rearrange(torch.arange(position), "p -> p 1")
        two_i = rearrange(torch.arange(d_model), "p -> 1 p")

        div_term = torch.exp(two_i * -(math.log(10000.0) / d_model))
        angle_rates = position * div_term * self.pos_factor

        angle_rates[:, 0::2] = torch.sin(angle_rates[:, 0::2])
        angle_rates[:, 1::2] = torch.cos(angle_rates[:, 1::2])

        return rearrange(angle_rates, "b d -> 1 b d")
