import math

import torch
from einops import rearrange
from torch import nn as nn, Tensor


class PositionalEncoder(nn.Module):
    def __init__(self, d_model: int, pos_factor: int = 1, dropout: int = 0.0) -> None:
        super().__init__()

        self.d_model = d_model
        self.pos_factor = pos_factor

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, position: int) -> Tensor:
        pe = self.__get_positional_encoding(position)

        return self.dropout(pe)

    def __get_positional_encoding(self, pos: int) -> Tensor:
        position = rearrange(torch.arange(pos), "p -> p 1")
        two_i = rearrange(torch.arange(self.d_model), "p -> 1 p")

        div_term = torch.exp(two_i * -(math.log(10000.0) / self.d_model))
        angle_rates = position * div_term * self.pos_factor

        angle_rates[:, 0::2] = torch.sin(angle_rates[:, 0::2])
        angle_rates[:, 1::2] = torch.cos(angle_rates[:, 1::2])

        return rearrange(angle_rates, "b d -> 1 b d").requires_grad_(False)


# class PositionalEncoder(nn.Module):
#     def __init__(self, d_model: int, pos_factor: int = 1):
#         """
#         _summary_
#
#         Parameters
#         ----------
#         d_model : int
#             _description_
#         pos_factor : int, optional
#             _description_, by default 1
#         """
#
#         super().__init__()
#
#         self.d_model = d_model
#         self.pos_factor = pos_factor
#
#     def forward(self, position: int) -> Tensor:
#         """
#         _summary_
#
#         Parameters
#         ----------
#         position : int
#             _description_
#
#         Returns
#         -------
#         Tensor
#             _description_
#         """
#
#         time = torch.arange(position)
#         half_dim = self.d_model // 2
#         embeddings = math.log(10000) / (half_dim - 1)
#
#         embeddings = torch.exp(torch.arange(half_dim) * -embeddings)
#         embeddings = time[:, None] * embeddings[None, :] * self.pos_factor
#         embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
#         embeddings = embeddings[None, ...]
#
#         return embeddings
