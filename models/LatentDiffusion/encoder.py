import itertools
import math

import torch
from torch import nn, Tensor

from .attention import WordAttention


class CharacterEncoder(nn.Module):
    def __init__(self, in_features: int, hidden_size: int, max_seq_len: int) -> None:
        super().__init__()

        self.embedding = nn.Embedding(in_features, hidden_size)
        self.attention = WordAttention(hidden_size, hidden_size)

        self.emb_dim = hidden_size
        self.max_seq_len = max_seq_len
        self.positional_encoding = PositionalEncoder(max_seq_len, hidden_size)()

    def forward(self, x: Tensor) -> Tensor:
        x = self.embedding(x)

        x += self.positional_encoding[: x.size(1), :]

        return self.attention(x)


class PositionalEncoder(nn.Module):
    def __init__(self, d_model: int, d_emb: int) -> None:
        super().__init__()

        self.register_buffer(
            "positional_encodings", self.get_positional_encoding(d_model, d_emb), False
        )

    def forward(self):
        return self.positional_encodings().detach().requires_grad_(False)

    @staticmethod
    def get_positional_encoding(d_model: int, d_emb: int) -> Tensor:
        encodings = torch.zeros(d_model, d_emb)
        for pos, i in itertools.product(range(d_model), range(0, d_emb, 2)):
            encodings[pos, i] = math.sin(pos / (10000 ** (i / d_emb)))
            encodings[pos, i + 1] = math.cos(pos / (10000 ** (i / d_emb)))

        return encodings
