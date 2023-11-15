import itertools
import math

import torch
from torch import Tensor, nn

from .attention import WordAttention


class CharacterEncoder(nn.Module):
    """
    This module encodes sequences of characters by first applying word-level attention to capture
    contextual information within the character sequence. It uses an embedding layer for character
    representation and adds positional encoding to the embedded characters.

    Args:
        in_features (int): The size of the character vocabulary (number of unique characters).
        hidden_size (int): The dimension of the character embeddings and hidden space for attention.
        max_seq_len (int): The maximum sequence length for positional encoding.
    """

    def __init__(self, in_features: int, hidden_size: int, max_seq_len: int) -> None:
        super().__init__()

        self.embedding = nn.Embedding(in_features, hidden_size)
        self.attention = WordAttention(hidden_size, hidden_size)

        self.emb_dim = hidden_size
        self.max_seq_len = max_seq_len
        self.positional_encoding = PositionalEncoder(max_seq_len, hidden_size)()

    def forward(self, x: Tensor) -> Tensor:
        if self.positional_encoding.device != x.device:
            self.positional_encoding = self.positional_encoding.to(x.device)

        x = self.embedding(x)

        x += self.positional_encoding[: x.size(1), :]

        return self.attention(x)


class PositionalEncoder(nn.Module):
    """
    Positional encodings are used to provide information about the position of elements in a sequence.
    This module generates positional encodings based on sine and cosine functions.

    Args:
        d_model (int): The dimension of the model's hidden space.
        d_emb (int): The dimension of the embeddings.
    """

    def __init__(self, d_model: int, d_emb: int) -> None:
        super().__init__()

        self.register_buffer(
            "positional_encodings", self.get_positional_encoding(d_model, d_emb), False
        )

    def forward(self):
        return self.positional_encodings.detach().requires_grad_(False)

    @staticmethod
    def get_positional_encoding(d_model: int, d_emb: int) -> Tensor:
        """
        Generate positional encodings.

        Args:
            d_model (int): The dimension of the model's hidden space.
            d_emb (int): The dimension of the embeddings.

        Returns:
            Tensor: Positional encodings for sequences.
        """

        encodings = torch.zeros(d_model, d_emb)
        for pos, i in itertools.product(range(d_model), range(0, d_emb, 2)):
            encodings[pos, i] = math.sin(pos / (10000 ** (i / d_emb)))
            encodings[pos, i + 1] = math.cos(pos / (10000 ** (i / d_emb)))

        return encodings
