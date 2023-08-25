from torch import nn

from models.LatentDiffusion.attention import WordAttention


class CharacterEncoder(nn.Module):
    def __init__(self, in_features: int, hidden_size: int, max_seq_len: int):
        super().__init__()

        self.embedding = nn.Embedding(in_features, hidden_size)
        self.attention = WordAttention(hidden_size, hidden_size)

        self.emb_dim = hidden_size
        self.max_seq_len = max_seq_len

        # TODO
