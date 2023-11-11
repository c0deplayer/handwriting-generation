import torch
from einops import einsum, rearrange
from torch import Tensor, nn


class CrossAttention(nn.Module):
    """
    Cross-Attention module for neural networks, optionally supporting Flash Attention.

    This module performs cross-attention between input `x` and a context `context`, producing an output tensor.

    Args:
        d_model (int): The input feature dimension.
        d_cond (int, optional): The dimension of the conditioning context (default is `None`, which uses `d_model`).
        n_heads (int): The number of attention heads.
        d_head (int): The dimension of each attention head.
        dropout (float): Dropout probability.
        use_flash_attention (bool): Whether to use Flash Attention if available.

    Raises:
        ValueError: If the head size is too large for Flash Attention.
    """

    def __init__(
        self,
        d_model: int,
        d_cond: int = None,
        n_heads: int = 8,
        d_head: int = 64,
        dropout: float = 0.0,
        *,
        use_flash_attention: bool = False,
    ) -> None:
        super().__init__()

        self.use_flash_attention = use_flash_attention
        self.n_heads = n_heads
        self.d_head = d_head
        self.scale = d_head**-0.5

        if d_cond is None:
            d_cond = d_model

        self.to_q = nn.Linear(d_model, d_head * n_heads, bias=False)
        self.to_k = nn.Linear(d_cond, d_head * n_heads, bias=False)
        self.to_v = nn.Linear(d_cond, d_head * n_heads, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(d_head * n_heads, d_model), nn.Dropout(dropout)
        )

        try:
            from flash_attn.modules.mha import FlashSelfAttention

            self.flash = FlashSelfAttention(
                softmax_scale=self.scale, attention_dropout=dropout
            )
        except ImportError:
            self.flash = None

    def forward(
        self, x: Tensor, *, context: Tensor = None, mask: Tensor = None
    ) -> Tensor:
        has_context = context is not None
        if not has_context:
            context = x

        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        if (
            self.use_flash_attention
            and self.flash is not None
            and not has_context
            and mask is None
            and self.d_head <= 128
        ):
            return self.flash_attention(q, k, v)
        else:
            return self.normal_attention(q, k, v, mask=mask)

    def flash_attention(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        batch_size, seq_len, _ = q.size()

        qkv = torch.stack((q, k, v), dim=2)
        qkv = qkv.view(batch_size, seq_len, 3, self.n_heads, self.d_head)

        if self.d_head <= 32:
            pad = 32 - self.d_head
        elif self.d_head <= 64:
            pad = 64 - self.d_head
        elif self.d_head <= 128:
            pad = 128 - self.d_head
        else:
            raise ValueError(f"Head size {self.d_head} too large for Flash Attention")

        if pad:
            qkv = torch.cat(
                (qkv, qkv.new_zeros(batch_size, seq_len, 3, self.n_heads, pad)), dim=-1
            )

        out = self.flash(qkv)
        out = out[:, :, :, : self.d_head]
        out = out.reshape(batch_size, seq_len, self.n_heads * self.d_head)

        return self.to_out(out)

    def normal_attention(
        self, q: Tensor, k: Tensor, v: Tensor, *, mask: Tensor = None
    ) -> Tensor:
        q, k, v = [
            rearrange(t, "b n (h d) -> (b h) n d", h=self.n_heads) for t in (q, k, v)
        ]

        attention: Tensor = einsum(q, k, "b i d, b j d -> b i j") * self.scale

        if mask is not None:
            mask = rearrange(mask, "b j -> b 1 1 j")
            attention.masked_fill_(~mask, -10000.0)

        attention = torch.softmax(attention, dim=-1)

        out = einsum(attention, v, "b i j, b j d -> b i d")
        out = rearrange(out, "(b h) n d -> b n (h d)", h=self.n_heads)

        return self.to_out(out)


class WordAttention(nn.Module):
    """
    Word-level Attention module for neural networks.

    This module performs word-level attention on the input tensor `x` to compute attention scores
    and generate weighted sums of values. It is commonly used in natural language processing and
    sequence modeling tasks.

    Args:
        in_features (int): The input feature dimension.
        hidden_size (int): The dimension of the intermediate hidden space for queries, keys, and values.
    """

    def __init__(self, in_features: int, hidden_size: int) -> None:
        super().__init__()

        self.to_q = nn.Linear(in_features, hidden_size)
        self.to_k = nn.Linear(in_features, hidden_size)
        self.to_v = nn.Linear(in_features, hidden_size)

    def forward(self, x: Tensor) -> Tensor:
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        scores = q @ k.transpose(-2, -1)
        scores = torch.softmax(scores, dim=-1)

        return scores @ v
