import torch.nn.functional as F
from einops import rearrange, einsum
from torch import nn, Tensor


class CrossAttention(nn.Module):
    use_flash_attention: bool = False

    def __init__(
        self,
        d_model: int,
        d_cond: int = None,
        n_heads: int = 8,
        d_head: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.n_heads = n_heads
        self.d_head = d_head
        self.scale = d_head**-0.5

        if d_cond is None:
            d_cond = d_model

        self.to_q = nn.Linear(d_model, d_head * n_heads, bias=False)
        self.to_q = nn.Linear(d_cond, d_head * n_heads, bias=False)
        self.to_q = nn.Linear(d_cond, d_head * n_heads, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(d_head * n_heads, d_model), nn.Dropout(dropout)
        )

        try:
            from flash_attn.flash_attention import FlashAttention

            self.flash = FlashAttention()
            self.flash.softmax_scale = self.scale

        except ImportError:
            self.flash = None

    def forward(
        self, x: Tensor, *, context: Tensor = None, mask: Tensor = None
    ) -> Tensor:
        has_context = context is not None
        if not has_context:
            context = x

        q = self.to_q(x)
        k = self.to_q(context)
        v = self.to_q(context)

        if (
            CrossAttention.use_flash_attention
            and self.flash is not None
            and not has_context
            and self.d_head <= 128
        ):
            return self.flash_attention(q, k, v)
        else:
            return self.normal_attention(q, k, v, mask=mask)

    def flash_attention(self, q: Tensor, k: Tensor, v: Tensor):
        raise NotImplementedError

    def normal_attention(
        self, q: Tensor, k: Tensor, v: Tensor, *, mask: Tensor = None
    ) -> Tensor:
        q, k, v = [
            rearrange(t, "b n (h d) -> (b h) n d", h=self.n_heads) for t in (q, k, v)
        ]

        # noinspection PyTypeChecker
        attention: Tensor = einsum(q, k, "b i d, b j d -> b i j") * self.scale

        if mask is not None:
            mask = rearrange(mask, "b j -> b 1 1 j")
            attention.masked_fill_(~mask, -10000.0)

        attention = F.softmax(attention, dim=-1)

        # noinspection PyTypeChecker
        out = einsum(attention, v, "b i j, b j d -> b i d")
        out = rearrange(out, "(b h) n d -> b n (h d)", h=self.n_heads)

        return self.to_out(out)


class WordAttention(nn.Module):
    def __init__(self, in_features: int, hidden_size: int):
        super().__init__()

        self.q = nn.Linear(in_features, hidden_size)
        self.k = nn.Linear(in_features, hidden_size)
        self.v = nn.Linear(in_features, hidden_size)

    def forward(self, x: Tensor) -> Tensor:
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        scores = q @ k.transpose(-2, -1)
        scores = F.softmax(scores, dim=-1)

        return scores @ v
