import torch
from einops import einsum, rearrange
from torch import Tensor, nn


class CrossAttention(nn.Module):
    """
    _summary_
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
        """
        _summary_

        Parameters
        ----------
        d_model : int
            _description_
        d_cond : int, optional
            _description_, by default None
        n_heads : int, optional
            _description_, by default 8
        d_head : int, optional
            _description_, by default 64
        dropout : float, optional
            _description_, by default 0.0
        use_flash_attention : bool, optional
            _description_, by default False
        """
        
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
        """
        _summary_

        Parameters
        ----------
        x : Tensor
            _description_
        context : Tensor, optional
            _description_, by default None
        mask : Tensor, optional
            _description_, by default None

        Returns
        -------
        Tensor
            _description_
        """
        
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
        """
        _summary_

        Parameters
        ----------
        q : Tensor
            _description_
        k : Tensor
            _description_
        v : Tensor
            _description_

        Returns
        -------
        Tensor
            _description_

        Raises
        ------
        ValueError
            _description_
        """
        
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
        """
        _summary_

        Parameters
        ----------
        q : Tensor
            _description_
        k : Tensor
            _description_
        v : Tensor
            _description_
        mask : Tensor, optional
            _description_, by default None

        Returns
        -------
        Tensor
            _description_
        """
        
        q, k, v = [
            rearrange(t, "b n (h d) -> (b h) n d", h=self.n_heads) for t in (q, k, v)
        ]

        # noinspection PyTypeChecker
        attention: Tensor = einsum(q, k, "b i d, b j d -> b i j") * self.scale

        if mask is not None:
            mask = rearrange(mask, "b j -> b 1 1 j")
            attention.masked_fill_(~mask, -10000.0)

        attention = torch.softmax(attention, dim=-1)

        # noinspection PyTypeChecker
        out = einsum(attention, v, "b i j, b j d -> b i d")
        out = rearrange(out, "(b h) n d -> b n (h d)", h=self.n_heads)

        return self.to_out(out)


class WordAttention(nn.Module):
    """
    _summary_
    """
    
    def __init__(self, in_features: int, hidden_size: int) -> None:
        """
        _summary_

        Parameters
        ----------
        in_features : int
            _description_
        hidden_size : int
            _description_
        """
        
        super().__init__()

        self.to_q = nn.Linear(in_features, hidden_size)
        self.to_k = nn.Linear(in_features, hidden_size)
        self.to_v = nn.Linear(in_features, hidden_size)

    def forward(self, x: Tensor) -> Tensor:
        """
        _summary_

        Parameters
        ----------
        x : Tensor
            _description_

        Returns
        -------
        Tensor
            _description_
        """
        
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        scores = q @ k.transpose(-2, -1)
        scores = torch.softmax(scores, dim=-1)

        return scores @ v
