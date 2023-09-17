import math
from typing import Tuple, Union

import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor


class PrepareForMultiHeadAttention(nn.Module):
    """
    _summary_
    """
    
    def __init__(self, d_model: int, heads: int, *, bias: bool = True) -> None:
        """
        _summary_

        Parameters
        ----------
        d_model : int
            _description_
        heads : int
            _description_
        bias : bool, optional
            _description_, by default True
        """
        
        super().__init__()

        self.d_model = d_model
        self.heads = heads

        self.linear = nn.Linear(d_model, d_model, bias=bias)

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
        
        x = self.linear(x)

        return self.split_heads(x)

    def split_heads(self, x: Tensor) -> Tensor:
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
        
        return rearrange(
            x,
            "b s (h d) -> b h s d",
            b=x.size(0),
            h=self.heads,
            d=(self.d_model // self.heads),
        )


class MultiHeadAttention(nn.Module):
    """
    _summary_
    """
    
    def __init__(
        self,
        d_model: int,
        heads: int,
        *,
        dropout: float = 0.1,
        bias: bool = True,
        return_weights: bool = True,
    ) -> None:
        """
        _summary_

        Parameters
        ----------
        d_model : int
            _description_
        heads : int
            _description_
        dropout : float, optional
            _description_, by default 0.1
        bias : bool, optional
            _description_, by default True
        return_weights : bool, optional
            _description_, by default True
        """
        
        super().__init__()

        self.d_model = d_model
        self.heads = heads
        self.return_weights = return_weights

        self.query = PrepareForMultiHeadAttention(d_model, heads, bias=bias)
        self.key = PrepareForMultiHeadAttention(d_model, heads, bias=bias)
        self.value = PrepareForMultiHeadAttention(d_model, heads, bias=True)

        self.output = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, q: Tensor, k: Tensor, v: Tensor, *, mask: Tensor = None
    ) -> Union[Tuple[Tensor, Tensor], Tensor]:
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
        Union[Tuple[Tensor, Tensor], Tensor]
            _description_
        """
        
        query = self.query(q)
        key = self.key(k)
        value = self.value(v)

        attention, attention_weights = self.scaled_dot_product_attention(
            query, key, value, mask=mask
        )

        attention = self.dropout(attention)

        attention = rearrange(
            attention,
            "b h s d -> b s (h d)",
            b=q.size(0),
            s=q.size(1),
            h=self.heads,
            d=(self.d_model // self.heads),
        )

        if self.return_weights:
            return self.output(attention), attention_weights
        else:
            return self.output(attention)

    @staticmethod
    def scaled_dot_product_attention(
        q: Tensor, k: Tensor, v: Tensor, *, mask: Tensor = None
    ) -> Tuple[Tensor, Tensor]:
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
        Tuple[Tensor, Tensor]
            _description_
        """

        scores = q @ k.transpose(-2, -1)
        dk = k.size(-1)
        scores /= math.sqrt(dk)

        if mask is not None:
            scores += mask * -1e12

        attention_weights = F.softmax(scores, dim=-1)

        return attention_weights @ v, attention_weights


class AffineTransformLayer(nn.Module):
    """
    _summary_
    """
    
    def __init__(
        self, in_features: int, out_features: int, channel_first_input: bool = False
    ) -> None:
        """
        _summary_

        Parameters
        ----------
        in_features : int
            _description_
        out_features : int
            _description_
        channel_first_input : bool, optional
            _description_, by default False
        """

        super().__init__()

        self.gamma = nn.Linear(in_features, out_features)
        self.beta = nn.Linear(in_features, out_features)
        self.channel_first_input = channel_first_input

        # * `bias_initializer='ones'` in original implementation *
        self.gamma.bias.data.fill_(1.0)

    def forward(self, x: Tensor, sigma: Tensor) -> Tensor:
        """
        _summary_

        Parameters
        ----------
        x : Tensor
            _description_
        sigma : Tensor
            _description_

        Returns
        -------
        Tensor
            _description_
        """

        gammas = self.gamma(sigma)
        betas = self.beta(sigma)

        if self.channel_first_input:
            return rearrange(
                rearrange(x, "b h w -> b w h") * gammas + betas, "b h w -> b w h"
            )

        return x * gammas + betas
