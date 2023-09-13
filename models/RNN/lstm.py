from typing import Tuple

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor


class LSTMCell(nn.Module):
    """
    _summary_
    """

    def __init__(
        self, input_size: int, hidden_size: int, *, layer_norm: bool = False
    ) -> None:
        """
        _summary_

        Parameters
        ----------
        input_size : int
            _description_
        hidden_size : int
            _description_
        layer_norm : bool, optional
            _description_, by default False
        """

        super().__init__()

        self.hidden_lin = nn.Linear(hidden_size, 4 * hidden_size)
        self.input_lin = nn.Linear(input_size, 4 * hidden_size, bias=False)

        if layer_norm:
            self.layer_norm = nn.ModuleList(
                [nn.LayerNorm(hidden_size) for _ in range(4)]
            )
            self.layer_norm_c = nn.LayerNorm(hidden_size)
        else:
            self.layer_norm = nn.ModuleList([nn.Identity() for _ in range(4)])
            self.layer_norm_c = nn.Identity()

    def forward(self, x: Tensor, h: Tensor, c: Tensor) -> Tuple[Tensor, Tensor]:
        """
        _summary_

        Parameters
        ----------
        x : Tensor
            _description_
        h : Tensor
            _description_
        c : Tensor
            _description_

        Returns
        -------
        Tuple[Tensor, Tensor]
            _description_
        """

        ifgo = self.hidden_lin(h) + self.input_lin(x)

        ifgo = ifgo.chunk(4, dim=-1)
        ifgo = [self.layer_norm[i](ifgo[i]) for i in range(4)]

        i, f, g, o = ifgo

        c_next = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h_next = torch.sigmoid(o) * torch.tanh(self.layer_norm_c(c_next))

        return h_next, c_next


class LSTM(nn.Module):
    """
    _summary_
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        *,
        n_layers: int = 1,
        layer_norm: bool = False,
        batch_first: bool = False,
    ) -> None:
        """
        _summary_

        Parameters
        ----------
        input_size : int
            _description_
        hidden_size : int
            _description_
        n_layers : int, optional
            _description_, by default 1
        layer_norm : bool, optional
            _description_, by default False
        batch_first : bool, optional
            _description_, by default False
        """

        super().__init__()

        self.n_layers = n_layers
        self.batch_first = batch_first
        self.hidden_size = hidden_size

        self.cells = nn.ModuleList(
            [LSTMCell(input_size, hidden_size, layer_norm=layer_norm)]
            + [
                LSTMCell(hidden_size, hidden_size, layer_norm=layer_norm)
                for _ in range(n_layers - 1)
            ]
        )

    def forward(
        self, x: Tensor, state: Tuple[Tensor, Tensor] = None
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """
        _summary_

        Parameters
        ----------
        x : Tensor
            _description_
        state : Tuple[Tensor, Tensor], optional
            _description_, by default None

        Returns
        -------
        Tuple[Tensor, Tuple[Tensor, Tensor]]
            _description_
        """

        if self.batch_first:
            batch_size, n_steps, _ = x.size()
        else:
            n_steps, batch_size, _ = x.size()

        if state is None:
            h = [
                x.new_zeros(batch_size, self.hidden_size) for _ in range(self.n_layers)
            ]

            c = [
                x.new_zeros(batch_size, self.hidden_size) for _ in range(self.n_layers)
            ]

        else:
            (h, c) = state

            h, c = list(torch.unbind(h)), list(torch.unbind(c))

        out = []
        for t in range(n_steps):
            inp = x[:, t] if self.batch_first else x[t]

            for layer in range(self.n_layers):
                h[layer], c[layer] = self.cells[layer](inp, h[layer], c[layer])
                inp = h[layer]

            out.append(h[-1])

        out = torch.stack(out)
        h = torch.stack(h)
        c = torch.stack(c)

        if self.batch_first:
            out = rearrange(out, "n b h -> b n h")

        return out, (h, c)
