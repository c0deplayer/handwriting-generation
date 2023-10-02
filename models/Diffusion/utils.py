import math
import os
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from einops import rearrange, repeat
from torch import Tensor, nn as nn

from data.utils import uniquify


def reshape_up(x: Tensor, *, factor: int = 2) -> Tensor:
    """
    _summary_

    Parameters
    ----------
    x : Tensor
        _description_
    factor : int, optional
        _description_, by default 2

    Returns
    -------
    Tensor
        _description_
    """

    return rearrange(x, "b h (f w) -> b (h f) w", f=factor)


def get_beta_set(*, device: torch.device) -> Tensor:
    """
    _summary_

    Parameters
    ----------
    device : torch.device
        _description_

    Returns
    -------
    Tensor
        _description_
    """

    start = 1e-5
    end = 0.4
    num_steps = 60

    return 0.02 + torch.exp(
        torch.linspace(math.log(start), math.log(end), num_steps, device=device)
    )


def get_alphas(batch_size: int, alpha_set: Tensor, *, device: torch.device) -> Tensor:
    """
    _summary_

    Parameters
    ----------
    batch_size : int
        _description_
    alpha_set : Tensor
        _description_

    Returns
    -------
    Tensor
        _description_
    """

    if alpha_set.device != device:
        alpha_set = alpha_set.to(device=device)

    alpha_indices = torch.randint(
        low=0,
        high=(len(alpha_set) - 1),
        size=(batch_size, 1),
        dtype=torch.int64,
        device=device,
    )
    lower_alphas = alpha_set[alpha_indices]
    upper_alphas = alpha_set[alpha_indices + 1]
    alphas = torch.rand(lower_alphas.shape, device=device) * (
        upper_alphas - lower_alphas
    )
    alphas += lower_alphas
    alphas = rearrange(alphas, "b 1 -> b 1 1")

    return alphas


def create_padding_mask(text: Tensor, repeats: int = 1) -> Tensor:
    """
    _summary_

    Parameters
    ----------
    text : Tensor
        _description_
    repeats : int, optional
        _description_, by default 1

    Returns
    -------
    Tensor
        _description_
    """

    text = torch.eq(text, 0).float()
    text = rearrange(text, "b h -> b 1 1 h")
    return repeat(text, "b 1 1 h -> b 1 1 (h repeats)", repeats=repeats)


def diffusion_step(
    strokes: Tensor, eps: Tensor, beta: Tensor, alpha: Tensor, alpha_next: Tensor
) -> Tensor:
    """
    _summary_

    Parameters
    ----------
    strokes : Tensor
        _description_
    eps : Tensor
        _description_
    beta : Tensor
        _description_
    alpha : Tensor
        _description_
    alpha_next : Tensor
        _description_

    Returns
    -------
    Tensor
        _description_
    """

    strokes_t_minus = (strokes - torch.sqrt(1 - alpha) * eps) / torch.sqrt(1 - beta)
    strokes_t_minus += torch.randn(strokes.shape, device=strokes.device) * torch.sqrt(
        1 - alpha_next
    )

    return strokes_t_minus


def generate_stroke_image(
    strokes: np.ndarray,
    *,
    save_path: Union[str, None],
    scale: float = 1.0,
) -> plt.Figure:
    """
    _summary_

    Parameters
    ----------
    strokes : np.ndarray
        _description_
    save_path : str
        _description_
    scale : float, optional
        _description_, by default 1.0

    Returns
    -------
    plt.Figure
        _description_
    """

    strokes = strokes.squeeze()
    positions, pen_lifts = np.cumsum(strokes, axis=0).T[:2], strokes[:, 2].round()

    prev_index = 0
    width, height = np.max(positions, axis=-1) - np.min(positions, axis=-1)
    generated_image = plt.figure(figsize=(scale * width / height, scale))

    for index, value in enumerate(pen_lifts):
        if value:
            plt.plot(
                positions[0][prev_index : index + 1],
                positions[1][prev_index : index + 1],
                color="black",
            )
            prev_index = index + 1

    plt.axis("off")
    if save_path is not None:
        if os.path.isfile(save_path):
            save_path = uniquify(save_path)

        plt.savefig(save_path, bbox_inches="tight")
    else:
        generated_image.canvas.draw_idle()

    plt.close()
    return generated_image


class FeedForwardNetwork(nn.Module):
    """
    _summary_
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        hidden_size: int = 768,
        act_before: bool = True,
    ) -> None:
        """
        _summary_

        Parameters
        ----------
        in_features : int
            _description_
        out_features : int
            _description_
        hidden_size : int, optional
            _description_, by default 768
        act_before : bool, optional
            _description_, by default True
        """

        super().__init__()
        self.act_before = act_before

        ff_network = [
            nn.Linear(in_features, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, out_features),
        ]

        if act_before:
            ff_network.insert(0, nn.SiLU())

        self.ff_net = nn.Sequential(*ff_network)

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

        return self.ff_net(x)
