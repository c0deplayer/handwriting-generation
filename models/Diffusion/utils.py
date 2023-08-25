import math

import matplotlib.pyplot as plt
import numpy as np
import torch
from einops import rearrange, repeat
from torch import Tensor


def reshape_up(x: Tensor, *, factor: int = 2) -> Tensor:
    """_summary_
    Args:
        x (Tensor): _description_
        factor (int, optional): _description_. Defaults to 2.
    Returns:
        Tensor: _description_
    """
    return rearrange(x, "b h (f w) -> b (h f) w", f=factor)


def get_beta_set() -> Tensor:
    """_summary_

    Returns:
        Tensor: _description_
    """
    start = 1e-5
    end = 0.4
    num_steps = 60

    return 0.02 + torch.exp(torch.linspace(math.log(start), math.log(end), num_steps))


def get_alphas(batch_size: int, alpha_set: Tensor) -> Tensor:
    """_summary_

    Args:
        batch_size (int): _description_
        alpha_set (Tensor): _description_

    Returns:
        Tensor: _description_
    """
    alpha_indices = torch.randint(
        low=0, high=(len(alpha_set) - 1), size=(batch_size, 1), dtype=torch.int64
    )
    lower_alphas = alpha_set[alpha_indices]
    upper_alphas = alpha_set[alpha_indices + 1]
    alphas = torch.rand(lower_alphas.shape) * (upper_alphas - lower_alphas)
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


def standard_diffusion_step(
    strokes: Tensor, out: Tensor, beta: Tensor, alpha: Tensor, *, add_sigma: bool
) -> Tensor:
    """_summary_

    Args:
        strokes (Tensor): _description_
        out (Tensor): _description_
        beta (Tensor): _description_
        alpha (Tensor): _description_
        add_sigma (bool): _description_

    Returns:
        Tensor: _description_
    """
    strokes = (1 / torch.sqrt(1 - beta)) * (
        strokes - (beta * out / torch.sqrt(1 - alpha))
    )

    if add_sigma:
        strokes += torch.sqrt(beta) * torch.randn(strokes.shape)

    return strokes


def new_diffusion_step(
    strokes: Tensor, eps: Tensor, beta: Tensor, alpha: Tensor, alpha_next: Tensor
) -> Tensor:
    """_summary_

    Args:
        strokes (Tensor): _description_
        eps (Tensor): _description_
        beta (Tensor): _description_
        alpha (Tensor): _description_
        alpha_next (Tensor): _description_

    Returns:
        Tensor: _description_
    """
    strokes_t_minus = (strokes - torch.sqrt(1 - alpha) * eps) / torch.sqrt(1 - beta)
    strokes_t_minus += torch.randn(strokes.shape) * torch.sqrt(1 - alpha_next)

    return strokes_t_minus


def generate_stroke_image(
    strokes: np.ndarray,
    *,
    save_path: str | None = None,
    scale: float = 1.0,
) -> plt.Figure:
    """_summary_

    Args:
        strokes (np.ndarray): _description_
        save_path (str | None, optional): _description_. Defaults to None.
        scale (float, optional): _description_. Defaults to 1.0.

    Returns:
        plt.Figure: _description_
    """
    strokes = strokes.squeeze()
    positions, pen_lifts = np.cumsum(strokes, axis=0).T[:2], strokes[:, 2].round()

    prev_index = 0
    width, height = np.max(positions, axis=-1) - np.min(positions, axis=-1)
    generated_image = plt.figure(figsize=(scale * width / height, scale))

    for index, value in enumerate(pen_lifts):
        if value:
            plt.plot(
                positions[0][prev_index:index],
                positions[1][prev_index:index],
                color="black",
            )
            prev_index = index

    plt.axis("off")
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")

    plt.close()
    return generated_image
