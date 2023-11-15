import math
import os
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from einops import rearrange, repeat
from torch import Tensor
from torch import nn as nn

from data.utils import uniquify


def reshape_up(x: Tensor, *, factor: int = 2) -> Tensor:
    """
    Reshape a 2D tensor by repeating elements along specific dimensions.

    Args:
        x (Tensor): Input tensor.
        factor (int, optional): The factor by which to reshape the tensor. Defaults to 2.

    Returns:
        Tensor: Reshaped tensor.
    """

    return rearrange(x, "b h (f w) -> b (h f) w", f=factor)


def get_beta_set(device: torch.device) -> Tensor:
    """
    Generate a set of beta values for diffusion models.

    Args:
        device (torch.device): The device (e.g., 'cpu' or 'cuda') on which to create the tensor.

    Returns:
        Tensor: A tensor containing a set of beta values.
    """

    start = 1e-5
    end = 0.4
    num_steps = 60

    return 0.02 + torch.exp(
        torch.linspace(math.log(start), math.log(end), num_steps, device=device)
    )


def get_alphas(batch_size: int, alpha_set: Tensor, *, device: torch.device) -> Tensor:
    """
    Sample alpha values from the given set of alphas for diffusion models.

    Args:
        batch_size (int): The number of alpha samples to generate.
        alpha_set (Tensor): The set of alpha values to sample from.
        device (torch.device): The device (e.g., 'cpu' or 'cuda') on which to create the tensor.

    Returns:
        Tensor: A tensor containing sampled alpha values.
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
    Create a padding mask for text data with specified repeats.

    Args:
        text (Tensor): Text data tensor where padding values are marked as 0.
        repeats (int, optional): The number of repeats for the mask. Defaults to 1.

    Returns:
        Tensor: A padding mask tensor with the specified number of repeats.
    """

    text = torch.eq(text, 0).float()
    text = rearrange(text, "b h -> b 1 1 h")
    return repeat(text, "b 1 1 h -> b 1 1 (h repeats)", repeats=repeats)


def diffusion_step(
    strokes: Tensor, eps: Tensor, beta: Tensor, alpha: Tensor, alpha_next: Tensor
) -> Tensor:
    """
    Perform a new diffusion step in a diffusion model.

    Args:
        strokes (Tensor): Input tensor representing strokes.
        eps (Tensor): Noise tensor for diffusion.
        beta (Tensor): Beta values for diffusion.
        alpha (Tensor): Alpha values for diffusion.
        alpha_next (Tensor): Alpha values for the next diffusion step.

    Returns:
        Tensor: Updated tensor after the diffusion step.
    """

    strokes_minus = (strokes - torch.sqrt(1 - alpha) * eps) / torch.sqrt(1 - beta)
    strokes_minus += torch.randn(strokes.shape, device=strokes.device) * torch.sqrt(
        1 - alpha_next
    )

    return strokes_minus


def generate_stroke_image(
    strokes: np.ndarray,
    *,
    color: str,
    save_path: Optional[str],
    scale: float = 1.0,
) -> Union[plt.Figure, List[plt.Figure]]:
    """
    Generate images from stroke data and optionally save them.

    Args:
        strokes (np.ndarray): Array of stroke data. Each entry is an array with shape (n, 3),
                              where n is the number of points and each row contains (x, y, pen_lift) information.
        color (str): Color of the strokes in the generated images.
        save_path (Optional[str]): If provided, the path where the image will be saved. If None, images are not saved.
        scale (float, optional): Scaling factor for the generated image. Defaults to 1.0.

    Returns:
        Union[plt.Figure, List[plt.Figure]]: If a single image is generated, it returns a single matplotlib figure.
            If multiple images are generated, it returns a list of figures.
    """

    fake_samples = []
    for stroke in strokes:
        positions, pen_lifts = np.cumsum(stroke, axis=0).T[:2], stroke[:, 2].round()

        prev_index = 0
        width, height = np.max(positions, axis=-1) - np.min(positions, axis=-1)
        generated_image = plt.figure(figsize=(scale * width / height, scale))

        for index, value in enumerate(pen_lifts):
            if value:
                plt.plot(
                    positions[0][prev_index : index + 1],
                    positions[1][prev_index : index + 1],
                    color=color,
                )
                prev_index = index + 1

        plt.axis("off")
        if save_path is not None:
            if os.path.isfile(save_path):
                save_path = uniquify(save_path)

            plt.savefig(save_path, bbox_inches="tight", format=save_path.suffix[1:])
        else:
            generated_image.canvas.draw_idle()

        plt.close()
        fake_samples.append(generated_image)

    return fake_samples[0] if len(fake_samples) == 1 else fake_samples


class FeedForwardNetwork(nn.Module):
    """
    Feed-Forward Neural Network (FFN).

    A simple feed-forward neural network with an optional activation function.

    Args:
        in_features (int): The input feature dimension.
        out_features (int): The output feature dimension.
        hidden_size (int, optional): The hidden size of the network. Defaults to 768.
        act_before (bool, optional): If True, apply the activation function before hidden layers.
            Defaults to True.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        hidden_size: int = 768,
        act_before: bool = True,
    ) -> None:
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
        return self.ff_net(x)
