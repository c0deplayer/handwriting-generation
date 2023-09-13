from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms
from PIL.Image import Image
from einops import rearrange
from torch import nn, Tensor

from .activation import GeGLU


class GroupNorm32(nn.GroupNorm):
    """
    _summary_
    """
    
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.float()).type(x.dtype)


class FeedForwardNetwork(nn.Module):
    """
    _summary_
    """
    
    def __init__(
        self,
        d_model: int,
        d_out: int = None,
        *,
        d_mult: int = 4,
        dropout: float = 0.0,
    ) -> None:
        """
        _summary_

        Parameters
        ----------
        d_model : int
            _description_
        d_out : int, optional
            _description_, by default None
        d_mult : int, optional
            _description_, by default 4
        dropout : float, optional
            _description_, by default 0.0
        """
        
        super().__init__()

        if d_out is None:
            d_out = d_model

        self.ff_net = nn.Sequential(
            GeGLU(d_model, d_model * d_mult),
            nn.Dropout(dropout),
            nn.Linear(d_model * d_mult, d_out),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.ff_net(x)


def noise_image(
    x: Tensor, time_step: Tensor, alpha_bar: Tensor
) -> Tuple[Tensor, Tensor]:
    """
    _summary_

    Parameters
    ----------
    x : Tensor
        _description_
    time_step : Tensor
        _description_
    alpha_bar : Tensor
        _description_

    Returns
    -------
    Tuple[Tensor, Tensor]
        _description_
    """
    
    sqrt_alpha_bar = rearrange(torch.sqrt(alpha_bar[time_step]), "v -> v 1 1 1")
    sqrt_one_minus_alpha_bar = rearrange(
        torch.sqrt(1 - alpha_bar[time_step]), "v -> v 1 1 1"
    )
    
    noise = torch.randn_like(x)

    return sqrt_alpha_bar * x + sqrt_one_minus_alpha_bar * noise, noise


def __crop_whitespaces(image: Image) -> Image:
    """
    _summary_

    Parameters
    ----------
    image : Image
        _description_

    Returns
    -------
    Image
        _description_
    """
    
    img_gray = image.convert("L")
    # noinspection PyTypeChecker
    img_gray = np.array(img_gray)
    _, threshold = cv2.threshold(
        img_gray, thresh=0, maxval=255, type=cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    cords = cv2.findNonZero(threshold)
    x, y, w, h = cv2.boundingRect(cords)
    return image.crop((x, y, x + w, y + h))


def save_image(image: Tensor, path: Path) -> None:
    """
    _summary_

    Parameters
    ----------
    image : Tensor
        _description_
    path : Path
        _description_
    """
    
    image = rearrange(image, "1 h w c -> h w c")

    img = torchvision.transforms.ToPILImage()(image)
    img = __crop_whitespaces(img)
    img.save(path)
