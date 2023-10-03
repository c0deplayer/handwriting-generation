import os
from pathlib import Path
from typing import Tuple, List, Union

import torch
import torchvision.transforms
from PIL import Image as ImageModule, ImageColor
from PIL.Image import Image
from einops import rearrange
from torch import nn, Tensor

from data.utils import uniquify
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
    bbox = img_gray.point(lambda p: p < 128 and 255).getbbox()
    image = img_gray.crop(bbox)

    return image


def generate_image(image: Tensor, path: Union[Path, None], *, color: str) -> Image:
    """
    _summary_

    Parameters
    ----------
    image : Tensor
        _description_
    path : Path
        _description_
    color : str
        _description_
    """

    if path is not None and os.path.isfile(path):
        path = uniquify(path)

    if image.size(0) == 1:
        image = rearrange(image, "1 h w c -> h w c")

        img = torchvision.transforms.ToPILImage()(image)
        img = __crop_whitespaces(img)
        img = __change_image_colors(img, color=color)

        if path is not None:
            img.save(path)

        return img
    else:
        images = list(image)
        images = [torchvision.transforms.ToPILImage()(img) for img in images]
        images = [__crop_whitespaces(img) for img in images]
        images = [__change_image_colors(img, color=color) for img in images]

        # TODO: Try to improve the combining of images so that the words are at a similar height
        #       (for example, the word "quick" is higher than the end of the word "the")
        combined_images = combine_images_with_space(images)

        if path is not None:
            combined_images.save(path)

        return combined_images


def combine_images_with_space(
    images: List[Image],
    *,
    spacing: int = 20,
) -> Image:
    """
    Combine multiple images with a specified spacing between them.

    Parameters
    ----------
    images : List[Image]
        A list of PIL Images to combine.
    spacing : int, optional
        The spacing (in pixels) between the images, by default 10.

    Returns
    -------
    Image
        The combined PIL Image.
    """

    if not isinstance(images, list):
        raise TypeError(f"Expected images to be a list, got {type(images)}")
    elif not images:
        raise ValueError("Input image list is empty")

    total_width, total_height = images[0].size

    for image in images[1:]:
        w, h = image.size
        total_width += w + spacing
        total_height = max(total_height, h)

    combined_image = ImageModule.new("RGB", (total_width, total_height), "white")

    x_offset = 0
    for img in images:
        combined_image.paste(img, (x_offset, 0))
        x_offset += img.width + spacing

    return combined_image


def __change_image_colors(image: Image, *, color: str) -> Image:
    rgb_image = image.convert("RGB")
    datas = rgb_image.getdata()
    new_image_data = []
    desired_color = ImageColor.getrgb(color)

    for item in datas:
        if item[0] in list(range(200, 256)):
            new_image_data.append((255, 255, 255))
        elif item[0] in list(range(151)) and color != "black":
            new_image_data.append(desired_color)
        else:
            new_image_data.append(item)

    rgb_image.putdata(new_image_data)

    return rgb_image
