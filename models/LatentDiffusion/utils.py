import os
from pathlib import Path
from typing import Tuple, List, Union, Optional

import potrace
import torch
import torchvision.transforms
from PIL import Image as ImageModule, ImageColor, ImageOps
from PIL.Image import Image
from einops import rearrange
from torch import nn, Tensor

from data.utils import uniquify
from .activation import GeGLU

T2PIL = torchvision.transforms.ToPILImage()


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


def generate_image(
    image: Tensor,
    path: Union[Path, None],
    *,
    color: str,
    labels: Optional[List[str]] = None,
    is_fid: bool = False,
) -> Union[Image, List[Image]]:
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
    labels : Optional[List[str]]
        _description_ by default None
    is_fid : bool
        _description_, by default False
    """

    if path is not None and os.path.isfile(path):
        path = uniquify(path)

    if image.size(0) == 1:
        image = rearrange(image, "1 c h w -> c h w")

        image = T2PIL(image)
        image = __crop_whitespaces(image)
        image = __change_image_colors(image, color=color)

        if path is not None:
            __save_image(image, path)

        return image
    elif is_fid:
        images = list(image)
        images = [T2PIL(image) for image in images]

        return images
    else:
        images = list(image)
        images = [T2PIL(image) for image in images]
        images = [__crop_whitespaces(image) for image in images]
        images = [__change_image_colors(image, color=color) for image in images]
        images = [
            ImageOps.contain(image, size=(256, 64), method=ImageModule.LANCZOS)
            for image in images
        ]

        combined_images = combine_images_with_space(images, labels)

        if path is not None:
            __save_image(combined_images, path)

        return combined_images


def __save_image(image: Image, path: Path) -> None:
    if path.suffix == ".svg":
        bitmap = potrace.Bitmap(image, blacklevel=0.7)
        path_b = bitmap.trace()

        __backend_svg(image, path, path_b)
    else:
        image.save(path)


def __backend_svg(image: Image, path: Path, path_bitmap: Path) -> None:
    with open(path, "w") as fp:
        fp.write(
            '<svg version="1.1"'
            + ' xmlns="http://www.w3.org/2000/svg"'
            + ' xmlns:xlink="http://www.w3.org/1999/xlink"'
            + f' width="{image.width}" height="{image.height}"'
            + f' viewBox="0 0 {image.width} {image.height}">'
        )
        parts = []
        for curve in path_bitmap:
            fs = curve.start_point
            parts.append(f"M{fs.x},{fs.y}")

            for segment in curve.segments:
                if segment.is_corner:
                    a = segment.c
                    parts.append(f"L{a.x},{a.y}")
                    b = segment.end_point
                    parts.append(f"L{b.x},{b.y}")
                else:
                    a = segment.c1
                    b = segment.c2
                    c = segment.end_point
                    parts.append(f"C{a.x},{a.y} {b.x},{b.y} {c.x},{c.y}")

            parts.append("z")

        fp.write(
            f'<path stroke="none" fill="black" fill-rule="evenodd" d="{"".join(parts)}"/>'
        )
        fp.write("</svg>")


def combine_images_with_space(
    images: List[Image],
    labels: List[str],
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
    char_attention = ["g", "j", "p", "q", "y"]
    char_attention_img = []

    if not isinstance(images, list):
        raise TypeError(f"Expected images to be a list, got {type(images)}")
    elif not images:
        raise ValueError("Input image list is empty")

    total_width, total_height = images[0].size

    if any((c in char_attention) for c in labels[0]):
        char_attention_img.append(0)

    for i, image in enumerate(images[1:], start=1):
        w, h = image.size
        total_width += w + spacing
        total_height = max(total_height, h)

        if any((c in char_attention) for c in labels[i]):
            char_attention_img.append(i)

    if char_attention_img:
        combined_image = ImageModule.new(
            "RGB", (total_width, int(total_height * 1.2)), "white"
        )
    else:
        combined_image = ImageModule.new("RGB", (total_width, total_height), "white")

    x_offset = 0
    for i, img in enumerate(images):
        if i in char_attention_img:
            combined_image.paste(img, (x_offset, int(total_height * 0.2)))
        else:
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
        elif item[0] in list(range(176)) and color != "black":
            new_image_data.append(desired_color)
        else:
            new_image_data.append(item)

    rgb_image.putdata(new_image_data)

    return rgb_image
