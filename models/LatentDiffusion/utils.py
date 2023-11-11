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
    Group normalization with support for 32-bit data type.

    Args:
        num_groups (int): Number of groups for normalization.
        num_channels (int): Number of channels in the input.
        eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-5.

    Attributes:
        num_groups (int): Number of groups for normalization.
        num_channels (int): Number of channels in the input.
    """

    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.float()).type(x.dtype)


class FeedForwardNetwork(nn.Module):
    """
    Feedforward neural network with optional dropout.

    Args:
        d_model (int): Input feature dimension.
        d_out (int, optional): Output feature dimension. If not provided, it is set to d_model. Default is None.
        d_mult (int, optional): Multiplier for the hidden layer dimension. Default is 4.
        dropout (float, optional): Dropout probability. Default is 0.0.

    Attributes:
        d_model (int): Input feature dimension.
        d_out (int): Output feature dimension.
        d_mult (int): Multiplier for the hidden layer dimension.
        dropout (float): Dropout probability.
    """

    def __init__(
        self,
        d_model: int,
        d_out: int = None,
        *,
        d_mult: int = 4,
        dropout: float = 0.0,
    ) -> None:
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
    Add noise to an input image tensor based on time steps and alpha_bar values.

    Args:
        x (Tensor): Input image tensor of shape (batch_size, channels, height, width).
        time_step (Tensor): Time step tensor of shape (batch_size, 1).
        alpha_bar (Tensor): Alpha_bar values tensor of shape (batch_size, 1).

    Returns:
        Tuple[Tensor, Tensor]: A tuple containing two tensors:
        - noisy_image (Tensor): Image tensor with added noise, of the same shape as input x.
        - noise (Tensor): Noise tensor used for adding to the input image, of the same shape as input x.)
    """

    sqrt_alpha_bar = rearrange(torch.sqrt(alpha_bar[time_step]), "v -> v 1 1 1")
    sqrt_one_minus_alpha_bar = rearrange(
        torch.sqrt(1 - alpha_bar[time_step]), "v -> v 1 1 1"
    )

    noise = torch.randn_like(x)

    return sqrt_alpha_bar * x + sqrt_one_minus_alpha_bar * noise, noise


def __crop_whitespaces(image: Image) -> Image:
    """
    Crop white spaces from the edges of an input image.

    Args:
        image (Image): An input image to be cropped.

    Returns:
        Image: A cropped image with white spaces removed.
    """

    img_gray = image.convert("L")
    bbox = img_gray.point(lambda p: p < 128 and 255).getbbox()
    image = img_gray.crop(bbox)

    return image


def generate_image(
    image: Tensor,
    path: Optional[Path],
    *,
    color: str,
    labels: Optional[List[str]] = None,
    is_fid: bool = False,
) -> Union[Image, List[Image]]:
    """
    Generate and process images from an input tensor.

    Args:
        image (Tensor): The input tensor representing the image(s).
        path (Optional[Path]): The path to save the image(s). If None, images won't be saved.
        color (str): The desired color for the image(s).
        labels (Optional[List[str]]): List of labels for each image. If provided, labels will be added to images.
        is_fid (bool): If True, return a list of PIL images without processing. Ignored if labels are provided.

    Returns:
        Union[Image, List[Image]]: A processed image or a list of processed images.
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

        combined_images = __combine_images_with_space(images, labels)

        if path is not None:
            __save_image(combined_images, path)

        return combined_images


def __save_image(image: Image, path: Path) -> None:
    """
    Save a PIL image to the specified path, supporting SVG format.

    Args:
        image (Image): The PIL image to be saved.
        path (Path): The path where the image will be saved.

    Returns:
        None
    """

    if path.suffix == ".svg":
        path_bitmap = potrace.Bitmap(image, blacklevel=0.7).trace()
        __backend_svg(image, path, path_bitmap)
    else:
        image.save(path)


def __backend_svg(image: Image, path: Path, path_bitmap: Path) -> None:
    """
    Generate SVG content from a potrace bitmap and save it to a file.

    Args:
        image (Image): The PIL image.
        path (Path): The path where the SVG file will be saved.
        path_bitmap (Path): The potrace bitmap.

    Returns:
        None
    """

    path_data = []
    for curve in path_bitmap:
        start_point = curve.start_point
        path_data.append(f"M{start_point.x},{start_point.y}")

        for segment in curve.segments:
            if segment.is_corner:
                a = segment.c
                path_data.append(f"L{a.x},{a.y}")
                b = segment.end_point
                path_data.append(f"L{b.x},{b.y}")
            else:
                a = segment.c1
                b = segment.c2
                c = segment.end_point
                path_data.append(f"C{a.x},{a.y} {b.x},{b.y} {c.x},{c.y}")

        path_data.append("z")

    path_element = f'<path stroke="none" fill="black" fill-rule="evenodd" d="{"".join(path_data)}"/>'

    svg_content = (
        f'<svg version="1.1" xmlns="http://www.w3.org/2000/svg" '
        f'xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'width="{image.width}" height="{image.height}" '
        f'viewBox="0 0 {image.width} {image.height}">'
    )

    with open(path, mode="w") as svg:
        svg.write(svg_content)
        svg.write(path_element)
        svg.write("</svg>")


def __combine_images_with_space(
    images: List[Image],
    labels: List[str],
    *,
    spacing: int = 20,
) -> Image:
    """
    This function combines a list of images horizontally with an optional spacing between them.

    Args:
        images (List[Image]): A list of PIL images to be combined.
        labels (List[str]): A list of labels associated with each image.
        spacing (int, optional): Spacing (in pixels) between images. Default is 20.

    Returns:
        Image: The combined image.
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
    """
    This function changes the colors of a handwriting based on the specified color.
    It converts the image to RGB mode, processes the pixel data, and returns
    the modified image.

    Args:
        image (Image): A PIL image to change the colors of.
        color (str): The desired color to apply to the handwriting.

    Returns:
        Image: The modified image with adjusted colors.
    """

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
