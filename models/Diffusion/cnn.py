import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from .attention import AffineTransformLayer


class ConvBlock(nn.Module):
    """
    The ConvBlock consists of multiple convolutional layers with affine transformations and dropout. The block
    processes the input data and provides feature extraction capabilities.

    Args:
        in_features (int): Number of input features.
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        dilatation (int, optional): Dilation factor for convolution. (default: 1)
        drop_rate (float, optional): Dropout rate to apply between layers. (default: 0.0)

    Attributes:
        conv_skip (nn.Conv1d): Skip connection convolutional layer.
        conv_0 (nn.Conv1d): First convolutional layer.
        affine_0 (AffineTransformLayer): Affine transformation layer 0.
        conv_1 (nn.Conv1d): Second convolutional layer.
        affine_1 (AffineTransformLayer): Affine transformation layer 1.
        fc (nn.Linear): Linear fully connected layer.
        affine_2 (AffineTransformLayer): Affine transformation layer 2.
        dropout (nn.Dropout): Dropout layer.
    """

    def __init__(
        self,
        in_features: int,
        in_channels: int,
        out_channels: int,
        dilatation: int = 1,
        *,
        drop_rate: float = 0.0,
    ) -> None:
        super().__init__()

        self.conv_skip = nn.Conv1d(
            in_channels, out_channels, kernel_size=3, padding="same"
        )
        self.conv_0 = nn.Conv1d(
            in_channels,
            out_channels // 2,
            kernel_size=3,
            dilation=dilatation,
            padding="same",
        )
        self.affine_0 = AffineTransformLayer(
            in_features // 4, out_channels // 2, channel_first_input=True
        )
        self.conv_1 = nn.Conv1d(
            out_channels // 2,
            out_channels,
            kernel_size=3,
            dilation=dilatation,
            padding="same",
        )
        self.affine_1 = AffineTransformLayer(
            in_features // 4, out_channels, channel_first_input=True
        )

        self.fc = nn.Linear(out_channels, out_channels)
        self.affine_2 = AffineTransformLayer(in_features // 4, out_channels)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x: Tensor, alpha: Tensor) -> Tensor:
        x_skip = self.conv_skip(x)
        x = self.conv_0(F.silu(x))
        x = self.dropout(self.affine_0(x, alpha))

        x = self.conv_1(F.silu(x))
        x = self.dropout(self.affine_1(x, alpha))

        x = rearrange(x, "b h w -> b w h")
        x = self.fc(F.silu(x))
        x = self.dropout(self.affine_2(x, alpha))
        x = rearrange(x, "b h w -> b w h")

        x += x_skip
        return x
