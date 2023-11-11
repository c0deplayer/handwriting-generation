from einops import rearrange
from torch import nn, Tensor

from .utils import GroupNorm32


class ResBlock(nn.Module):
    """
    A residual block typically consists of a sequence of layers, including convolutional layers and skip connections.
    Residual blocks are used to enable residual learning, which helps the network learn identity mappings
    and is especially effective in deep networks.

    Args:
        channels (int): The number of input channels.
        d_t_emb (int): The dimension of time embeddings.
        out_channels (int, optional): The number of output channels. If not provided, it defaults to `channels`.
        dropout (float, optional): The dropout rate applied to the output.
    """

    def __init__(
        self,
        channels: int,
        d_t_emb: int,
        *,
        out_channels: int = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if out_channels is None:
            out_channels = channels

        self.in_layers = nn.Sequential(
            GroupNorm32(32, channels),
            nn.SiLU(),
            nn.Conv2d(channels, out_channels, kernel_size=3, padding=1),
        )

        self.emb_layers = nn.Sequential(nn.SiLU(), nn.Linear(d_t_emb, out_channels))

        self.out_layers = nn.Sequential(
            GroupNorm32(32, out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )

        self.skip_connection = (
            nn.Identity()
            if out_channels == channels
            else nn.Conv2d(channels, out_channels, kernel_size=1)
        )

    def forward(self, x: Tensor, emb: Tensor) -> Tensor:
        h = self.in_layers(x)

        emb = self.emb_layers(emb).type(h.dtype)

        h += rearrange(emb, "b v -> b v 1 1")

        h = self.out_layers(h)

        return self.skip_connection(x) + h


class UpSample(nn.Module):
    """
    The UpSample module is used to increase the spatial resolution of the input data by a factor of 2. It employs a
    convolutional transpose operation to achieve the up-sampling.

    Args:
        channels (int): The number of input channels.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()

        self.up_trans = nn.ConvTranspose2d(channels, channels, kernel_size=2, stride=2)

    def forward(self, x: Tensor) -> Tensor:
        return self.up_trans(x)


class DownSample(nn.Module):
    """
    The DownSample module employs a 2D convolution operation with a kernel size of 3x3 and a stride of 2 in order
    to perform downsampling.

    Args:
        channels (int): The number of input channels.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()

        self.down = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.down(x)
