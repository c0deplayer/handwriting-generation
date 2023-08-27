from einops import rearrange
from torch import nn, Tensor

from .utils import GroupNorm32


class ResBlock(nn.Module):
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
    """Learned 2x up-sample without padding"""

    def __init__(self, channels: int) -> None:
        super().__init__()

        self.conv = nn.ConvTranspose2d(channels, channels, kernel_size=5, stride=2)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class DownSample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()

        self.op = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.op(x)
