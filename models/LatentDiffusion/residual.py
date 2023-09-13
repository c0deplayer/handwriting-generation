from einops import rearrange
from torch import nn, Tensor

from .utils import GroupNorm32


class ResBlock(nn.Module):
    """
    _summary_
    """
    
    def __init__(
        self,
        channels: int,
        d_t_emb: int,
        *,
        out_channels: int = None,
        dropout: float = 0.0,
    ) -> None:
        """
        _summary_

        Parameters
        ----------
        channels : int
            _description_
        d_t_emb : int
            _description_
        out_channels : int, optional
            _description_, by default None
        dropout : float, optional
            _description_, by default 0.0
        """
        
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
        """
        _summary_

        Parameters
        ----------
        x : Tensor
            _description_
        emb : Tensor
            _description_

        Returns
        -------
        Tensor
            _description_
        """
        
        h = self.in_layers(x)

        emb = self.emb_layers(emb).type(h.dtype)

        h += rearrange(emb, "b v -> b v 1 1")

        h = self.out_layers(h)

        return self.skip_connection(x) + h


class UpSample(nn.Module):
    """Learned 2x up-sample without padding"""

    def __init__(self, channels: int) -> None:
        """
        _summary_

        Parameters
        ----------
        channels : int
            _description_
        """
        
        super().__init__()

        self.up_trans = nn.ConvTranspose2d(channels, channels, kernel_size=2, stride=2)

    def forward(self, x: Tensor) -> Tensor:
        return self.up_trans(x)


class DownSample(nn.Module):
    def __init__(self, channels: int) -> None:
        """
        _summary_

        Parameters
        ----------
        channels : int
            _description_
        """
        
        super().__init__()

        self.down = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.down(x)
