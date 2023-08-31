import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from models.Diffusion.attention import AffineTransformLayer


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_features: int,
        in_channels: int,
        out_channels: int,
        dilatation: int = 1,
        *,
        drop_rate: float = 0.0,
    ) -> None:
        """
        _summary_

        Parameters
        ----------
        in_features : int
            _description_
        in_channels : int
            _description_
        out_channels : int
            _description_
        dilatation : int, optional
            _description_, by default 1
        drop_rate : float, optional
            _description_, by default 0.0
        """

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
        self.dropout = nn.Dropout(p=drop_rate)

    def forward(self, x: Tensor, alpha: Tensor) -> Tensor:
        """
        _summary_

        Parameters
        ----------
        x : Tensor
            _description_
        alpha : Tensor
            _description_

        Returns
        -------
        Tensor
            _description_
        """

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


class FeedForwardNetwork(nn.Module):
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

        # TODO: Testing SiLU (original implementation) with ReLU and SeLU
        ff_network = [
            nn.Linear(in_features, hidden_size),
            nn.SELU(),
            nn.Linear(hidden_size, out_features),
        ]

        if act_before:
            ff_network.insert(0, nn.SELU())

        self.ff_net = nn.Sequential(*ff_network)

    def forward(self, x: Tensor) -> Tensor:
        return self.ff_net(x)
