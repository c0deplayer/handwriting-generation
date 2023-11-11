import math
from typing import Tuple, List, Union, Optional

import torch
import torch.nn as nn
from einops import repeat, rearrange
from torch import Tensor

from .encoder import CharacterEncoder
from .residual import ResBlock, DownSample, UpSample
from .transformers import SpatialTransformer
from .utils import GroupNorm32


class TimestepEmbedSequential(nn.Sequential):
    """
    A custom sequential module that extends the behavior of a standard PyTorch `nn.Sequential`.
    This module processes the input data through its submodules, which can include `ResBlock` and
    `SpatialTransformer` layers.

    Args:
        x (Tensor): The input data.
        t_emb (Tensor): The timestep embedding.
        context (Tensor, optional): The context data for spatial transformers. Default is None.

    Returns:
        Tensor: The output of the sequential processing.
    """

    def forward(self, x: Tensor, t_emb: Tensor, context: Tensor = None) -> Tensor:
        for layer in self:
            if isinstance(layer, ResBlock):
                x = layer(x, t_emb)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, context=context)
            else:
                x = layer(x.float())

        return x


class UNetModel(nn.Module):
    """
    UNet-based neural network architecture for image generation.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        channels (int): Base number of channels.
        res_layers (int): Number of residual layers.
        vocab_size (int): Vocabulary size for character encoding.
        attention_levels (Tuple[int]): Tuple of attention levels.
        channel_multipliers (Tuple[int]): Tuple of channel multipliers.
        heads (int): Number of attention heads.
        d_cond (int, optional): Dimension of cross-attention conditioning. Default is 768.
        dropout (float, optional): Dropout rate. Default is 0.0.
        n_style_classes (int, optional): Number of style classes. Default is None.
        tf_layers (int, optional): Number of spatial transformer layers. Default is 1.
        max_seq_len (int, optional): Maximum sequence length. Default is 20.

    Attributes:
        input_blocks (nn.ModuleList): List of input block modules.
        middle_block (TimestepEmbedSequential): Middle block module.
        output_blocks (nn.ModuleList): List of output block modules.
        time_embed (nn.Sequential): Time embedding layer.
        label_emb (nn.Embedding): Label embedding layer.
        word_emb (CharacterEncoder): Character embedding layer.
        out (nn.Sequential): Output layer.

    Methods:
        forward(x, time_steps, context=None, writer_id=None, interpolation=False, mix_rate=None):
            Forward pass of the model.
        time_step_embedding(time_steps, max_period=10000, repeat_only=False):
            Compute the time step embedding.
        interpolation(writer_id, t_emb, mix_rate=1.0):
            Interpolate between writer styles.

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        channels: int,
        res_layers: int,
        vocab_size: int,
        attention_levels: Tuple[int],
        channel_multipliers: Tuple[int],
        heads: int,
        d_cond: int = 768,
        *,
        dropout: float = 0.0,
        n_style_classes: int = None,
        tf_layers: int = 1,
        max_seq_len: int = 20,
    ):
        super().__init__()
        if d_cond is None:
            raise TypeError(
                "Expected dimension of cross_attention conditioning to be int, got None"
            )

        self.channels = channels
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.n_style_classes = n_style_classes

        self.word_emb = CharacterEncoder(vocab_size, d_cond, max_seq_len)

        levels = len(channel_multipliers)

        d_time_emb = channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(channels, d_time_emb),
            nn.SiLU(),
            nn.Linear(d_time_emb, d_time_emb),
        )
        if n_style_classes is not None:
            self.label_emb = nn.Embedding(n_style_classes, d_time_emb)

        input_block_channels = [channels]
        channels_list = [channels * m for m in channel_multipliers]

        input_block_channels = self.__init_input_blocks__(
            attention_levels,
            channels,
            channels_list,
            d_cond,
            d_time_emb,
            dropout,
            heads,
            in_channels,
            input_block_channels,
            levels,
            res_layers,
            tf_layers,
        )

        self.__init_middle_blocks__(
            channels, d_cond, d_time_emb, dropout, heads, tf_layers
        )

        self.__init_output_blocks__(
            attention_levels,
            channels,
            channels_list,
            d_cond,
            d_time_emb,
            dropout,
            heads,
            input_block_channels,
            levels,
            res_layers,
            tf_layers,
        )

        self.out = nn.Sequential(
            GroupNorm32(32, channels),
            nn.SiLU(),
            nn.Conv2d(channels, out_channels, kernel_size=3, padding=1),
        )

    def __init_input_blocks__(
        self,
        attention_levels: Tuple[int],
        channels: int,
        channels_list: List[int],
        d_cond: int,
        d_time_emb: int,
        dropout: float,
        heads: int,
        in_channels: int,
        input_block_channels: List[int],
        levels: int,
        res_layers: int,
        tf_layers: int,
    ) -> List[int]:
        self.input_blocks = nn.ModuleList()
        self.input_blocks.append(
            TimestepEmbedSequential(
                nn.Conv2d(in_channels, channels, kernel_size=3, padding=1)
            )
        )

        for i in range(levels):
            for _ in range(res_layers):
                layers = [
                    ResBlock(
                        channels,
                        d_time_emb,
                        out_channels=channels_list[i],
                        dropout=dropout,
                    )
                ]
                channels = channels_list[i]

                if i in attention_levels:
                    layers.append(
                        SpatialTransformer(
                            channels,
                            heads,
                            channels // heads,
                            n_layers=tf_layers,
                            dropout=dropout,
                            d_cond=d_cond,
                        )
                    )

                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_channels.append(channels)

            if i != levels - 1:
                self.input_blocks.append(TimestepEmbedSequential(DownSample(channels)))
                input_block_channels.append(channels)

        return input_block_channels

    def __init_middle_blocks__(
        self,
        channels: int,
        d_cond: int,
        d_time_emb: int,
        dropout: float,
        heads: int,
        tf_layers: int,
    ) -> None:
        self.middle_block = TimestepEmbedSequential(
            ResBlock(channels, d_time_emb),
            SpatialTransformer(
                channels,
                heads,
                channels // heads,
                n_layers=tf_layers,
                dropout=dropout,
                d_cond=d_cond,
            ),
            ResBlock(channels, d_time_emb),
        )

    def __init_output_blocks__(
        self,
        attention_levels: Tuple[int],
        channels: int,
        channels_list: List[int],
        d_cond: int,
        d_time_emb: int,
        dropout: float,
        heads: int,
        input_block_channels: List[int],
        levels: int,
        res_layers: int,
        tf_layers: int,
    ) -> None:
        self.output_blocks = nn.ModuleList()

        for i in reversed(range(levels)):
            for j in range(res_layers + 1):
                layers = [
                    ResBlock(
                        channels + input_block_channels.pop(),
                        d_time_emb,
                        out_channels=channels_list[i],
                    )
                ]
                channels = channels_list[i]

                if i in attention_levels:
                    layers.append(
                        SpatialTransformer(
                            channels,
                            heads,
                            channels // heads,
                            n_layers=tf_layers,
                            dropout=dropout,
                            d_cond=d_cond,
                        )
                    )

                if i != 0 and j == res_layers:
                    layers.append(UpSample(channels))

                self.output_blocks.append(TimestepEmbedSequential(*layers))

    def time_step_embedding(
        self, time_steps: Tensor, *, max_period: int = 10000, repeat_only: bool = False
    ) -> Tensor:
        """
        Compute the time step embedding.

        Args:
            time_steps (Tensor): Timestep information.
            max_period (int, optional): Maximum period. Default is 10000.
            repeat_only (bool, optional): Whether to repeat the timestep information only. Default is False.

        Returns:
            Tensor: Time step embedding.
        """

        if repeat_only:
            return repeat(time_steps, "b -> b d", d=self.channels)

        half = self.channels // 2
        half_zeros = torch.arange(start=0, end=half, dtype=torch.float32)

        frequencies = torch.exp(
            -math.log(max_period) * half_zeros / half,
        ).to(device=time_steps.device)

        args = rearrange(time_steps, "v -> v 1").float() * rearrange(
            frequencies, "v -> 1 v"
        )

        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

        if self.channels % 2:
            return torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        else:
            return embedding

    def forward(
        self,
        x: Tensor,
        time_steps: Tensor,
        *,
        context: Tensor = None,
        writer_id: Optional[Union[Tensor, Tuple[int, ...]]] = None,
        interpolation: bool = False,
        mix_rate: float = None,
    ) -> Tensor:
        if writer_id is not None and self.n_style_classes is None:
            raise RuntimeError(
                "Writer_id must be specified if and only if the model is class-conditional"
            )

        x_input_block = []
        t_emb = self.time_step_embedding(time_steps)
        t_emb = self.time_embed(t_emb)

        if (
            self.n_style_classes is not None
            and writer_id is not None
            and writer_id.size(0) != x.size(0)
        ):
            raise RuntimeError(
                f"Expected size to be {x.size(0)}, got {writer_id.size(0)}"
            )

        if interpolation:
            if mix_rate is None:
                raise ValueError(f"Invalid mix_rate value: {mix_rate}")

            t_emb = self.interpolation(writer_id, t_emb, mix_rate=mix_rate)
        elif writer_id is not None:
            t_emb = t_emb + self.label_emb(writer_id)

        if context is not None:
            context = self.word_emb(context)

        for module in self.input_blocks:
            x = module(x, t_emb, context)
            x_input_block.append(x)

        x = self.middle_block(x, t_emb, context)

        for module in self.output_blocks:
            x = torch.cat([x, x_input_block.pop()], dim=1)
            x = module(x, t_emb, context)

        return self.out(x)

    def interpolation(
        self,
        writer_id: Tuple[int, int],
        t_emb: Tensor,
        *,
        mix_rate: float = 1.0,
    ) -> Tensor:
        """
        Interpolate between writer styles.

        Args:
            writer_id (Tuple[int, int]): Writer IDs for interpolation.
            t_emb (Tensor): Time step embedding.
            mix_rate (float, optional): Mixing rate. Default is 1.0.

        Returns:
            Tensor: Interpolated time step embedding.
        """

        if not isinstance(writer_id, tuple):
            raise TypeError(
                f"Expected writer_id to be tuple for interpolation, got {type(writer_id).__name__}"
            )
        if writer_id[0] == writer_id[1]:
            raise ValueError("Writer IDs must be unique")

        s1, s2 = writer_id
        y1 = torch.tensor([s1], dtype=torch.long, device=t_emb.device)
        y2 = torch.tensor([s2], dtype=torch.long, device=t_emb.device)

        y1 = self.label_emb(y1).to(device=t_emb.device)
        y2 = self.label_emb(y2).to(device=t_emb.device)

        y = (1 - mix_rate) * y1 - mix_rate * y2
        y = y.to(device=t_emb.device)

        return t_emb + self.label_emb(y)
