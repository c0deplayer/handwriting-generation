import torch.nn as nn
from torch import Tensor

from .encoder import CharacterEncoder
from .residual import ResBlock, DownSample, UpSample
from .transformers import SpatialTransformer
from .utils import GroupNorm32


# noinspection PyMethodOverriding
class TimestepEmbedSequential(nn.Sequential):
    def forward(self, x: Tensor, t_emb: Tensor, *, context: Tensor = None) -> Tensor:
        for layer in self:
            if isinstance(layer, ResBlock):
                x = layer(x, t_emb)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, context=context)
            else:
                x = layer(x)

        return x


class UNetModel(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        channels: int,
        res_layers: int,
        vocab_size: int,
        attention_levels: tuple[int],
        channel_multipliers: tuple[int],
        d_cond: int,  # 768
        heads: int,
        d_head: int,
        dropout: float = 0.0,
        n_style_classes: int = None,
        tf_layers: int = 1,
        max_seq_len: int = 20,
    ):
        super().__init__()
        if d_cond is None:
            raise RuntimeError(
                "Expected dimension of cross_attention conditioning to be int, got None"
            )

        self.channels = channels
        self.max_seq_len = max_seq_len
        self.dropout = dropout

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

        self.input_blocks = nn.ModuleList()
        self.input_blocks.append(
            TimestepEmbedSequential(
                nn.Conv2d(in_channels, channels, kernel_size=3, padding=1)
            )
        )

        input_block_channels = [channels]
        channels_list = [channels * m for m in channel_multipliers]

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

        self.output_block = nn.ModuleList()

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

                self.output_block.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            GroupNorm32(32, channels),
            nn.SiLU(),
            nn.Conv2d(channels, out_channels, kernel_size=3, padding=1),
        )
