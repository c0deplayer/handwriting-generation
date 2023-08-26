import torch.nn as nn
from torch import Tensor

from .residual import ResBlock
from .transformers import SpatialTransformer


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
