from torch import nn, Tensor

from .activation import GeGLU


class FeedForwardNetwork(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_out: int = None,
        *,
        d_mult: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        d_out = d_model if d_out is None else d_out
        self.ff_net = nn.Sequential(
            nn.Linear(d_model, d_model * d_mult),
            GeGLU(d_model, d_model * d_mult),
            nn.Dropout(dropout),
            nn.Linear(d_model * d_mult, d_out),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.ff_net(x)
