from typing import Any

import lightning.pytorch as pl
import torch
from diffusers import AutoencoderKL
from torch import nn, Tensor

from . import utils
from .activation import GeGLU
from .unet import UNetModel


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

        if d_out is None:
            d_out = d_model

        self.ff_net = nn.Sequential(
            nn.Linear(d_model, d_model * d_mult),
            GeGLU(d_model, d_model * d_mult),
            nn.Dropout(dropout),
            nn.Linear(d_model * d_mult, d_out),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.ff_net(x)


class DiffusionWrapper(nn.Module):
    def __init__(self, kwargs_unet: dict[str, Any]) -> None:
        super().__init__()

        self.diffusion_model = UNetModel(**kwargs_unet)

    def forward(
        self,
        x: Tensor,
        time_step: Tensor,
        *,
        context: Tensor = None,
        writer_id: Tensor | tuple[int, int] = None,
        interpolation: bool = False,
    ) -> Tensor:
        return self.diffusion_model(
            x,
            time_step,
            context=context,
            writer_id=writer_id,
            interpolation=interpolation,
        )


class LatentDiffusion(pl.LightningModule):
    def __init__(
        self,
        unet_params: dict[str, Any],
        autoencoder_path: str,
        n_steps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        img_size: tuple[int, int] = (64, 128),
    ) -> None:
        super().__init__()

        self.model = DiffusionWrapper(unet_params)
        self.autoencoder = AutoencoderKL.from_pretrained(
            autoencoder_path, subfolder="vae"
        ).requires_grad_(False)
        self.n_steps = n_steps
        self.img_size = img_size

        beta = torch.linspace(beta_start, beta_end, n_steps, dtype=torch.float64)
        self.beta = nn.Parameter(beta.to(torch.float32), requires_grad=False)

        self.alpha = 1.0 - beta
        alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.alpha_bar = nn.Parameter(alpha_bar.to(torch.float32), requires_grad=False)

    def forward(
        self,
        batch: tuple[Tensor, ...],
        *,
        writer_id: Tensor | tuple[int, int] = None,
        interpolation: bool = False,
    ):
        writers, images, text = batch

        images = (
            self.autoencoder.encode(images.to(torch.float32)).latent_dist.sample()
            * 0.18215
        )

        time_step = torch.randint(low=1, high=self.n_steps, size=(images.size(0),))
        x_t, noise = utils.noise_image(images, time_step, self.alpha_bar)

        return self.model(
            x_t,
            time_step,
            context=text,
            writer_id=writer_id,
            interpolation=interpolation,
        )
