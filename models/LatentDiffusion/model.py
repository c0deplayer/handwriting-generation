from typing import Any

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL
from torch import Tensor, nn

from . import utils
from .unet import UNetModel


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


class LatentDiffusionModel(pl.LightningModule):
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
        interpolation: bool = False,
    ) -> tuple[Tensor, Tensor]:
        writers, images, text = batch

        images = (
            self.autoencoder.encode(images.to(torch.float32)).latent_dist.sample()
            * 0.18215
        )

        time_step = torch.randint(low=1, high=self.n_steps, size=(images.size(0),))
        x_t, noise = utils.noise_image(images, time_step, self.alpha_bar)

        return (
            self.model(
                x_t,
                time_step,
                context=text,
                writer_id=writers,
                interpolation=interpolation,
            ),
            noise,
        )

    def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        noise_pred, noise = self(batch)

        return F.mse_loss(noise_pred, noise, reduction="mean")

    def validation_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        with torch.no_grad():
            noise_pred, noise = self(batch)

        return F.mse_loss(noise_pred, noise, reduction="mean")
