from pathlib import Path
from typing import Any, Dict, Tuple, Union

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL
from einops import rearrange
from rich.progress import track
from torch import Tensor, nn
from torch.optim import Optimizer

from data.tokenizer import Tokenizer
from . import utils
from .unet import UNetModel


class DiffusionWrapper(nn.Module):
    def __init__(self, kwargs_unet: Dict[str, Any], img_size: Tuple[int, int]) -> None:
        super().__init__()

        self.diffusion_model = UNetModel(**kwargs_unet)
        self.img_size = img_size

    def forward(
        self,
        x: Tensor,
        time_step: Tensor,
        *,
        context: Tensor = None,
        writer_id: Union[Tensor, Tuple[int, int]] = None,
        interpolation: bool = False,
        mix_rate: float = None,
    ) -> Tensor:
        return self.diffusion_model(
            x,
            time_step,
            context=context,
            writer_id=writer_id,
            interpolation=interpolation,
            mix_rate=mix_rate,
        )

    def generate_image_noise(
        self,
        beta_alpha: Tuple[Tensor, Tensor, Tensor],
        n: int,
        writer_id: Union[Tensor, Tuple[int, int]],
        word: Tensor,
        n_steps: int,
        *,
        mix_rate: float = None,
        interpolation: bool = False,
        cfg_scale: int = 3,
    ) -> Tensor:
        beta, alpha, alpha_bar = beta_alpha
        with torch.no_grad():
            x = torch.randn(
                (n, 4, self.img_size[0] // 8, self.img_size[1] // 8),
                device=word.device,
            )

            p_bar = track(reversed(range(1, n_steps)))
            for i in p_bar:
                time_step = torch.ones(n, dtype=torch.long, device=word.device) * i
                predicted_noise = self(
                    x,
                    time_step,
                    context=word,
                    writer_id=writer_id,
                    interpolation=interpolation,
                    mix_rate=mix_rate,
                )
                if cfg_scale > 0:
                    uncond_predicted_noise = self(
                        x,
                        time_step,
                        context=word,
                        writer_id=writer_id,
                        interpolation=interpolation,
                        mix_rate=mix_rate,
                    )
                    predicted_noise = torch.lerp(
                        uncond_predicted_noise, predicted_noise, cfg_scale
                    )

                alpha = rearrange(alpha[time_step], "v -> v 1 1 1")
                alpha_bar = rearrange(alpha_bar[time_step], "v -> v 1 1 1")
                beta = rearrange(beta[time_step], "v -> v 1 1 1")
                noise = torch.randn_like(x) if i > 1 else torch.zeros_like(x)

                scaling_factor = (1 - alpha) / (torch.sqrt(1 - alpha_bar))
                x = (
                    alpha**-0.5 * (x - scaling_factor * predicted_noise)
                    + torch.sqrt(beta) * noise
                )

        return x


class LatentDiffusionModel(pl.LightningModule):
    def __init__(
        self,
        unet_params: Dict[str, Any],
        autoencoder_path: str,
        n_steps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        img_size: Tuple[int, int] = (64, 128),
    ) -> None:
        super().__init__()

        self.model = DiffusionWrapper(unet_params, img_size)
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

        self.save_hyperparameters()

    def forward(
        self,
        batch: Tuple[Tensor, ...],
        *,
        interpolation: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        writers, images, text = batch

        images = self.autoencoder.encode(images.to(torch.float32)).latent_dist.sample()
        images *= 0.18215

        time_step = torch.randint(
            low=1, high=self.n_steps, size=(images.size(0),), device=images.device
        )
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

    def training_step(
        self, batch: Tuple[Tensor, Tensor, Tensor], batch_idx: int
    ) -> Tensor:
        noise_pred, noise = self(batch)

        loss = F.mse_loss(noise_pred, noise, reduction="mean")
        self.log("loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(
        self, batch: Tuple[Tensor, Tensor, Tensor], batch_idx: int
    ) -> Tensor:
        with torch.no_grad():
            noise_pred, noise = self(batch)

        loss = F.mse_loss(noise_pred, noise, reduction="mean")
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self) -> Dict[str, Optimizer]:
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)

        return {"optimizer": optimizer}

    def generate(
        self,
        text_line: str,
        vocab: str,
        writer_id: Union[int, Tuple[int, int]],
        *,
        save_path: Path,
        interpolation: bool = False,
        mix_rate: float = None,
    ) -> None:
        words = text_line.split(" ")
        tokenizer = Tokenizer(vocab)
        if isinstance(writer_id, int):
            writer_id = torch.tensor(writer_id, dtype=torch.int32)
        elif not isinstance(writer_id, tuple):
            raise TypeError(
                f"Expected writer_id to be int or tuple, got {type(writer_id)}"
            )

        # TODO: Combine images into one image to create a line of text
        for word in words:
            word_enc = tokenizer.encode(word)
            word_tensor = torch.tensor(word_enc, dtype=torch.long)

            x = self.model.generate_image_noise(
                beta_alpha=(self.beta, self.alpha, self.alpha_bar),
                n=len(writer_id),
                writer_id=writer_id,
                word=word_tensor,
                n_steps=self.n_steps,
                interpolation=interpolation,
                mix_rate=mix_rate,
            )

            x /= 0.18215
            image = self.autoencoder.decode(x).sample()

            image = (image / 2 + 0.5).clamp(0, 1).cpu()
            utils.save_image(image, save_path)
