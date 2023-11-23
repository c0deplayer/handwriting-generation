from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL
from einops import rearrange, repeat
from PIL.Image import Image
from pytorch_lightning.utilities.types import STEP_OUTPUT
from rich.progress import track
from torch import Tensor, nn
from torch.optim import Optimizer

from data.tokenizer import Tokenizer
from data.utils import get_encoded_text_with_one_hot_encoding
from models.ema import ExponentialMovingAverage

from . import utils
from .unet import UNetModel


class DiffusionWrapper(nn.Module):
    """Wrapper for a diffusion model.

    Args:
        kwargs_unet (Dict[str, Any]): Keyword arguments for UNetModel constructor.
        img_size (Tuple[int, int]): Size of the input image (height, width).

    Attributes:
        diffusion_model (UNetModel): The UNet model for diffusion.
        img_size (Tuple[int, int]): Size of the input image (height, width).

    """

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
        batch_size: int,
        writer_id: Tensor,
        word: Tensor,
        n_steps: int,
        *,
        mix_rate: float = None,
        interpolation: bool = False,
        cfg_scale: int = 0,
    ) -> Tensor:
        """
        Generate an image with diffusion noise.

        Args:
            beta_alpha (Tuple[Tensor, Tensor, Tensor]): Tuple of beta, alpha, and alpha_bar tensors.
            batch_size (int): Batch size.
            writer_id (Tensor): Writer ID tensor.
            word (Tensor): Input word tensor.
            n_steps (int): Number of diffusion steps.
            mix_rate (float, optional): Mixing rate. Default is None.
            interpolation (bool, optional): Enable interpolation. Default is False.
            cfg_scale (int, optional): Configuration scale. Default is 0.

        Returns:
            Tensor: Generated image with diffusion noise.

        """

        beta, alpha, alpha_bar = beta_alpha
        if alpha.device != word.device:
            beta = beta.to(device=word.device)
            alpha = alpha.to(device=word.device)
            alpha_bar = alpha_bar.to(device=word.device)

        with torch.inference_mode():
            x = torch.randn(
                (batch_size, 4, self.img_size[0] // 8, self.img_size[1] // 8),
                device=word.device,
            )

            p_bar = track(reversed(range(1, n_steps)))
            for i in p_bar:
                time_step = (
                    torch.ones(batch_size, dtype=torch.long, device=word.device) * i
                )
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
                        mix_rate=mix_rate,
                    )
                    predicted_noise = torch.lerp(
                        uncond_predicted_noise, predicted_noise, cfg_scale
                    )

                alpha_t = rearrange(alpha[time_step], "v -> v 1 1 1")
                alpha_bar_t = rearrange(alpha_bar[time_step], "v -> v 1 1 1")
                beta_t = rearrange(beta[time_step], "v -> v 1 1 1")
                noise = torch.randn_like(x) if i > 1 else torch.zeros_like(x)

                scaling_factor = (1 - alpha_t) / (torch.sqrt(1 - alpha_bar_t))
                x = (
                    alpha_t**-0.5 * (x - scaling_factor * predicted_noise)
                    + torch.sqrt(beta_t) * noise
                )

        return x


class LatentDiffusionModel(pl.LightningModule):
    """
    PyTorch Lightning module for a Latent Diffusion Model.

    Args:
        unet_params (Dict[str, Any]): Keyword arguments for UNetModel constructor.
        autoencoder_path (str): Path to the pre-trained autoencoder model.
        n_steps (int, optional): Number of diffusion steps. Default is 1000.
        beta_start (float, optional): Starting value for beta. Default is 1e-4.
        beta_end (float, optional): Ending value for beta. Default is 2e-2.
        img_size (Tuple[int, int], optional): Size of the input image (height, width). Default is (64, 256).

    Attributes:
        model (DiffusionWrapper): The diffusion model.
        ema (ExponentialMovingAverage): Exponential moving average of the model.
        autoencoder (AutoencoderKL): Pre-trained autoencoder model.
        n_steps (int): Number of diffusion steps.
        img_size (Tuple[int, int]): Size of the input image (height, width).
        beta (nn.Parameter): The beta parameter for diffusion.
        alpha (Tensor): The alpha parameter for diffusion.
        alpha_bar (nn.Parameter): The cumulative alpha parameter for diffusion.
    """

    def __init__(
        self,
        unet_params: Dict[str, Any],
        autoencoder_path: str,
        n_steps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        img_size: Tuple[int, int] = (64, 256),
    ) -> None:
        super().__init__()

        self.model = DiffusionWrapper(unet_params, img_size)
        self.ema = ExponentialMovingAverage(
            self.model,
            beta=0.995,
            update_after_step=2000,
            update_every=1,
            inv_gamma=1.0,
            power=1.0,
        )
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

    def on_fit_start(self) -> None:
        torch.autograd.profiler.emit_nvtx(False)
        torch.autograd.profiler.profile(False)

    def on_train_batch_end(
        self, outputs: STEP_OUTPUT, batch: Tuple[Tensor, ...], batch_idx: int
    ) -> None:
        self.ema.update()

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

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(
            "train/mae_loss",
            F.l1_loss(noise_pred, noise, reduction="mean"),
            on_step=True,
            on_epoch=True,
            prog_bar=False,
        )

        return loss

    def validation_step(
        self, batch: Tuple[Tensor, Tensor, Tensor], batch_idx: int
    ) -> Tensor:
        with torch.inference_mode():
            noise_pred, noise = self(batch)

            loss = F.mse_loss(noise_pred, noise, reduction="mean")

            self.log("val/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
            self.log(
                "val/mae_loss",
                F.l1_loss(noise_pred, noise, reduction="mean"),
                on_step=True,
                on_epoch=True,
                prog_bar=False,
            )

        return loss

    def configure_optimizers(self) -> Dict[str, Optimizer]:
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)

        return {"optimizer": optimizer}

    def generate(
        self,
        text_line: Union[str, Tensor],
        vocab: str,
        max_text_len: int,
        writer_id: Union[int, Tuple[int, ...], Tensor],
        *,
        color: str,
        save_path: Optional[Path],
        is_fid: bool = False,
        interpolation: bool = False,
        mix_rate: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> Image:
        """
        Generate an image from the model.

        Args:
            text_line (Union[str, Tensor]): Input text line or tensor.
            vocab (str): Vocabulary for text encoding.
            max_text_len (int): Maximum text length.
            writer_id (Union[int, Tuple[int, ...], Tensor]): Writer ID(s).
            color (str): Color for the generated image.
            save_path (Optional[Path]): Path to save the generated image.
            is_fid (bool, optional): Enable FID generation. Default is False.
            interpolation (bool, optional): Enable interpolation. Default is False.
            mix_rate (Optional[float]): Mixing rate. Default is None.
            seed (int, optional): Seed for random number generator to ensure reproducibility of the generated image.
                                  Default is None.

        Returns:
            Image: The generated image.
        """
        if seed is not None:
            pl.seed_everything(seed)

        if not is_fid:
            words = text_line.split(" ")
            labels = words.copy()
            words_n = []
            tokenizer = Tokenizer(vocab)

            if isinstance(writer_id, int):
                is_multiple_writer_ids = False
                writer_id = torch.as_tensor(
                    [writer_id] * len(words), device=self.device
                )
            elif isinstance(writer_id, tuple):
                is_multiple_writer_ids = True
                writer_id = torch.as_tensor(writer_id * len(words), device=self.device)
            else:
                raise TypeError(
                    f"Expected writer_id to be int or tuple, got {type(writer_id)}"
                )

            for word in words:
                _, word_enc = get_encoded_text_with_one_hot_encoding(
                    word, tokenizer=tokenizer, max_len=max_text_len
                )
                word_tensor = torch.tensor(
                    word_enc, dtype=torch.long, device=self.device
                )
                word_tensor = rearrange(word_tensor, "v -> 1 v")

                words_n.append(word_tensor)

            words_t = torch.cat(words_n, dim=0)
            if is_multiple_writer_ids:
                words_t = repeat(
                    words_t, "b v -> (b repeat) v", repeat=writer_id.size(0)
                )
        else:
            labels = None
            words_t = text_line.clone()

        x = self.ema.model.generate_image_noise(
            beta_alpha=(self.beta, self.alpha, self.alpha_bar),
            batch_size=words_t.size(0),
            writer_id=writer_id,
            word=words_t,
            n_steps=self.n_steps,
            interpolation=interpolation,
            mix_rate=mix_rate,
        )

        x /= 0.18215
        image = self.autoencoder.decode(x.float()).sample

        image = torch.clamp((image / 2 + 0.5), min=0, max=1).cpu()

        return utils.generate_image(
            image, save_path, color=color, labels=labels, is_fid=is_fid
        )
