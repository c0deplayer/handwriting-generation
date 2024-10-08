from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from matplotlib import pyplot as plt
from pytorch_lightning.utilities.types import STEP_OUTPUT
from rich.progress import track
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torchvision import transforms

from data.tokenizer import Tokenizer
from data.utils import get_encoded_text_with_one_hot_encoding, get_image

from ..ema import ExponentialMovingAverage
from . import utils
from .attention import AttentionBlock
from .cnn import ConvBlock
from .inv_sqrt_scheduler import InverseSqrtScheduler
from .text_style import StyleExtractor, TextStyleEncoder
from .utils import FeedForwardNetwork


class DiffusionModel(nn.Module):
    """
    This Diffusion Model is a neural network architecture used for generating
    realistic handwriting sequences. It incorporates multiple components, such as
    convolutional layers, attention blocks, and feedforward networks, to capture
    the complexity of handwriting generation.

    Args:
        ab (Tuple[Tensor, Tensor], optional): A tuple of alpha (1 - beta) and beta tensors.
        num_layers (int, optional): The number of AttentionBlock layers at the lowest resolution. (default: 2)
        c1 (int, optional): Dimensionality of the first convolutional layer. (default: 128)
        c2 (int, optional): Dimensionality of the second convolutional layer. (default: 192)
        c3 (int, optional): Dimensionality of the third convolutional layer. (default: 256)
        drop_rate (float, optional): Dropout rate. (default: 0.1)
        vocab_size (int, optional): Vocabulary size for text embeddings. (default: 73)

    Attributes:
        alpha (Tensor): Alpha values for diffusion.
        beta (Tensor): Beta values for diffusion.
    """

    def __init__(
        self,
        ab: Tuple[Tensor, Tensor] = None,
        num_layers: int = 2,
        c1: int = 128,
        c2: int = 192,
        c3: int = 256,
        drop_rate: float = 0.1,
        vocab_size: int = 73,
    ) -> None:
        super().__init__()

        if ab is not None:
            self.alpha, self.beta = ab
        else:
            self.beta = utils.get_beta_set(device=self.device)
            self.alpha = torch.cumprod(1 - self.beta, dim=0)

        self.sigma_ff_network = FeedForwardNetwork(1, c1 // 4, hidden_size=2048)
        self.text_style_encoder = TextStyleEncoder(c1 * 2, c2 * 2, vocab_size, c2 * 4)

        self.input_dense = nn.Linear(2, c1)
        self.enc_0 = ConvBlock(c1, c1, c1)
        self.avg_pool = nn.AvgPool1d(2)

        self.enc_1 = ConvBlock(c1, c1, c2)
        self.enc_2 = AttentionBlock(
            c2 * 2,
            c2,
            num_heads=3,
            drop_rate=drop_rate,
            pos_factor=4,
        )

        self.enc_3 = ConvBlock(c1, c2, c3)
        self.enc_4 = AttentionBlock(
            c2 * 2,
            c3,
            num_heads=4,
            drop_rate=drop_rate,
            pos_factor=2,
        )

        self.attention_dense = nn.Linear(c3, c2 * 2)
        self.attention_layers = nn.ModuleList(
            [
                AttentionBlock(
                    c2 * 2,
                    c2 * 2,
                    6,
                    drop_rate=drop_rate,
                    swap_channel_layer=False,
                )
                for _ in range(num_layers)
            ]
        )

        self.up_sample = nn.Upsample(scale_factor=2, mode="nearest")
        self.skip_conv_2 = nn.Conv1d(c3, c2 * 2, 3, padding="same")
        self.dec_2 = ConvBlock(c1, c2 * 2, c3)

        self.skip_conv_1 = nn.Conv1d(c2, c3, 3, padding="same")
        self.dec_1 = ConvBlock(c1, c3, c2)

        self.skip_conv_0 = nn.Conv1d(c1, c2, 3, padding="same")
        self.dec_0 = ConvBlock(c1, c2, c1)

        self.output_dense = nn.Linear(c1, 2)
        self.pen_lifts_dense = nn.Sequential(nn.Linear(c1, 1), nn.Sigmoid())

    def forward(self, batch: Tuple[Tensor, ...]) -> Tuple[Tensor, Tensor, Tensor]:
        x, text, sigma, style = batch
        if self.beta.device != x.device:
            self.beta = self.beta.to(device=x.device)
            self.alpha = self.alpha.to(device=x.device)

        sigma = self.sigma_ff_network(sigma)
        text_mask = utils.create_padding_mask(text)
        text = self.text_style_encoder(text, style, sigma)

        x = self.input_dense(x)
        x = rearrange(x, "b h w -> b w h")
        h_1 = self.enc_0(x, sigma)
        h_2 = self.avg_pool(h_1)

        h_2 = self.enc_1(h_2, sigma)
        h_2, _ = self.enc_2(
            h_2,
            text,
            sigma,
            mask=text_mask,
        )
        h_2 = rearrange(h_2, "b h w -> b w h")
        h_3 = self.avg_pool(h_2)

        h_3 = self.enc_3(h_3, sigma)
        h_3, _ = self.enc_4(
            h_3,
            text,
            sigma,
            mask=text_mask,
        )
        h_3 = rearrange(h_3, "b h w -> b w h")
        x = self.avg_pool(h_3)

        x = rearrange(x, "b h w -> b w h")
        x = self.attention_dense(x)
        for attention_layer in self.attention_layers:
            x, attention = attention_layer(
                x,
                text,
                sigma,
                mask=text_mask,
            )

        x = rearrange(x, "b h w -> b w h")
        x = self.up_sample(x) + self.skip_conv_2(h_3)
        x = self.dec_2(x, sigma)

        x = self.up_sample(x) + self.skip_conv_1(h_2)
        x = self.dec_1(x, sigma)

        x = self.up_sample(x) + self.skip_conv_0(h_1)
        x = self.dec_0(x, sigma)

        x = rearrange(x, "b h w -> b w h")
        output = self.output_dense(x)
        pen_lifts = self.pen_lifts_dense(x)

        return output, pen_lifts, attention


class DiffusionWrapper(pl.LightningModule):
    """
    This class serves as a PyTorch Lightning Module and EMA wrapper for training and using
    the Diffusion Model for generating handwriting sequences.

    Args:
        diffusion_params (Dict[str, Any]): Dictionary of parameters for initializing the
            DiffusionModel.
        use_ema (bool, optional): Use Exponential Moving Average during training.
            (default: True)

    Attributes:
        beta (Tensor): Beta values for diffusion.
        alpha (Tensor): Alpha values for diffusion.
        use_ema (bool): Flag to indicate if Exponential Moving Average is used.
        diffusion_model (DiffusionModel): The core Diffusion Model.
        ema (Optional[ExponentialMovingAverage]): Exponential Moving Average module.

    Note:
        For proper initialization of `diffusion_params`, ensure it contains the necessary
        parameters for configuring the DiffusionModel.
    """

    def __init__(
        self, diffusion_params: Dict[str, Any], *, use_ema: bool = True
    ) -> None:
        super().__init__()

        self.beta = utils.get_beta_set(device=self.device)
        self.alpha = torch.cumprod(1 - self.beta, dim=0)
        self.use_ema = use_ema

        self.diffusion_model = DiffusionModel(
            (self.alpha, self.beta), **diffusion_params
        )

        if use_ema:
            self.ema = ExponentialMovingAverage(
                self.diffusion_model,
                beta=0.995,
                update_after_step=2000,
                update_every=1,
                inv_gamma=1.0,
                power=1.0,
            )
        else:
            self.ema = None

        self.save_hyperparameters()

    def on_train_batch_end(
        self, outputs: STEP_OUTPUT, batch: Tuple[Tensor, ...], batch_idx: int
    ) -> None:
        if self.use_ema:
            self.ema.update()

    def on_fit_start(self) -> None:
        torch.autograd.profiler.emit_nvtx(False)
        torch.autograd.profiler.profile(False)

    def forward(self, batch: Tuple[Tensor, ...]) -> Tuple[Tensor, Tensor, Tensor]:
        return self.diffusion_model(batch)

    def training_step(self, batch: Tuple[Tensor, ...], batch_idx: int) -> Tensor:
        strokes, text, style, _ = batch
        strokes, pen_lifts = strokes[:, :, :2], strokes[:, :, 2]

        alphas = utils.get_alphas(strokes.size(0), self.alpha, device=strokes.device)
        eps = torch.randn_like(strokes)

        strokes_perturbed = torch.sqrt(alphas) * strokes + torch.sqrt(1 - alphas) * eps

        model_batch = (strokes_perturbed, text, torch.sqrt(alphas), style)
        strokes_pred, pen_lifts_pred, _ = self(model_batch)

        loss_batch = (eps, strokes_pred, pen_lifts, pen_lifts_pred, alphas)
        loss, mse_loss, bce_loss = self.loss(loss_batch)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(
            "train/mse_strokes",
            mse_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
        )
        self.log("train/bce_eos", bce_loss, on_step=True, on_epoch=True, prog_bar=False)

        return loss

    def validation_step(self, batch: Tuple[Tensor, ...], batch_idx: int) -> Tensor:
        strokes, text, style, _ = batch
        strokes, pen_lifts = strokes[:, :, :2], strokes[:, :, 2]

        alphas = utils.get_alphas(strokes.size(0), self.alpha, device=strokes.device)
        eps = torch.randn_like(strokes)

        strokes_perturbed = torch.sqrt(alphas) * strokes + torch.sqrt(1 - alphas) * eps
        model_batch = (strokes_perturbed, text, torch.sqrt(alphas), style)

        with torch.inference_mode():
            strokes_pred, pen_lifts_pred, _ = self(model_batch)

            loss_batch = (eps, strokes_pred, pen_lifts, pen_lifts_pred, alphas)
            loss, mse_loss, bce_loss = self.loss(loss_batch)

            self.log("val/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
            self.log(
                "val/mse_strokes",
                mse_loss,
                on_step=True,
                on_epoch=True,
                prog_bar=False,
            )
            self.log(
                "val/bce_eos", bce_loss, on_step=True, on_epoch=True, prog_bar=False
            )

            self.log(
                "val/mae_strokes",
                F.l1_loss(strokes_pred, strokes, reduction="mean"),
                on_step=True,
                on_epoch=True,
                prog_bar=False,
            )
            self.log(
                "val/mae_eos",
                F.l1_loss(
                    rearrange(pen_lifts_pred, "b h 1 -> b h"),
                    pen_lifts,
                    reduction="mean",
                ),
                on_step=True,
                on_epoch=True,
                prog_bar=False,
            )

        return loss

    def configure_optimizers(
        self,
    ) -> Dict[str, Union[Optimizer, Dict[str, Union[LRScheduler, str, int]]]]:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=1e-3,
            betas=(0.9, 0.98),
        )
        scheduler = InverseSqrtScheduler(
            optimizer=optimizer,
            lr_mul=1.0,
            d_model=256,
            n_warmup_steps=10000,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    @staticmethod
    def loss(loss_batch: Tuple[Tensor, ...]) -> Tuple[Tensor, Tensor, Tensor]:
        strokes, strokes_pred, pen_lifts, pen_lifts_pred, alphas = loss_batch

        pen_lifts_pred = rearrange(pen_lifts_pred, "b h 1 -> b h")

        strokes_loss = torch.mean(
            torch.sum(F.mse_loss(strokes_pred, strokes, reduction="none"), dim=-1)
        )

        pen_lifts_loss = torch.mean(
            F.binary_cross_entropy(pen_lifts_pred, pen_lifts, reduction="mean")
            * torch.squeeze(alphas, dim=-1)
        )

        return strokes_loss + pen_lifts_loss, strokes_loss, pen_lifts_loss

    def generate(
        self,
        sequence: Union[str, Tensor],
        vocab: str,
        max_text_len: int,
        *,
        color: str,
        save_path: Optional[str],
        style_path: Union[str, Tensor],
        is_fid: bool = False,
        seed: Optional[int] = None,
    ) -> Union[plt.Figure, List[plt.Figure]]:
        """
        Generate handwriting sequence images using the trained Diffusion Model.

        Args:
            sequence (Union[str, Tensor]): Either the input text sequence or the encoded
                text tensor.
            vocab (str): Vocabulary used for text generation.
            max_text_len (int): Maximum text length for generation.
            color (str): Color of the generated strokes.
            save_path (Union[str, None]): Path to save the generated image (or None for no
                saving).
            style_path (Union[str, Tensor]): Path to the style image or the style vector.
            is_fid (bool, optional): Use FID (Fréchet Inception Distance) generation mode.
                (default: False)
            seed (int, optional): Seed for random number generator to ensure reproducibility of the generated image.
                                 (default: None)

        Returns:
            Union[plt.Figure, List[plt.Figure]]: The generated handwriting sequence as a Matplotlib figure.
        """
        if seed is not None:
            pl.seed_everything(seed)

        if not is_fid:
            style_path = Path(f"assets/{style_path}")
            style_extractor = StyleExtractor(device=self.device)

            writer_style = get_image(style_path, 1400, 96)
            writer_style = rearrange(
                transforms.PILToTensor()(writer_style).to(torch.float32),
                "1 h w -> 1 1 h w",
            )

            style_vector = style_extractor(writer_style)
            style_vector = rearrange(style_vector, "h w -> 1 h w")
            style_vector = style_vector.to(device=self.device)

            _, text = get_encoded_text_with_one_hot_encoding(
                sequence, tokenizer=Tokenizer(vocab), max_len=max_text_len
            )
            text = rearrange(torch.tensor(text, device=self.device), "v -> 1 v")
        else:
            text = sequence.clone()
            style_vector = style_path

        time_steps = (text.size(1) - 2) * 16
        time_steps = time_steps - (time_steps % 8) + 8
        batch_size = text.size(0)
        beta_len = len(self.beta)
        strokes = torch.randn((batch_size, time_steps, 2), device=self.device)

        with torch.inference_mode():
            for i in track(range(beta_len - 1, -1, -1)):
                alpha = self.alpha[i] * torch.ones(
                    (batch_size, 1, 1), device=self.device
                )
                beta = self.beta[i] * torch.ones((batch_size, 1, 1), device=self.device)
                alpha_next = (
                    self.alpha[i - 1]
                    if i > 1
                    else torch.tensor([1.0], device=self.device)
                )
                batch = (strokes, text, torch.sqrt(alpha), style_vector)

                with torch.inference_mode():
                    model_out, pen_lifts, _ = (
                        self.ema(batch) if self.use_ema else self(batch)
                    )
                    strokes = utils.diffusion_step(
                        strokes, model_out, beta, alpha, alpha_next
                    )

        strokes = torch.cat([strokes, pen_lifts], dim=-1)

        return utils.generate_stroke_image(
            strokes.detach().cpu().numpy(), scale=1.0, save_path=save_path, color=color
        )
