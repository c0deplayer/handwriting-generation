import os
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from matplotlib import pyplot as plt
from rich.progress import track
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from data.tokenizer import Tokenizer
from data.utils import get_image, pad_image
# noinspection PyPackages
from . import utils
# noinspection PyPackages
from .attention import AffineTransformLayer, MultiHeadAttention
# noinspection PyPackages
from .cnn import ConvBlock, FeedForwardNetwork
# noinspection PyPackages
from .encoder import PositionalEncoder
# noinspection PyPackages
from .inv_sqrt_scheduler import InverseSqrtScheduler
# noinspection PyPackages
from .text_style import StyleExtractor, TextStyleEncoder


class AttentionalBlock(nn.Module):
    def __init__(
        self,
        in_features: int,
        d_model: int,
        num_heads: int,
        *,
        device: torch.device,
        drop_rate: float = 0.1,
        pos_factor: int = 1,
        swap_channel_layer: bool = True,
    ) -> None:
        """
        _summary_

        Parameters
        ----------
        in_features: int
            _description_
        d_model : int
            _description_
        num_heads : int
            _description_
        device: torch.device
            _description_
        drop_rate : float, optional
            _description_, by default 0.1
        pos_factor : int, optional
            _description_, by default 1
        swap_channel_layer : bool, optional
            _description_, by default True
        """

        super().__init__()
        self.__device = device

        self.swap_channel_layer = swap_channel_layer
        self.text_pos = PositionalEncoder(2000, d_model)().to(device)
        self.stroke_pos = PositionalEncoder(2000, d_model, pos_factor=pos_factor)().to(
            device
        )

        self.dense_layer = nn.Linear(in_features=in_features, out_features=d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6, elementwise_affine=False)
        self.affine_0 = AffineTransformLayer(in_features // 12, d_model)

        self.mha_0 = MultiHeadAttention(d_model, num_heads)
        self.affine_1 = AffineTransformLayer(in_features // 12, d_model)

        self.mha_1 = MultiHeadAttention(d_model, num_heads)
        self.affine_2 = AffineTransformLayer(in_features // 12, d_model)

        self.ff_network = FeedForwardNetwork(d_model, d_model, hidden_size=d_model * 2)
        self.dropout = nn.Dropout(p=drop_rate)
        self.affine_3 = AffineTransformLayer(in_features // 12, d_model)

    def forward(
        self, batch: tuple[Tensor, Tensor, Tensor, Tensor]
    ) -> tuple[Tensor, Tensor]:
        """
        _summary_

        Parameters
        ----------
        batch : tuple[Tensor, Tensor, Tensor, Tensor]
            _description_

        Returns
        -------
        tuple[Tensor, Tensor]
            _description_
        """

        x, text, sigma, text_mask = batch

        text = self.dense_layer(F.silu(text))
        text = self.affine_0((self.layer_norm(text), sigma))
        text_pos = text + self.text_pos[:, : text.size(1)]

        # text_mask = rearrange(text_mask, "b h w c -> b (h w c)")
        if self.swap_channel_layer:
            x = rearrange(x, "b h w -> b w h")

        x_pos = x + self.stroke_pos[:, : x.size(1)]
        x_2, attention = self.mha_0(x_pos, text_pos, text, mask=text_mask)
        x_2 = self.layer_norm(self.dropout(x_2))
        x_2 = self.affine_1((x_2, sigma)) + x

        x_2_pos = x_2 + self.stroke_pos[:, : x.size(1)]
        x_3, _ = self.mha_1(x_2_pos, x_2_pos, x_2)
        x_3 = self.layer_norm(x_2 + self.dropout(x_3))
        x_3 = self.affine_2((x_3, sigma))

        x_4 = self.ff_network(x_3)
        x_4 = self.dropout(x_4) + x_3
        output = self.affine_3((self.layer_norm(x_4), sigma))

        return output, attention


class DiffusionModel(pl.LightningModule):
    def __init__(
        self,
        device: str,
        num_layers: int = 2,
        c1: int = 128,
        c2: int = 192,
        c3: int = 256,
        drop_rate: float = 0.1,
        vocab_size: int = 73,
    ) -> None:
        """
        _summary_

        Parameters
        ----------
        device : str
            _description_
        num_layers : int, optional
            _description_, by default 2
        c1 : int, optional
            _description_, by default 128
        c2 : int, optional
            _description_, by default 192
        c3 : int, optional
            _description_, by default 256
        drop_rate : float, optional
            _description_, by default 0.1
        vocab_size : int, optional
            _description_, by default 73
        """

        super().__init__()

        self.__device = torch.device(device)
        self.beta = utils.get_beta_set()
        self.alpha = torch.cumprod(1 - self.beta, dim=0)

        self.sigma_ff_network = FeedForwardNetwork(1, c1 // 4, hidden_size=2048)
        self.text_style_encoder = TextStyleEncoder(c1 * 2, c2 * 2, vocab_size, c2 * 4)

        self.input_dense = nn.LazyLinear(c1)
        self.enc_0 = ConvBlock(c1, c1, c1)
        self.avg_pool = nn.AvgPool1d(2)

        self.enc_1 = ConvBlock(c1, c1, c2)
        self.enc_2 = AttentionalBlock(
            c2 * 2,
            c2,
            num_heads=3,
            drop_rate=drop_rate,
            pos_factor=4,
            device=self.__device,
        )

        self.enc_3 = ConvBlock(c1, c2, c3)
        self.enc_4 = AttentionalBlock(
            c2 * 2,
            c3,
            num_heads=4,
            drop_rate=drop_rate,
            pos_factor=2,
            device=self.__device,
        )

        self.attention_dense = nn.Linear(c3, c2 * 2)
        self.attention_layers = nn.ModuleList(
            [
                AttentionalBlock(
                    c2 * 2,
                    c2 * 2,
                    6,
                    drop_rate=drop_rate,
                    swap_channel_layer=False,
                    device=self.__device,
                )
                for _ in range(num_layers)
            ]
        )

        self.up_sample = nn.Upsample(scale_factor=2)
        self.skip_conv_2 = nn.Conv1d(c3, c2 * 2, 3, padding="same")
        self.dec_2 = ConvBlock(c1, c2 * 2, c3)

        self.skip_conv_1 = nn.Conv1d(c2, c3, 3, padding="same")
        self.dec_1 = ConvBlock(c1, c3, c2)

        self.skip_conv_0 = nn.Conv1d(c1, c2, 3, padding="same")
        self.dec_0 = ConvBlock(c1, c2, c1)

        self.output_dense = nn.Linear(c1, 2)
        self.pen_lifts_dense = nn.Linear(c1, 1)

        self.save_hyperparameters()

    def forward(
        self, batch: tuple[Tensor, Tensor, Tensor, Tensor]
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        _summary_

        Parameters
        ----------
        batch : tuple[Tensor, Tensor, Tensor, Tensor]
            _description_

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            _description_
        """

        strokes, text, sigma, style = batch

        sigma = self.sigma_ff_network(sigma)
        text_mask = utils.create_padding_mask(text)
        text = self.text_style_encoder((text, style, sigma))

        x = self.input_dense(strokes)
        x = rearrange(x, "b h w -> b w h")
        h_1 = self.enc_0((x, sigma))
        h_2 = self.avg_pool(h_1)

        h_2 = self.enc_1((h_2, sigma))
        h_2, _ = self.enc_2(
            (h_2, text, sigma, text_mask),
        )
        h_2 = rearrange(h_2, "b h w -> b w h")
        h_3 = self.avg_pool(h_2)

        h_3 = self.enc_3((h_3, sigma))
        h_3, _ = self.enc_4(
            (h_3, text, sigma, text_mask),
        )
        h_3 = rearrange(h_3, "b h w -> b w h")
        x = self.avg_pool(h_3)

        x = rearrange(x, "b h w -> b w h")
        x = self.attention_dense(x)
        for attention_layer in self.attention_layers:
            x, attention = attention_layer(
                (
                    x,
                    text,
                    sigma,
                    text_mask,
                ),
            )

        x = rearrange(x, "b h w -> b w h")
        x = self.up_sample(x) + self.skip_conv_2(h_3)
        x = self.dec_2((x, sigma))

        x = self.up_sample(x) + self.skip_conv_1(h_2)
        x = self.dec_1((x, sigma))

        x = self.up_sample(x) + self.skip_conv_0(h_1)
        x = self.dec_0((x, sigma))

        x = rearrange(x, "b h w -> b w h")
        output = self.output_dense(x)
        pen_lifts = self.pen_lifts_dense(x)

        # noinspection PyUnboundLocalVariable
        return output, pen_lifts, attention

    def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        strokes, text, style = batch
        strokes, pen_lifts = strokes[:, :, :2], strokes[:, :, 2]

        alphas = utils.get_alphas(strokes.size(0), self.alpha)
        alphas = alphas.to(self.__device)
        eps = torch.randn_like(strokes)

        strokes_perturbed = torch.sqrt(alphas) * strokes

        strokes_perturbed += torch.sqrt(1 - alphas) * eps

        model_batch = (strokes_perturbed, text, torch.sqrt(alphas), style)
        strokes_pred, pen_lifts_pred, _ = self(model_batch)

        loss_batch = (eps, strokes_pred, pen_lifts, pen_lifts_pred, alphas)
        loss = self.loss(loss_batch)

        self.log("loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        strokes, text, style = batch
        strokes, pen_lifts = strokes[:, :, :2], strokes[:, :, 2]

        alphas = utils.get_alphas(strokes.size(0), self.alpha)
        alphas = alphas.to(self.__device)
        eps = torch.randn_like(strokes)

        strokes_perturbed = torch.sqrt(alphas) * strokes

        strokes_perturbed += torch.sqrt(1 - alphas) * eps
        model_batch = (strokes_perturbed, text, torch.sqrt(alphas), style)

        with torch.no_grad():
            strokes_pred, pen_lifts_pred, _ = self(model_batch)

        loss_batch = (eps, strokes_pred, pen_lifts, pen_lifts_pred, alphas)
        loss = self.loss(loss_batch)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self) -> dict[str, Optimizer | LRScheduler]:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=1e-3,
            betas=(0.9, 0.98),
            weight_decay=1e-4,
        )
        lr_scheduler = InverseSqrtScheduler(
            optimizer=optimizer,
            lr_mul=1.0,
            d_model=256,
            n_warmup_steps=10000,
            verbose=True,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
        }

    @staticmethod
    def loss(loss_batch: tuple[Tensor, ...]) -> Tensor:
        """
        _summary_

        Parameters
        ----------
        loss_batch : tuple[Tensor, ...]
            _description_

        Returns
        -------
        Tensor
            _description_
        """

        strokes, strokes_pred, pen_lifts, pen_lifts_pred, alphas = loss_batch

        pen_lifts = torch.clamp_(pen_lifts, min=1e-7, max=1 - 1e-7)
        pen_lifts_pred = rearrange(F.sigmoid(pen_lifts_pred), "b h 1 -> b h")
        strokes_loss = torch.mean(
            torch.sum(torch.square(strokes - strokes_pred), dim=-1)
        )
        pen_lifts_loss = torch.mean(
            F.binary_cross_entropy_with_logits(pen_lifts_pred, pen_lifts)
            * torch.squeeze(alphas, dim=-1)
        )

        return strokes_loss + pen_lifts_loss

    def generate(
        self,
        sequence: str,
        style_path: Path | None = None,
        diffusion_mode: str = "new",
        show_every: bool = False,
    ) -> None:
        """
        _summary_

        Parameters
        ----------
        sequence : str
            _description_
        style_path : Path | None, optional
            _description_, by default None
        diffusion_mode : str, optional
            _description_, by default "new"
        show_every : bool, optional
            _description_, by default False

        Returns
        -------
        plt.Figure
            _description_
        """

        if style_path is None:
            asset_dir = os.listdir(
                "/home/codeplayer/Studia/Inżynierka/Handwriting-Synthesis-Thesis-fr-fr/assets"
            )
            style_path = Path(
                "/home/codeplayer/Studia/Inżynierka/Handwriting-Synthesis-Thesis-fr-fr/assets/"
                f"{asset_dir[np.random.randint(0, len(asset_dir))]}"
            )

        tokenizer = Tokenizer()
        style_extractor = StyleExtractor(device=self.__device)
        beta_set = utils.get_beta_set()
        alpha_set = torch.cumprod(1 - beta_set, dim=0)

        time_steps = len(sequence) * 16
        time_steps -= (time_steps % 8) + 8

        writer_style = get_image(style_path, height=96)
        writer_style = pad_image(writer_style, width=1400, height=96)
        writer_style = rearrange(torch.tensor(writer_style), "h w -> 1 1 h w")

        style_vector = style_extractor(writer_style)
        style_vector = rearrange(style_vector, "1 h w -> 1 w h")

        sequence = torch.IntTensor([tokenizer.encode(sequence)])
        batch_size = sequence.size(0)
        beta_len = len(beta_set)
        strokes = torch.randn((batch_size, time_steps, 2))

        for i in track(range(beta_len - 1, -1, -1)):
            alpha = alpha_set[i] * torch.ones((batch_size, 1, 1))
            beta = beta_set[i] * torch.ones((batch_size, 1, 1))
            alpha_next = alpha_set[i - 1] if i > 0 else torch.tensor(1.0)
            batch = (strokes, sequence, torch.sqrt(alpha), style_vector)

            with torch.no_grad():
                model_out, pen_lifts, _ = self(batch)

            if diffusion_mode == "standard":
                strokes = utils.standard_diffusion_step(
                    strokes, model_out, beta, alpha, add_sigma=bool(i)
                )
            else:
                strokes = utils.new_diffusion_step(
                    strokes, model_out, beta, alpha, alpha_next
                )

        pen_lifts = rearrange(F.sigmoid(pen_lifts), "h w 1 -> h w")
        strokes = torch.cat([strokes, pen_lifts], dim=-1)
        save_path = "/home/codeplayer/Studia/Inżynierka/Handwriting-Synthesis-Thesis-fr-fr/handwriting.png"

        utils.generate_stroke_image(
            strokes.detach().cpu().numpy(), scale=1.0, save_path=save_path
        )
