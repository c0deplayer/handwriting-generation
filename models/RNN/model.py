from typing import Tuple, Union, Dict

import lightning.pytorch as pl
import torch
import torch.nn as nn
from einops import rearrange, repeat
from rich.progress import track
from torch import Tensor
from torch.optim import Optimizer

from data.tokenizer import Tokenizer
from models.Diffusion.utils import generate_stroke_image
from . import utils
from .lstm import LSTM
from .network import MixtureDensityNetwork
from .window import GaussianWindow


class RNNModel(pl.LightningModule):
    """
    _summary_
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 400,
        num_window: int = 10,
        num_mixture: int = 20,
        vocab_size: int = 73,
        bias: float = None,
        clip_grads: Tuple[float, float] = (None, None),
    ) -> None:
        """
        _summary_

        Parameters
        ----------
        input_size : int
            _description_
        hidden_size : int, optional
            _description_, by default 400
        num_window : int, optional
            _description_, by default 10
        num_mixture : int, optional
            _description_, by default 20
        vocab_size : int, optional
            _description_, by default 73
        bias : float, optional
            _description_, by default None
        clip_grads : Tuple[float, float], optional
            _description_, by default (None, None)
        """

        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_window = num_window
        self.num_mixture = num_mixture
        self.bias = bias
        self.lstm_clip, self.mdn_clip = clip_grads

        self.lstm_0 = LSTM(
            input_size=input_size + vocab_size,
            hidden_size=hidden_size,
            batch_first=True,
            layer_norm=True,
        )

        self.window = GaussianWindow(in_features=hidden_size, out_features=num_window)

        self.lstm_1 = LSTM(
            input_size=(input_size + hidden_size + vocab_size),
            hidden_size=hidden_size,
            batch_first=True,
            layer_norm=True,
        )

        self.lstm_2 = LSTM(
            input_size=(input_size + hidden_size + vocab_size),
            hidden_size=hidden_size,
            batch_first=True,
            layer_norm=True,
        )

        self.mdn = MixtureDensityNetwork(
            in_features=hidden_size * 3, out_features=num_mixture, bias=bias
        )

        self.lstm_0.apply(self.__init_weights__)
        self.lstm_1.apply(self.__init_weights__)
        self.lstm_2.apply(self.__init_weights__)
        self.window.apply(self.__init_weights__)
        self.mdn.apply(self.__init_weights__)

        self.save_hyperparameters()

    def on_before_optimizer_step(self, optimizer: Optimizer) -> None:
        """
        _summary_

        Parameters
        ----------
        optimizer : Optimizer
            _description_
        """
        
        if self.lstm_clip is not None:
            lstm_tuple = (self.lstm_0, self.lstm_1, self.lstm_2)
            for lstm in lstm_tuple:
                nn.utils.clip_grad_value_(lstm.parameters(), self.lstm_clip)

        if self.mdn_clip is not None:
            nn.utils.clip_grad_value_(self.mdn.parameters(), self.mdn_clip)

    # noinspection PyProtectedMember
    @staticmethod
    def __init_weights__(model: nn.Module) -> None:
        """
        _summary_

        Parameters
        ----------
        model : nn.Module
            _description_
        """
        
        class_name = model.__class__.__name__

        if (
            class_name.find("LSTM") != -1
            or class_name.find("MixtureDensityNetwork") != -1
            or class_name.find("GaussianWindow") != -1
        ):
            for name, param in model.named_parameters():
                if "weight" in name:
                    utils.truncated_normal_(param.data, mean=0.0, std=0.075)

    def forward(
        self,
        batch: Tuple[Tensor, Tensor],
        *,
        states: Tuple[Tensor, Tensor, Tensor] = None,
        kw: Tuple[Tensor, Tensor] = (None, None),
        return_all: bool = False,
    ) -> Union[
        Tuple[
            Tuple[Tensor, ...],
            Tuple[Tensor, Tensor, Tensor],
            Tuple[Tensor, Tensor, Tensor],
        ],
        Tuple[Tensor, ...],
    ]:
        """
        _summary_

        Parameters
        ----------
        batch : Tuple[Tensor, Tensor]
            _description_
        states : Tuple[Tensor, Tensor, Tensor], optional
            _description_, by default None
        kw : Tuple[Tensor, Tensor], optional
            _description_, by default (None, None)
        return_all : bool, optional
            _description_, by default False

        Returns
        -------
        Union[Tuple[ 
            Tuple[Tensor, ...], 
            Tuple[Tensor, Tensor, Tensor], 
            Tuple[Tensor, Tensor, Tensor], 
        ], 
        Tuple[Tensor, ...], ]
            _description_
        """
        
        strokes, text = batch
        kappa, window = kw
        batch_size = strokes.size(0)

        hidden_1, hidden_2, hidden_3 = (
            states
            if states is not None
            else utils.get_initial_states(
                1, batch_size, self.hidden_size, device=self.device
            )
        )

        window_s = (
            window
            if window is not None
            else torch.zeros(
                batch_size,
                1,
                self.vocab_size,
                device=self.device,
            )
        )

        kappa = (
            kappa
            if kappa is not None
            else torch.zeros(
                batch_size,
                self.num_window,
                dtype=torch.float32,
                device=self.device,
            )
        )

        out_1, window = [], []

        for stroke in strokes.unbind(1):
            stroke = rearrange(stroke, "b v -> b 1 v")
            stroke_window = torch.cat([stroke, window_s], dim=-1)

            out_s, hidden_1 = self.lstm_0(stroke_window, hidden_1)
            phi, kappa, window_s = self.window(out_s, text, kappa)

            out_1.append(out_s)
            window.append(window_s)

        out_1 = torch.cat(out_1, dim=1)
        window = torch.cat(window, dim=1)

        inputs = torch.cat([strokes, out_1, window], dim=-1)
        out_2, hidden_2 = self.lstm_1(inputs, hidden_2)

        inputs = torch.cat([strokes, out_2, window], dim=-1)
        out_3, hidden_3 = self.lstm_2(inputs, hidden_3)

        inputs = torch.cat([out_1, out_2, out_3], dim=-1)
        pi, mu, sigma, rho, eos = self.mdn(inputs)

        if return_all:
            # noinspection PyUnboundLocalVariable
            return (
                (pi, mu, sigma, rho, eos),
                (phi, kappa, window),
                (hidden_1, hidden_2, hidden_3),
            )
        else:
            return pi, mu, sigma, rho, eos

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        y_hat = self(batch)

        loss = self.loss((y_hat, batch[0]))

        self.log("loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        with torch.no_grad():
            y_hat = self(batch)

        loss = self.loss((y_hat, batch[0]))

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self) -> Dict[str, Optimizer]:
        optimizer = torch.optim.Adam(
            self.parameters(), lr=1e-4, betas=(0.9, 0.95), weight_decay=1e-6
        )

        return {"optimizer": optimizer}

    @staticmethod
    def loss(batch: Tuple[Tensor, Tensor], *, eps: float = 1e-8) -> Tensor:
        """
        _summary_

        Parameters
        ----------
        batch : Tuple[Tensor, ...]
            _description_
        eps : float, optional
            _description_, by default 1e-8

        Returns
        -------
        float
            _description_
        """

        y_hat, strokes = batch
        pi, mu, sigma, rho, eos_hat = y_hat
        batch_size, time_steps, _ = strokes.size()

        num_components = pi.size(-1)
        mu_1 = mu[:, :, :num_components]
        mu_2 = mu[:, :, num_components:]

        sigma_1 = sigma[:, :, :num_components]
        sigma_2 = sigma[:, :, num_components:]

        pi = utils.reshape_down(pi, strokes)
        mu_1 = utils.reshape_down(mu_1, strokes)
        mu_2 = utils.reshape_down(mu_2, strokes)
        sigma_1 = utils.reshape_down(sigma_1, strokes)
        sigma_2 = utils.reshape_down(sigma_2, strokes)
        rho = utils.reshape_down(rho, strokes)
        eos_hat = utils.reshape_down(eos_hat, strokes)
        strokes = utils.reshape_down(strokes)

        eos_hat = eos_hat[:, 0]
        x, y, eos = strokes[:, 0], strokes[:, 1], strokes[:, 2]

        x = repeat(x, "v -> v new_axis", new_axis=num_components)
        y = repeat(y, "v -> v new_axis", new_axis=num_components)

        mixtures = (mu_1, mu_2, sigma_1, sigma_2, rho)

        densities = utils.bi_variate_gaussian(x, y, mixtures) + eps**2
        mixture_densities = torch.sum(torch.multiply(densities, pi), dim=1)
        # noinspection PyTypeChecker
        density = -torch.log(mixture_densities + eps)

        binary_log_likelihood = -torch.log(
            eos * eos_hat + (1.0 - eos) * (1.0 - eos_hat) + eps
        )

        return torch.sum(density + binary_log_likelihood) / (batch_size * time_steps)

    def generate(
        self,
        raw_text: Union[str, Tensor],
        *,
        vocab: str,
        states: Tuple[Tensor, Tensor, Tensor] = None,
        kw: Tuple[Tensor, Tensor] = (None, None),
        n_steps: int = 700,
        primed: bool = False,
        stochastic: bool = True,
    ) -> None:
        """
        _summary_

        Parameters
        ----------
        raw_text : Union[str, Tensor]
            _description_
        vocab : str
            _description_
        states : Tuple[Tensor, Tensor, Tensor], optional
            _description_, by default None
        kw : Tuple[Tensor, Tensor], optional
            _description_, by default (None, None)
        n_steps : int, optional
            _description_, by default 700
        primed : bool, optional
            _description_, by default False
        stochastic : bool, optional
            _description_, by default True
        """
        
        if not primed:
            tokenizer = Tokenizer(vocab)
            eye = torch.eye(tokenizer.get_vocab_size(), device=self.device)
            indices = tokenizer.encode(raw_text)
            text = eye[indices]
            text = rearrange(text, "o h -> 1 o h")
        else:
            # noinspection PyUnresolvedReferences
            text = raw_text.copy()

        strokes = torch.zeros((1, 1, 3), device=self.device)
        outputs = []

        for _ in track(range(n_steps)):
            batch = (strokes, text)
            mixtures, pkw, states = self(batch, states=states, kw=kw, return_all=True)
            pi, mu, sigma, rho, eos = mixtures
            phi, kw = pkw[0], pkw[1:]

            pi, mu, sigma, rho, eos, phi = (
                pi.squeeze(),
                mu.squeeze(),
                sigma.squeeze(),
                rho.squeeze(),
                eos.squeeze(),
                phi.squeeze(),
            )

            mixtures = (pi, mu, sigma, rho, eos)

            strokes_tmp = utils.get_mean_predictions(mixtures, stochastic=stochastic)
            is_last_phi_high = phi[text.size(1) - 1] > 0.8
            is_eos = strokes_tmp[0, 2] > 0.5

            if is_last_phi_high or (phi.argmax() == text.size(1) - 1 and is_eos):
                strokes_tmp[0, 2] = 1.0
                outputs.append(strokes_tmp)
                break

            outputs.append(strokes_tmp)
            strokes = rearrange(strokes_tmp, "1 v -> 1 1 v")

        strokes = torch.cat(outputs, dim=0)
        save_path = "handwriting.png"

        generate_stroke_image(
            strokes.detach().cpu().numpy(), scale=1.0, save_path=save_path
        )

    def generate_primed(
        self,
        primed_strokes: Tensor,
        text_priming: str,
        text: str,
        vocab: str,
        *,
        steps: int = 700,
    ) -> None:
        """
        _summary_

        Parameters
        ----------
        primed_strokes : Tensor
            _description_
        text_priming : str
            _description_
        text : str
            _description_
        vocab : str
            _description_
        steps : int, optional
            _description_, by default 700

        Returns
        -------
        Tensor
            _description_
        """
        tokenizer = Tokenizer(vocab)
        eye = torch.eye(tokenizer.get_vocab_size(), device=self.device)

        indices = tokenizer.encode(text_priming)
        text_priming = eye[indices]

        indices = tokenizer.encode(text)
        text = eye[indices]

        text_cat = torch.cat([text_priming, text], dim=1)

        primed_strokes = rearrange(primed_strokes, "h w -> 1 h w")
        batch_size, priming_steps, input_len = primed_strokes.size()

        prefix = torch.zeros([batch_size, 1, input_len], device=self.device)
        prefix[:, :, 2] = 1.0
        primed_strokes = torch.cat([prefix, primed_strokes[:, :-1]], dim=1)

        states, kw = None, (None, None)
        for _ in priming_steps.unbind(1):
            x_strokes = rearrange(primed_strokes, "1 h -> 1 1 h")
            batch = (x_strokes, text_cat)
            _, pkw, states = self(batch, states=states, kw=kw, return_all=True)
            kw = pkw[1:]

        # noinspection PyUnboundLocalVariable
        self.generate(
            text_cat,
            vocab=vocab,
            states=states,
            kw=kw,
            n_steps=steps,
            stochastic=True,
        )
