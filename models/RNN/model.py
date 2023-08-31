from typing import Tuple, Union

import lightning.pytorch as pl
import torch
import torch.nn as nn
from einops import rearrange, repeat
from rich.progress import track
from torch import Tensor
from torch.optim import Optimizer

from data.tokenizer import Tokenizer
from . import utils
from .network import MixtureDensityNetwork
from .window import GaussianWindow


class RNNModel(pl.LightningModule):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 400,
        num_window: int = 10,
        num_mixture: int = 20,
        vocab_size: int = 73,
        bias: float = None,
        clip_grads: Tuple[float, float] = None,
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
            _description_, by default None
        """

        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_window = num_window
        self.num_mixture = num_mixture
        self.bias = bias
        self.lstm_clip, self.mdn_clip = clip_grads

        self.lstm_0 = nn.LSTM(
            input_size=input_size + vocab_size,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.window = GaussianWindow(in_features=hidden_size, out_features=num_window)

        self.lstm_1 = nn.LSTM(
            input_size=(input_size + hidden_size + vocab_size),
            hidden_size=hidden_size,
            batch_first=True,
        )

        self.lstm_2 = nn.LSTM(
            input_size=(input_size + hidden_size + vocab_size),
            hidden_size=hidden_size,
            batch_first=True,
        )

        self.mdn = MixtureDensityNetwork(
            in_features=hidden_size * 3, out_features=num_mixture, bias=bias
        )

        if clip_grads[0] is not None and clip_grads[1] is not None:
            self.__register_layers_hook()

    def __register_layers_hook(self) -> None:
        lstm_tuple = (self.lstm_0, self.lstm_1, self.lstm_2)
        for lstm in lstm_tuple:
            for p in track(lstm.parameters(), description="Registering LSTM hook..."):
                p.register_hook(
                    lambda grad: torch.clamp(grad, -self.lstm_clip, self.lstm_clip)
                )

        for p in track(self.mdn.parameters(), description="Registering MDN hook..."):
            p.register_hook(
                lambda grad: torch.clamp(grad, -self.mdn_clip, self.mdn_clip)
            )

    def forward(
        self, batch: Tuple[Tensor, Tensor], *, states: Tuple[Tensor, ...] = None
    ) -> Tuple[Tuple[Tensor, ...], Tensor, Tuple[Tensor, ...]]:
        strokes, text = batch
        batch_size, steps, _ = strokes.size()

        hidden_1, hidden_2, hidden_3 = (
            states
            if states is not None
            else utils.get_initial_states(
                batch_size, self.hidden_size, device=self.device
            )
        )

        window_s = torch.zeros(
            batch_size,
            1,
            self.vocab_size,
            device=self.device,
        )
        kappa = torch.zeros(
            batch_size,
            self.num_window,
            dtype=torch.float32,
            device=self.device,
        )

        out_1, window = [], []

        for s in range(steps):
            strokes_s = strokes[:, s : s + 1, :]
            strokes_window = torch.cat([strokes_s, window_s], dim=-1)
            out_s, hidden_1 = self.lstm_0(strokes_window, hidden_1)
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
        pi, mu, sd, rho, eos = self.mdn(inputs)

        # noinspection PyUnboundLocalVariable
        return (
            (pi, mu, sd, rho, eos),
            phi,
            (hidden_1, hidden_2, hidden_3),
        )

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        batch, strokes = utils.add_prefix(batch)

        y_hat, _, _ = self(batch)

        loss = self.loss((y_hat, strokes))

        self.log("loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        batch, strokes = utils.add_prefix(batch)

        with torch.no_grad():
            y_hat, _, _ = self(batch)

        loss = self.loss((y_hat, strokes))

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self) -> Optimizer:
        return torch.optim.RMSprop(
            self.parameters(),
            lr=1e-4,
            momentum=0.9,
            weight_decay=1e-4,
            alpha=0.95,
            centered=True,
        )

    @staticmethod
    def loss(batch: Tuple[Tensor, ...], *, eps: float = 1e-8) -> Tensor:
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
        batch_size = strokes.size(0)

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

        # x = rearrange(x, "v -> v 1").repeat(1, num_components)
        x = repeat(x, "v -> v new_axis", new_axis=num_components)
        y = repeat(y, "v -> v new_axis", new_axis=num_components)

        mixtures = (mu_1, mu_2, sigma_1, sigma_2, rho)

        densities = utils.bivariate_gaussian(x, y, mixtures) + eps**2
        mixture_densities = torch.sum(torch.multiply(densities, pi), dim=1)
        # noinspection PyTypeChecker
        density = -torch.log(mixture_densities + eps)

        binary_log_likelihood = -torch.log(
            eos * eos_hat + (1.0 - eos) * (1.0 - eos_hat) + eps
        )

        return torch.sum(density + binary_log_likelihood) / batch_size

    def generate(
        self,
        raw_text: Union[str, Tensor],
        *,
        vocab: str,
        states: Tuple[Tensor, ...] = None,
        steps: int = 700,
        color: str = "black",
        primed: bool = False,
        stochastic: bool = True,
    ) -> Tensor:
        """
        _summary_

        Parameters
        ----------
        raw_text : str | Tensor
            _description_
        vocab : str
            _description_
        states : Tuple[Tensor, ...], optional
            _description_, by default None
        steps : int, optional
            _description_, by default 700
        color : str, optional
            _description_, by default "black"
        primed : bool, optional
            _description_, by default False
        stochastic : bool, optional
            _description_, by default True

        Returns
        -------
        Tensor
            _description_
        """

        if not primed:
            tokenizer = Tokenizer(vocab)
            eye = torch.eye(tokenizer.get_vocab_size(), device=self.device)
            indices = tokenizer.encode(raw_text)
            text = eye[indices]
        else:
            # noinspection PyUnresolvedReferences
            text = raw_text.copy()

        strokes = torch.zeros((1, 1, 3), device=self.device)
        outputs = []

        for _ in range(steps):
            batch = (strokes, text)
            pi, mu, sigma, rho, eos, phi = self(batch, states=states)

            pi, mu, sigma, rho, eos = (
                pi[0, 0],
                mu[0, 0],
                sigma[0, 0],
                rho[0, 0],
                eos[0, 0],
            )
            mixtures = (pi, mu, sigma, rho, eos)

            strokes_tmp = utils.get_mean_predictions(mixtures, stochastic=stochastic)
            is_last_phi_high = phi[0, 0, text.size(1) - 1] > 0.8
            is_eos = strokes_tmp[0, 2] > 0.5

            if is_last_phi_high or (phi[0, 0].argmax() == text.size(1) - 1 and is_eos):
                strokes_tmp[0, 2] = 1.0
                outputs.append(strokes_tmp)
                break

            outputs.append(strokes_tmp)
            strokes = rearrange(strokes_tmp, "x y eos -> 1 x y eos")

        return torch.cat(outputs, dim=0)

    # TODO: Complete, Check and Test (CCT)
    def generate_primed(
        self,
        primed_strokes: Tensor,
        text_priming: str,
        text: str,
        vocab: str,
        *,
        color: str = "black",
        steps: int = 700,
    ) -> Tensor:
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
        color : str, optional
            _description_, by default "black"
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
        primed_strokes, _ = utils.add_prefix(
            (rearrange(primed_strokes, "h w -> 1 h w"), text_cat),
            return_batch=False,
        )
        priming_steps = primed_strokes.size(1)

        for step in range(priming_steps):
            # x_strokes = primed_strokes[:, step].unsqueeze(1)
            x_strokes = rearrange(primed_strokes[:, step], "1 h -> 1 1 h")
            batch = (x_strokes, text_cat)
            _, _, states = self(batch)

        # noinspection PyUnboundLocalVariable
        return self.generate(
            text_cat,
            vocab=vocab,
            states=states,
            steps=steps,
            color=color,
            stochastic=True,
        )
