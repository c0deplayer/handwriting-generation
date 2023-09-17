import copy
from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor


class ExponentialMovingAverage(nn.Module):
    """
    Exponential Moving Average (EMA) is used to stabilize the training process of diffusion models
    by computing a moving average of the parameters, which can help to reduce
    the noise in the gradients and improve the performance of the model.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        ema_model: nn.Module = None,
        beta: float = 0.9999,
        update_after_step: int = 1000,
        update_every: int = 10,
        inv_gamma: float = 1.0,
        power: float = 2 / 3,
        min_value: float = 0.0,
    ):
        """
        _summary_

        Parameters
        ----------
        model : nn.Module
            _description_
        ema_model : nn.Module, optional
            _description_, by default None
        beta : float, optional
            _description_, by default 0.9999
        update_after_step : int, optional
            _description_, by default 1000
        update_every : int, optional
            _description_, by default 10
        inv_gamma : float, optional
            _description_, by default 1.0
        power : float, optional
            _description_, by default 2 / 3
        min_value : float, optional
            _description_, by default 0.0
        """

        super().__init__()

        self.beta = beta
        self.inv_gamma = inv_gamma
        self.power = power
        self.min_value = min_value
        self.__model = model
        self.ema_model = ema_model

        if ema_model is None:
            self.ema_model = copy.deepcopy(model)

        self.ema_model.requires_grad_(False)

        self.parameter_names = {
            name
            for name, param in self.ema_model.named_parameters()
            if param.dtype in [torch.float, torch.float16, torch.float32]
        }

        self.update_every = update_every
        self.update_after_step = update_after_step

        self.register_buffer("initiated", torch.tensor([False]))
        self.register_buffer("step", torch.tensor([0]))

    @property
    def model(self) -> nn.Module:
        return self.ema_model

    # noinspection PyUnresolvedReferences
    @torch.no_grad()
    def update_moving_average(self) -> None:
        """
        _summary_
        """

        current_decay = self.get_current_decay()

        for (_, ma_params), (_, current_params) in zip(
            self.get_params_iter(self.ema_model), self.get_params_iter(self.__model)
        ):
            if ma_params.data is None:
                ma_params.data.copy_(current_params.data)
            else:
                ma_params.data.lerp_(current_params.data, 1.0 - current_decay)

    def update(self) -> None:
        """
        _summary_
        """

        step = self.step.item()
        self.step += 1

        if step % self.update_every:
            return

        if self.step <= self.update_after_step:
            self.copy_params_from_model_to_ema()
            return

        if not self.initiated.item():
            self.copy_params_from_model_to_ema()
            self.initiated.data.copy_(torch.Tensor([True]))

        self.update_moving_average()

    def get_current_decay(self) -> float:
        step = max(self.step.item() - self.update_after_step - 1, 0.0)
        value = 1 - (1 + step / self.inv_gamma) ** -self.power

        return 0.0 if step <= 0 else min(max(value, self.min_value), self.beta)

    def copy_params_from_model_to_ema(self) -> None:
        """
        _summary_
        """

        for (_, ma_params), (_, current_params) in zip(
            self.get_params_iter(self.ema_model), self.get_params_iter(self.__model)
        ):
            ma_params.data.copy_(current_params.data)

    def get_params_iter(self, model: nn.Module) -> Tuple[str, Tensor]:
        for name, param in model.named_parameters():
            if name not in self.parameter_names:
                continue

            yield name, param

    def __call__(self, *args, **kwargs) -> Tensor:
        """
        _summary_

        Returns
        -------
        Tensor
            _description_
        """

        return self.ema_model(*args, **kwargs)
