import copy

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
        update_after_step: int = 100,
        update_every: int = 10,
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
            _description_, by default 100
        update_every : int, optional
            _description_, by default 10
        """
        
        super().__init__()

        self.beta = beta
        self.__model = model
        self.ema_model = ema_model

        if ema_model is None:
            self.ema_model = copy.deepcopy(model)

        self.ema_model.requires_grad_(False)

        self.update_every = update_every
        self.update_after_step = update_after_step

        self.register_buffer("initiated", torch.tensor([False]))
        self.register_buffer("step", torch.tensor([0]))

    @property
    def model(self) -> nn.Module:
        return self.ema_model

    @torch.no_grad()
    def update_moving_average(
        self, ma_model: nn.Module, current_model: nn.Module
    ) -> None:
        """
        _summary_

        Parameters
        ----------
        ma_model : nn.Module
            _description_
        current_model : nn.Module
            _description_
        """
        
        for current_params, ma_params in zip(
            current_model.parameters(), ma_model.parameters()
        ):
            old_weight, up_weight = ma_params.data, current_params.data
            if old_weight is None:
                ma_params.data.copy_(up_weight)
            else:
                ma_params.data.copy_(
                    old_weight * self.beta + (1 - self.beta) * up_weight
                )

    def update(
        self,
    ) -> None:
        """
        _summary_
        """
        
        step = self.step.item()
        self.step += 1

        if not step % self.update_every:
            return

        if self.step <= self.update_after_step:
            self.copy_params_from_model_to_ema()
            return

        if not self.initiated.item():
            self.copy_params_from_model_to_ema()
            self.initted.data.copy_(torch.Tensor([True]))

        self.update_moving_average(self.ema_model, self.__model)

    def copy_params_from_model_to_ema(self) -> None:
        """
        _summary_
        """
        
        for (_, ma_params), (_, current_params) in zip(
            self.get_params_iter(self.ema_model), self.get_params_iter(self.__model)
        ):
            ma_params.data.copy_(current_params.data)

        for (_, ma_buffers), (_, current_buffers) in zip(
            self.get_buffers_iter(self.ema_model), self.get_buffers_iter(self.__model)
        ):
            ma_buffers.data.copy_(current_buffers.data)

    def __call__(self, *args, **kwargs) -> Tensor:
        """
        _summary_

        Returns
        -------
        Tensor
            _description_
        """
        
        return self.ema_model(*args, **kwargs)
