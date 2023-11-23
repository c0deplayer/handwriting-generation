import copy
from typing import Tuple, Optional

import torch
import torch.nn as nn
from torch import Tensor


class ExponentialMovingAverage(nn.Module):
    """
    Exponential Moving Average (EMA) implementation for model parameter averaging.

    This class provides a mechanism for maintaining an EMA of a PyTorch model's parameters
    to improve the stability and convergence of training. It updates the EMA model with a
    moving average of the original model's parameters over time.

    Notes:
        If gamma=1 and power=1, implements a simple average. gamma=1, power=2/3 are
        good values for models you plan to train for a million or more steps (reaches decay
        factor 0.999 at 31.6K steps, 0.9999 at 1M steps), gamma=1, power=3/4 for models
        you plan to train for less (reaches decay factor 0.999 at 10K steps, 0.9999 at
        215.4k steps).

    Args:
        model (nn.Module): The original model whose parameters are to be averaged.
        ema_model (Optional[nn.Module]): The Exponential Moving Average (EMA) model to be updated. If not provided,
                             a copy of the original model is used.
        beta (float): The decay factor for EMA, usually close to 1.0. A higher value makes the EMA update slower.
        update_after_step (int): The step after which to start updating the EMA model.
        update_every (int): The frequency of EMA updates.
        inv_gamma (float): The inverse gamma value for EMA decay calculation.
        power (float): The power value for EMA decay calculation.
        min_value (float): The minimum value for the EMA decay.

    Attributes:
        initiated (torch.Tensor): A tensor indicating if EMA initiation has occurred.
        step (torch.Tensor): A tensor that keeps track of the number of steps.

    Example:
        ```
        original_model = ...  # Your original PyTorch model
        ema = ExponentialMovingAverage(original_model)

        for epoch in range(num_epochs):
            for batch in dataloader:
                loss = compute_loss(batch)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                ema.update()  # Update the EMA model at the specified frequency

            # Use the EMA model for evaluation or inference
            ema_model = ema.model
            evaluate(ema_model, dataloader)
        ```
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        ema_model: Optional[nn.Module] = None,
        beta: float = 0.9999,
        update_after_step: int = 1000,
        update_every: int = 10,
        inv_gamma: float = 1.0,
        power: float = 2 / 3,
        min_value: float = 0.0,
    ):
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

    @torch.inference_mode()
    def update_moving_average(self) -> None:
        current_decay = self.get_current_decay()

        for (_, ma_params), (_, current_params) in zip(
            self.get_params_iter(self.ema_model), self.get_params_iter(self.__model)
        ):
            if ma_params.data is None:
                ma_params.data.copy_(current_params.data)
            else:
                ma_params.data.lerp_(current_params.data, 1.0 - current_decay)

    def update(self) -> None:
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
        return self.ema_model(*args, **kwargs)
