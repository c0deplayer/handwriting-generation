from typing import Any, Dict

from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer


class InverseSqrtScheduler(LRScheduler):
    """
    The InverseSqrtScheduler is a learning rate scheduler used in training models.
    It gradually increases the learning rate for a specified number of warm-up steps
    and then decreases it inversely proportional to the square root of
    the step count. This helps stabilize training for Transformer models.

    Args:
        optimizer (Optimizer): The optimizer to adjust the learning rate for.
        lr_mul (float, optional): The initial learning rate multiplier. (default: 1.0)
        d_model (int, optional): The model's dimensionality. (default: 128)
        n_warmup_steps (int, optional): The number of warm-up steps. (default: 4000)
        last_epoch (int, optional): The index of the last epoch. (default: -1)
        verbose (bool, optional): If True, print learning rate changes. (default: False)

    Example:
        ```
        # Create a model and an optimizer
        model = YourModel()
        optimizer = Optimizer(model.parameters(), lr=0.001)

        # Create an InverseSqrtScheduler with default settings
        scheduler = InverseSqrtScheduler(optimizer)

        # Training loop
        for epoch in range(num_epochs):
            # Train your model
            ...

            # Update the learning rate
            scheduler.step(epoch)
        ```
    """

    def __init__(
        self,
        optimizer: Optimizer,
        lr_mul: float = 1.0,
        d_model: int = 128,
        n_warmup_steps: int = 4000,
        last_epoch: int = -1,
        verbose: bool = False,
    ) -> None:
        if not isinstance(optimizer, Optimizer):
            raise TypeError(f"{type(optimizer).__name__} is not an Optimizer")
        if lr_mul <= 0.0:
            raise ValueError(f"Invalid learning rate multiplier: {lr_mul}")
        if d_model <= 0:
            raise ValueError(f"Invalid d_model: {d_model}")
        if n_warmup_steps <= 0:
            raise ValueError(f"Invalid number of warmup steps: {n_warmup_steps}")

        self.d_model = d_model
        self.lr_mul = lr_mul
        self.n_warmup_steps = n_warmup_steps
        self.optimizer = optimizer
        self.verbose = verbose
        self._step_count = 0
        self._last_lr = None

        super().__init__(optimizer, last_epoch, verbose)

    def step(self, epoch: int = None) -> None:
        self._step_count += 1
        self.optimizer._step_count += 1

        self._last_lr = self.update_lr()

    def get_last_lr(self) -> float:
        return self._last_lr

    def get_lr(self) -> float:
        for g in self.optimizer.param_groups:
            return g["lr"]

    def get_lr_scale(self) -> float:
        return (self.d_model ** (-0.5)) * min(
            self._step_count ** (-0.5),
            self._step_count * self.n_warmup_steps ** (-1.5),
        )

    @staticmethod
    def print_lr(
        is_verbose: bool,
        group: Dict[str, Any],
        lr: float,
        epoch: int = None,
    ) -> None:
        if is_verbose:
            if epoch is None:
                print(f"Learning rate: {group:.4e} --> {lr:.4e}")
            else:
                print(f"Epoch {epoch} -- Learning rate: {group:.4e} --> {lr:.4e}")

    def set_lr(self, optimizer: Optimizer, lr: float) -> None:
        for group in optimizer.param_groups:
            self.print_lr(self.verbose, group["lr"], lr)
            group["lr"] = lr

    def update_lr(self) -> float:
        lr = self.lr_mul * self.get_lr_scale()
        self.set_lr(self.optimizer, lr)

        return lr
