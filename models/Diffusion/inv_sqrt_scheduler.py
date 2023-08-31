from typing import Any, Dict

from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer


class InverseSqrtScheduler(LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        lr_mul: float = 1.0,
        d_model: int = 128,
        n_warmup_steps: int = 10000,
        last_epoch: int = -1,
        verbose: bool = False,
    ) -> None:
        """
        _summary_

        Parameters
        ----------
        optimizer : Optimizer
            _description_
        lr_mul : float, optional
            _description_, by default 1.0
        d_model : int, optional
            _description_, by default 128
        n_warmup_steps : int, optional
            _description_, by default 10000
        last_epoch : int, optional
            _description_, by default ...
        verbose : bool, optional
            _description_, by default False

        Raises
        ------
        TypeError
            _description_
        """

        if not isinstance(optimizer, Optimizer):
            raise TypeError(f"{type(optimizer).__name__} is not an Optimizer")
        if lr_mul <= 0.0:
            raise ValueError(f"Invalid learning rate multiplier: {lr_mul}")
        if d_model <= 0:
            raise ValueError(f"Invalid d_model: {d_model}")
        if n_warmup_steps < 0:
            raise ValueError(f"Invalid number of warmup steps: {n_warmup_steps}")

        self.d_model = d_model
        self.lr_mul = lr_mul
        self.n_warmup_steps = n_warmup_steps
        self.optimizer = optimizer
        self.verbose = verbose
        self._step_count = 0
        self._last_lr = None

        super().__init__(optimizer, last_epoch, verbose)

    def step(self, epoch: int = None) -> float:
        if self._step_count == 0:
            self._step_count += 1
            # noinspection PyProtectedMember,PyUnresolvedReferences
            self.optimizer._step_count += 1

            return self.get_lr()

        self._last_lr = self.update_lr()

        return self._last_lr

    def get_last_lr(self) -> float:
        """
        Return last computed learning rate by current scheduler.
        """

        return self._last_lr

    def get_lr(self) -> float:
        for g in self.optimizer.param_groups:
            return g["lr"]

    def get_lr_scale(self) -> float:
        return (self.d_model ** (-0.5)) * min(
            self._step_count ** (-0.5),
            self._step_count * self.n_warmup_steps ** (-1.5),
        )

    # noinspection PyMethodOverriding
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
        self._step_count += 1
        lr = self.lr_mul * self.get_lr_scale()
        self.set_lr(self.optimizer, lr)

        return lr
