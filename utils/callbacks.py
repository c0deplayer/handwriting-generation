from pathlib import Path
from typing import Any, Callable, Literal, Optional, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import ModelCheckpoint


class EarlyStopper:
    """
    Early stopping to stop the training when the validation metric is not improving.
    """

    mode_dict = {"min": torch.lt, "max": torch.gt}

    def __init__(
        self,
        patience: int = 3,
        min_delta: float = 0.0,
        mode: Literal["min", "max"] = "min",
        checkpoint_path: Path = Path("./checkpoint.pt"),
        *,
        verbose: bool = False,
    ):
        """
        Initialize the EarlyStopper.

        Args:
            patience (int): Number of epochs with no improvement after which training will be stopped.
            min_delta (float): Minimum change in the monitored metric to qualify as an improvement.
            mode (Literal["min", "max"]): Whether to look for a minimum or maximum in the monitored metric.
            verbose (bool): If True, prints a message for each validation metric check.
        """
        self.patience = patience
        self.mode = mode
        self.min_delta = (
            torch.tensor([min_delta])
            if mode == "max"
            else torch.tensor([-min_delta])
        )
        self.counter = 0
        self.best_score = (
            torch.tensor(float("inf"))
            if mode == "min"
            else torch.tensor(float("-inf"))
        )
        self.checkpoint_path = checkpoint_path
        self.verbose = verbose
        self.__model_state_dict = {}

    def early_stop(self, model: nn.Module, validation_metric: float) -> bool:
        """
        Check if training should be stopped early.

        Args:
            validation_metric (float): The current value of the monitored metric.

        Returns:
            bool: True if training should be stopped, False otherwise.
        """
        current = torch.tensor([validation_metric])

        if self.monitor_metric(current - self.min_delta, self.best_score):
            self.best_score = current
            self.counter = 0

            self.__save_model(model, self.checkpoint_path)
        else:
            self.counter += 1

            if self.verbose:
                msg = f"The metric has not improved for {self.counter} epochs"

                if self.counter >= self.patience:
                    msg += ". Stopping training"

                print(msg)

        return self.counter >= self.patience

    def __save_model(self, model: nn.Module, path: Path) -> None:
        """
        Save the model state dictionary to a file.

        Args:
            model (nn.Module): The model to save.
            path (Path): The file path where the model state dictionary will be saved.
        """
        model = model.to("cpu")
        torch.save(model.state_dict(), path)
        self.__model_state_dict = model.state_dict()

    @property
    def monitor_metric(self) -> Callable:
        """
        Get the comparison function based on the mode.

        Returns:
            Callable: The comparison function (torch.lt or torch.gt).
        """
        return self.mode_dict[self.mode]

    @property
    def model_state_dict(self) -> dict[str, Any]:
        """
        Get the saved model state dictionary.

        Returns:
            dict: The saved model state dictionary.
        """
        return self.__model_state_dict


class PeriodicCheckpoint(ModelCheckpoint):
    """
    Custom callback for creating model checkpoints periodically during training.

    Args:
        every_steps (int): Number of training steps between successive checkpoints.
        dirpath (Optional[Union[str, Path]]): Directory path for saving the checkpoints.
        monitor (Optional[str]): Quantity to monitor for best model selection.
        filename (Optional[str]): Filename template for saving checkpoints.
        auto_insert_metric_name (bool): If True, automatically inserts the metric name into the filename.

    Raises:
        ValueError: If 'dirpath' is None or 'every_steps' is not greater than zero.
    """

    def __init__(
        self,
        every_steps: int = 10000,
        dirpath: Optional[Union[str, Path]] = None,
        monitor: Optional[str] = None,
        filename: Optional[str] = None,
        auto_insert_metric_name: bool = False,
    ) -> None:
        super().__init__(
            dirpath=dirpath,
            monitor=monitor,
            filename=filename,
            auto_insert_metric_name=auto_insert_metric_name,
            every_n_train_steps=0,
            every_n_epochs=0,
        )
        if self.dirpath is None:
            raise ValueError("You must specify a directory to save model")
        if every_steps <= 0:
            raise ValueError("Every steps must be greater than zero")

        self.every_steps = every_steps

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        *args,
        **kwargs,
    ) -> None:
        if (
            pl_module.global_step % self.every_steps == 0
            and pl_module.global_step != 0
        ):
            print(f"Saving model from {pl_module.current_epoch} epoch")
            monitor_candidates = self._monitor_candidates(trainer)
            filepath = self.format_checkpoint_name(monitor_candidates)
            trainer.save_checkpoint(filepath)
