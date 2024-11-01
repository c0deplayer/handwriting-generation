import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar, Literal

import lightning as lp
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import nn

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class EarlyStopper:
    """Early stopping to stop the training when the validation metric is not improving."""

    mode_dict: ClassVar = {"min": torch.lt, "max": torch.gt}

    def __init__(
        self,
        patience: int = 3,
        min_delta: float = 0.0,
        mode: Literal["min", "max"] = "min",
        checkpoint_path: Path = Path("./checkpoint.pt"),
        *,
        verbose: bool = False,
    ):
        """Initialize the EarlyStopper.

        Args:
            patience (int): Number of epochs with no improvement after which
                            training will be stopped.
            min_delta (float): Minimum change in the monitored metric to qualify
                            as an improvement.
            mode (Literal["min", "max"]): Whether to look for a minimum or maximum
                                        in the monitored metric.
            checkpoint_path (Path): Path to save the model checkpoint.
            verbose (bool): If True, prints a message for each validation metric
                            check.

        """
        self.patience = patience
        self.mode = mode
        self.min_delta = torch.tensor(
            min_delta if mode == "max" else -min_delta,
        )
        self.counter = 0
        self.best_score = torch.tensor(
            float("inf") if mode == "min" else float("-inf"),
        )
        self.checkpoint_path = checkpoint_path
        self.verbose = verbose
        self._model_state_dict = {}

    def early_stop(self, model: nn.Module, validation_metric: float) -> bool:
        """Check if training should be stopped early.

        Args:
            model (nn.Module): The model to save if the validation metric
                               improves.
            validation_metric (float): The current value of the monitored
                                       metric.

        Returns:
            bool: True if training should be stopped, False otherwise.

        """
        current = torch.tensor([validation_metric])

        if self.monitor_metric(current - self.min_delta, self.best_score):
            self.best_score = current
            self.counter = 0
            self._save_model(model)
        else:
            self.counter += 1

            if self.verbose:
                msg = f"The metric has not improved for {self.counter} epochs"
                if self.counter >= self.patience:
                    msg += ". Stopping training"

                logger.info(msg)

        return self.counter >= self.patience

    def _save_model(self, model: nn.Module) -> None:
        """Save the model state dictionary to a file.

        Args:
            model (nn.Module): The model to save.

        """
        model.to("cpu")
        torch.save(model.state_dict(), self.checkpoint_path)
        self._model_state_dict = model.state_dict()

    @property
    def monitor_metric(self) -> Callable:
        """Get the saved model state dictionary.

        Returns:
            Callable: The saved model state dictionary.

        """
        return self.mode_dict[self.mode]

    @property
    def model_state_dict(self) -> dict[str, Any]:
        """Get the saved model state dictionary.

        Returns:
            dict: The saved model state dictionary.

        """
        return self._model_state_dict


class PeriodicCheckpoint(ModelCheckpoint):
    """Custom callback for creating model checkpoints periodically during training."""

    def __init__(
        self,
        every_steps: int = 10000,
        dirpath: str | Path | None = None,
        monitor: str | None = None,
        filename: str | None = None,
        *,
        auto_insert_metric_name: bool = False,
    ) -> None:
        """Initialize the PeriodicCheckpoint.

        Args:
            every_steps (int): Number of steps between each checkpoint. Defaults
                            to 10000.
            dirpath (str | Path | None): Directory path to save the model
                                        checkpoints. Defaults to None.
            monitor (str | None): Metric to monitor for saving checkpoints.
            filename (str | None): Filename for the checkpoint. Defaults to None.
            auto_insert_metric_name (bool): If True, automatically insert the
                                            metric name into the filename.
                                            Defaults to False.

        Raises:
            ValueError: If dirpath is None or every_steps is less than or equal to
                        zero.

        """
        if dirpath is None:
            error_message = "You must specify a directory to save model"
            raise ValueError(error_message)
        if every_steps <= 0:
            error_message = "Every steps must be greater than zero"
            raise ValueError(error_message)

        super().__init__(
            dirpath=dirpath,
            monitor=monitor,
            filename=filename,
            auto_insert_metric_name=auto_insert_metric_name,
            every_n_train_steps=0,
            every_n_epochs=0,
        )

        self.every_steps = every_steps

    def on_train_batch_end(
        self,
        trainer: lp.Trainer,
        pl_module: lp.LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Call when the training batch ends.

        Args:
            trainer (lp.Trainer): The trainer instance.
            pl_module (lp.LightningModule): The LightningModule being trained.
            outputs (STEP_OUTPUT): The outputs from the training step.
            batch (Any): The current batch of data.
            batch_idx (int): The index of the current batch.

        """
        if (
            pl_module.global_step % self.every_steps == 0
            and pl_module.global_step != 0
        ):
            msg = f"Saving model from {pl_module.current_epoch} epoch"
            logger.info(msg)
            monitor_candidates = self._monitor_candidates(trainer)
            filepath = self.format_checkpoint_name(monitor_candidates)
            trainer.save_checkpoint(filepath)
