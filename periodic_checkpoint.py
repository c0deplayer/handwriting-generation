from pathlib import Path
from typing import Optional, Union

import pytorch_lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint


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
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, *args, **kwargs
    ) -> None:
        if pl_module.global_step % self.every_steps == 0 and pl_module.global_step != 0:
            print(f"Saving model from {pl_module.current_epoch} epoch")
            monitor_candidates = self._monitor_candidates(trainer)
            filepath = self.format_checkpoint_name(monitor_candidates)
            trainer.save_checkpoint(filepath)
