from pathlib import Path
from typing import Optional, Union

import pytorch_lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint


class PeriodicCheckpoint(ModelCheckpoint):
    def __init__(
        self,
        every: int = 100,
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
        )
        self.every = every

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if (pl_module.current_epoch + 1) % self.every == 0:
            if self.dirpath is None:
                raise ValueError("You must specify a directory to save model")

            print(f"Saving model from {pl_module.current_epoch} epoch")
            monitor_candidates = self._monitor_candidates(trainer)
            filepath = self.format_checkpoint_name(monitor_candidates)
            trainer.save_checkpoint(filepath)
