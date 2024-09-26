import lightning as L
import torch
import torch.nn as nn
from einops import repeat
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import Tensor
from torchmetrics.classification import MulticlassAccuracy
from torchvision import models
from torchvision.models import Inception_V3_Weights
from torchvision.transforms import v2


class InceptionV3(L.LightningModule):
    def __init__(self, num_class: int, gen_model_type: str) -> None:
        super(InceptionV3, self).__init__()

        self.__normalize = (
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            if gen_model_type == "diffusion"
            else nn.Identity()
        )

        self.gen_model_type = gen_model_type

        self.inception = models.inception_v3(
            weights=Inception_V3_Weights.DEFAULT
        )

        self.train_accuracy, self.val_accuracy = (
            MulticlassAccuracy(num_classes=num_class, average="macro"),
            MulticlassAccuracy(num_classes=num_class, average="macro"),
        )

        self.__froze_and_change_layers(num_class)

    def __froze_and_change_layers(self, num_class: int) -> None:
        for name, param in self.inception.named_parameters():
            if "Mixed_5" not in name:
                param.requires_grad = False

        self.inception.Mixed_6a = nn.Identity()
        self.inception.Mixed_6b = nn.Identity()
        self.inception.Mixed_6c = nn.Identity()
        self.inception.Mixed_6d = nn.Identity()
        self.inception.Mixed_6e = nn.Identity()
        self.inception.AuxLogits = nn.Identity()
        self.inception.Mixed_7a = nn.Identity()
        self.inception.Mixed_7b = nn.Identity()
        self.inception.Mixed_7c = nn.Identity()
        self.inception.fc = nn.Linear(288, num_class)

    def training_step(
        self, batch: tuple[Tensor, Tensor, Tensor], batch_idx: int
    ) -> Tensor:
        writer_id, image, _ = batch
        if self.gen_model_type == "diffusion":
            image = repeat(image, "b 1 h w -> b 3 h w")

        writer_id = writer_id.to(torch.long)

        outputs, _ = self.inception(self.__normalize(image))

        loss = nn.functional.cross_entropy(outputs, writer_id)
        self.train_accuracy.update(outputs, writer_id)

        self.log(
            "train/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return loss

    def validation_step(
        self, batch: tuple[Tensor, Tensor, Tensor], batch_idx: int
    ) -> Tensor:
        writer_id, image, _ = batch
        if self.gen_model_type == "diffusion":
            image = repeat(image, "b 1 h w -> b 3 h w")

        writer_id = writer_id.to(torch.long)

        with torch.inference_mode():
            outputs = self.inception(self.__normalize(image))

            loss = nn.functional.cross_entropy(outputs, writer_id)
            self.val_accuracy.update(outputs, writer_id)

        self.log(
            "val/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return loss

    def on_train_epoch_end(self) -> None:
        accuracy = self.train_accuracy.compute()

        self.log(
            "train/accuracy",
            accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        self.train_accuracy.reset()

    def on_validation_epoch_end(self) -> None:
        accuracy = self.val_accuracy.compute()

        self.log(
            "val/accuracy",
            accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=15, gamma=0.1
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }
