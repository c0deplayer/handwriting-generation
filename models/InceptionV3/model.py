import lightning as lp
import torch
from einops import repeat
from lightning.pytorch.utilities.types import (
    OptimizerLRScheduler,
)
from torch import Tensor, nn
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import StepLR
from torchmetrics.classification import MulticlassAccuracy
from torchvision import models
from torchvision.models import Inception_V3_Weights
from torchvision.transforms import v2


class InceptionV3(lp.LightningModule):
    """InceptionV3 model class for training and validation using PyTorch Lightning.

    Args:
        num_classes (int): Number of output classes for the classification task.
        gen_model_type (str): Type of generative model, e.g., 'diffusion'.

    Attributes:
        gen_model_type (str): Type of generative model.
        normalize (nn.Module): Normalization layer based on the model type.
        inception (nn.Module): InceptionV3 model from torchvision.
        train_accuracy (MulticlassAccuracy): Metric for tracking training accuracy.
        val_accuracy (MulticlassAccuracy): Metric for tracking validation accuracy.

    """

    def __init__(self, num_classes: int, gen_model_type: str) -> None:
        super().__init__()

        self.gen_model_type = gen_model_type
        self.normalize = (
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            if gen_model_type == "diffusion"
            else nn.Identity()
        )

        self.inception = models.inception_v3(
            weights=Inception_V3_Weights.DEFAULT,
        )
        self.train_accuracy = MulticlassAccuracy(
            num_classes=num_classes,
            average="macro",
        )
        self.val_accuracy = MulticlassAccuracy(
            num_classes=num_classes,
            average="macro",
        )

        self._freeze_and_modify_layers(num_classes)

    def _freeze_and_modify_layers(self, num_classes: int) -> None:
        """Freeze certain layers and modify the classifier layer.

        Args:
            num_classes (int): Number of output classes for the classifier.

        """
        for name, param in self.inception.named_parameters():
            if "Mixed_5" not in name:
                param.requires_grad = False

        for layer in [
            "Mixed_6a",
            "Mixed_6b",
            "Mixed_6c",
            "Mixed_6d",
            "Mixed_6e",
            "AuxLogits",
            "Mixed_7a",
            "Mixed_7b",
            "Mixed_7c",
        ]:
            setattr(self.inception, layer, nn.Identity())

        self.inception.fc = nn.Linear(288, num_classes)

    def training_step(
        self,
        batch: tuple[Tensor, Tensor, Tensor],
        batch_idx: int,
    ) -> Tensor:
        """Perform a single training step.

        Args:
            batch (tuple[Tensor, Tensor, Tensor]): Batch of data containing
                writer IDs, images, and labels.
            batch_idx (int): Index of the batch.

        Returns:
            Tensor: Computed loss for the batch.

        """
        writer_id, image, _ = batch
        if self.gen_model_type == "diffusion":
            image = repeat(image, "b 1 h w -> b 3 h w")

        writer_id = writer_id.to(torch.long)
        outputs, _ = self.inception(self.normalize(image))
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
        self,
        batch: tuple[Tensor, Tensor, Tensor],
        batch_idx: int,
    ) -> Tensor:
        """Perform a single validation step.

        Args:
            batch (tuple[Tensor, Tensor, Tensor]): Batch of data containing
                writer IDs, images, and labels.
            batch_idx (int): Index of the batch.

        Returns:
            Tensor: Computed loss for the batch.

        """
        writer_id, image, _ = batch
        if self.gen_model_type == "diffusion":
            image = repeat(image, "b 1 h w -> b 3 h w")

        writer_id = writer_id.to(torch.long)
        with torch.inference_mode():
            outputs = self.inception(self.normalize(image))
            loss = nn.functional.cross_entropy(outputs, writer_id)
            self.val_accuracy.update(outputs, writer_id)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self) -> None:
        """Actions to perform at the end of each training epoch.

        Logs the training accuracy.
        """
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
        """Actions to perform at the end of each validation epoch.

        Logs the validation accuracy.
        """
        accuracy = self.val_accuracy.compute()
        self.log(
            "val/accuracy",
            accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.val_accuracy.reset()

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """Configure the optimizer and learning rate scheduler.

        Returns:
            OptimizerLRScheduler: Dictionary containing the optimizer and
            learning rate scheduler.

        """
        optimizer = AdamW(self.parameters(), lr=1e-3)
        scheduler = StepLR(
            optimizer,
            step_size=15,
            gamma=0.1,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }
