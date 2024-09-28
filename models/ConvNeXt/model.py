import lightning as lp
import torch
from einops import repeat
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import Tensor, nn
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import StepLR
from torchmetrics.classification import MulticlassAccuracy
from torchvision import models
from torchvision.models import ConvNeXt_Tiny_Weights
from torchvision.transforms import v2


class ConvNeXt(lp.LightningModule):
    """ConvNeXt model class for training and validation using PyTorch Lightning.

    Args:
        num_classes (int): Number of output classes for the classification task.
        gen_model_type (str): Type of generative model, e.g., 'diffusion'.

    Attributes:
        gen_model_type (str): Type of generative model.
        lr (float): Learning rate for the optimizer.
        normalize (nn.Module): Normalization layer based on the model type.
        convnext (nn.Module): ConvNeXt model from torchvision.
        train_accuracy (MulticlassAccuracy): Metric for tracking training accuracy.
        val_accuracy (MulticlassAccuracy): Metric for tracking validation accuracy.

    """

    def __init__(self, num_classes: int, gen_model_type: str) -> None:
        """Initialize the ConvNeXt model.

        Args:
            num_classes (int): Number of output classes for the classification task.
            gen_model_type (str): Type of generative model, e.g., 'diffusion'.

        """
        super().__init__()

        self.gen_model_type = gen_model_type
        self.lr = 1e-3 if gen_model_type == "diffusion" else 1e-4

        self.normalize = (
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            if gen_model_type == "diffusion"
            else nn.Identity()
        )

        self.convnext = models.convnext_tiny(
            weights=ConvNeXt_Tiny_Weights.DEFAULT,
        )
        self.train_accuracy = MulticlassAccuracy(
            num_classes=num_classes,
            average="macro",
        )
        self.val_accuracy = MulticlassAccuracy(
            num_classes=num_classes,
            average="macro",
        )

        self.__freeze_and_modify_layers(num_classes)

    def __freeze_and_modify_layers(self, num_classes: int) -> None:
        """Freeze certain layers and modify the classifier layer.

        Args:
            num_classes (int): Number of output classes for the classifier.

        """
        for name, param in self.convnext.named_parameters():
            if "6" not in name:
                param.requires_grad = False

        self.convnext.features[7] = nn.Identity()
        self.convnext.classifier[-1] = nn.Linear(
            self.convnext.features[6][-1].out_channels,
            num_classes,
        )

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
        outputs = self.convnext(self.normalize(image))

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
        outputs = self.convnext(self.normalize(image))

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
        optimizer = AdamW(self.parameters(), lr=self.lr)
        scheduler = StepLR(optimizer, step_size=15, gamma=0.1)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }
