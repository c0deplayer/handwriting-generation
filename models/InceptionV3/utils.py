from typing import Callable, Dict, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
from einops import repeat
from rich.progress import track
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from data.utils import NormalizeInverse

undo_normalize = NormalizeInverse((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))


class EarlyStopper:
    mode_dict = {"min": torch.lt, "max": torch.gt}

    def __init__(
        self,
        patience: int = 3,
        min_delta: float = 0.0,
        mode: Literal["min", "max"] = "min",
        *,
        verbose: bool = False,
    ):
        self.patience = patience
        self.mode = mode
        self.min_delta = (
            torch.Tensor([min_delta * 1])
            if self.monitor_metric == torch.gt
            else torch.tensor([min_delta * -1])
        )
        self.counter = 0
        torch_inf = torch.tensor(torch.inf)
        self.best_score = -torch_inf if mode == "max" else torch_inf
        self.verbose = verbose

    def early_stop(self, validation_metric: float) -> bool:
        current = torch.Tensor([validation_metric])
        if self.monitor_metric(current - self.min_delta, self.best_score):
            self.best_score = current
            self.counter = 0
        else:
            self.counter += 1

            if self.counter >= self.patience:
                if self.verbose:
                    print(
                        f"The validation loss has not improved for {self.counter} epochs. Stopping training."
                    )

                return True
            elif self.verbose:
                print(f"The validation loss has not improved for {self.counter} epochs")

        return False

    @property
    def monitor_metric(self) -> Callable:
        return self.mode_dict[self.mode]


def train_loop(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    num_epochs: int,
    *,
    train_config: Dict[str, Union[int, str]],
    device: torch.device,
    scheduler: Optional[LRScheduler] = None,
    early_stopper: Optional[EarlyStopper] = None,
) -> None:
    for epoch in range(num_epochs):
        train_loss, train_acc = train(
            model,
            train_loader,
            criterion,
            optimizer,
            train_config,
            device,
        )

        val_loss, val_acc = validation(
            model, val_loader, criterion, train_config, device
        )

        if scheduler is not None:
            scheduler.step()

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] Loss: {train_loss:.4f}, Accuracy: {train_acc * 100:.4f}, "
            f"Val loss: {val_loss:.4f}, Val accuracy: {val_acc * 100:.4f}"
        )

        if early_stopper is not None:
            if (
                val_acc < early_stopper.best_score
                and early_stopper.monitor_metric == torch.lt
            ) or (
                val_acc > early_stopper.best_score
                and early_stopper.monitor_metric == torch.gt
            ):
                torch.save(
                    model.state_dict(),
                    f"./model_checkpoints/InceptionV3/inception_v3_{train_config['model']}"
                    f"_style-{train_config['num_class']}_epoch-{epoch + 1}"
                    f"_acc-{train_acc:.4f}_val-acc-{val_acc:.4f}.pth",
                )

            if early_stopper.early_stop(val_acc):
                break


def validation(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    train_config: Dict[str, Union[int, str]],
    device: torch.device,
) -> Tuple[float, float]:
    val_loss, val_acc = 0.0, 0.0
    model.eval()

    for batch in track(val_loader):
        with torch.inference_mode():
            writer_id, image, _ = batch
            if train_config["model"] == "diffusion":
                image = repeat(image, "b 1 h w -> b 3 h w")
            else:
                image = undo_normalize(image)

            image = image.to(device)
            writer_id = writer_id.type(torch.LongTensor).to(device)

            outputs = model(image)
            loss = criterion(outputs, writer_id)

            val_loss += loss.item()
            val_acc += compute_accuracy(outputs, writer_id)

    val_loss /= len(val_loader)
    val_acc /= len(val_loader)

    return val_loss, val_acc


def train(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    train_config: Dict[str, Union[int, str]],
    device: torch.device,
) -> Tuple[float, float]:
    train_loss, train_acc = 0.0, 0.0
    model.train()

    for batch in track(train_loader):
        writer_id, image, _ = batch
        if train_config["model"] == "diffusion":
            image = repeat(image, "b 1 h w -> b 3 h w")
        else:
            image = undo_normalize(image)

        image = image.to(device)
        writer_id = writer_id.type(torch.LongTensor).to(device)

        optimizer.zero_grad()

        outputs, _ = model(image)
        loss = criterion(outputs, writer_id)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_acc += compute_accuracy(outputs, writer_id)

    train_loss /= len(train_loader)
    train_acc /= len(train_loader)

    return train_loss, train_acc


def compute_accuracy(predictions: Tensor, labels: Tensor) -> float:
    classes = torch.argmax(predictions, dim=1)
    return torch.mean((classes == labels).float()).item()
