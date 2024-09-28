from enum import Enum
from pathlib import Path
from typing import Type

import lightning as L
import matplotlib.colors as plt_colors
from torch.utils.data import Dataset

from configs.config import (
    BaseConfig,
    ConfigConvNeXt,
    ConfigDiffusion,
    ConfigInception,
    ConfigLatentDiffusion,
)
from data.dataset import IAMDataset, IAMonDataset
from models.ConvNeXt.model import ConvNeXt
from models.Diffusion.model import DiffusionWrapper
from models.InceptionV3.model import InceptionV3
from models.LatentDiffusion.model import LatentDiffusionModel


class ModelType(Enum):
    DIFFUSION = "Diffusion"
    LATENT_DIFFUSION = "LatentDiffusion"
    INCEPTION = "Inception"
    CONVNEXT = "ConvNeXt"


MODELS: dict[str, Type[L.LightningModule]] = {
    ModelType.DIFFUSION.value: DiffusionWrapper,
    ModelType.LATENT_DIFFUSION.value: LatentDiffusionModel,
    ModelType.INCEPTION.value: InceptionV3,
    ModelType.CONVNEXT.value: ConvNeXt,
}

MODELS_SN: dict[str, str] = {
    ModelType.DIFFUSION.value: "DM",
    ModelType.LATENT_DIFFUSION.value: "LDM",
}

CONFIGS: dict[str, Type[BaseConfig]] = {
    ModelType.DIFFUSION.value: ConfigDiffusion,
    ModelType.LATENT_DIFFUSION.value: ConfigLatentDiffusion,
    ModelType.CONVNEXT.value: ConfigConvNeXt,
    ModelType.INCEPTION.value: ConfigInception,
}

DATASETS: dict[str, Type[Dataset]] = {
    ModelType.DIFFUSION.value: IAMonDataset,
    ModelType.LATENT_DIFFUSION.value: IAMDataset,
}

MODELS_APP: list[str] = [
    model.value
    for model in ModelType
    if model not in (ModelType.INCEPTION, ModelType.CONVNEXT)
]
COLORS: list[str] = list(plt_colors.CSS4_COLORS.keys())
CALCULATION_BASE_DIR: Path = Path("./calc_data")
GEN_DATASET_DIR: Path = Path("./gen_data")
