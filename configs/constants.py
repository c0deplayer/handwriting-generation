from enum import Enum
from pathlib import Path
from typing import Type

import lightning as L
import matplotlib.colors as plt_colors
from torch.utils.data import Dataset

from configs.config import BaseConfig, ConfigDiffusion, ConfigLatentDiffusion
from data.dataset import IAMDataset, IAMonDataset
from models.Diffusion.model import DiffusionWrapper
from models.LatentDiffusion.model import LatentDiffusionModel


class ModelType(Enum):
    DIFFUSION = "Diffusion"
    LATENT_DIFFUSION = "LatentDiffusion"


MODELS: dict[str, Type[L.LightningModule]] = {
    ModelType.DIFFUSION.value: DiffusionWrapper,
    ModelType.LATENT_DIFFUSION.value: LatentDiffusionModel,
}

MODELS_SN: dict[str, str] = {
    ModelType.DIFFUSION.value: "DM",
    ModelType.LATENT_DIFFUSION.value: "LDM",
}

CONFIGS: dict[str, Type[BaseConfig]] = {
    ModelType.DIFFUSION.value: ConfigDiffusion,
    ModelType.LATENT_DIFFUSION.value: ConfigLatentDiffusion,
}

DATASETS: dict[str, Type[Dataset]] = {
    ModelType.DIFFUSION.value: IAMonDataset,
    ModelType.LATENT_DIFFUSION.value: IAMDataset,
}

MODELS_APP: list[str] = [model.value for model in ModelType]
COLORS: list[str] = list(plt_colors.CSS4_COLORS.keys())
CALCULATION_BASE_DIR: Path = Path("./calc_data")
GEN_DATASET_DIR: Path = Path("./gen_data")
