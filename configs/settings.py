import matplotlib.colors as plt_colors

from configs.config import ConfigDiffusion, ConfigLatentDiffusion, ConfigRNN
from data.dataset import IAMDataset, IAMonDataset
from models.Diffusion.model import DiffusionWrapper
from models.LatentDiffusion.model import LatentDiffusionModel

MODELS = {
    "Diffusion": DiffusionWrapper,
    "LatentDiffusion": LatentDiffusionModel,
}

MODELS_SN = {
    "Diffusion": "DM",
    "LatentDiffusion": "LDM",
}

CONFIGS = {
    "Diffusion": ConfigDiffusion,
    "LatentDiffusion": ConfigLatentDiffusion,
}

DATASETS = {
    "Diffusion": IAMonDataset,
    "LatentDiffusion": IAMDataset,
}

MODELS_APP = ["Diffusion", "LatentDiffusion"]
COLORS = list(plt_colors.CSS4_COLORS.keys())
CALCULATION_BASE_DIR = "./calc_data"
GEN_DATASET_DIR = "./gen_data"
