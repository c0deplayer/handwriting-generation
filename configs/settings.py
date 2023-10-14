from configs.config import ConfigDiffusion, ConfigRNN, ConfigLatentDiffusion
from data.dataset import IAMonDataset, IAMDataset
from models.Diffusion.model import DiffusionWrapper
from models.LatentDiffusion.model import LatentDiffusionModel
from models.RNN.model import RNNModel

MODELS = {
    "Diffusion": DiffusionWrapper,
    "RNN": RNNModel,
    "LatentDiffusion": LatentDiffusionModel,
}

MODELS_SN = {
    "Diffusion": "DM",
    "RNN": "RNN",
    "LatentDiffusion": "LDM",
}

CONFIGS = {
    "Diffusion": ConfigDiffusion,
    "RNN": ConfigRNN,
    "LatentDiffusion": ConfigLatentDiffusion,
}

DATASETS = {
    "Diffusion": IAMonDataset,
    "RNN": IAMonDataset,
    "LatentDiffusion": IAMDataset,
}
