import torch

from configs.config import BaseConfig
from configs.constants import CONFIGS


def load_config(config_file: str, model: str) -> BaseConfig:
    """Load the configuration from a YAML file.

    Args:
        config_file (str): Path to the configuration file.

    Returns:
        BaseConfig: The loaded configuration object.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        ValueError: If the configuration file is invalid.
    """
    try:
        config = CONFIGS[model].from_yaml_file(file=config_file)
        if isinstance(config, list):
            config = config[0]
        return config
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Configuration file {config_file} not found.") from e
    except Exception as e:
        raise ValueError(f"Error loading configuration file {config_file}.") from e


def get_device(*, return_device: bool = False) -> str | torch.device:
    """
    Get the best available device (CUDA, MPS, or CPU).

    Args:
        return_device (bool): Whether to return a torch.device object. Defaults to False.

    Returns:
        str | torch.device: The best available device as a string or torch.device object.
    """
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available() and torch.backends.mps.is_built()
        else "cpu"
    )

    return torch.device(device) if return_device else device
