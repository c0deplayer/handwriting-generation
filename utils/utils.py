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
