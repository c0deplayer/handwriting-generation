import os
from argparse import ArgumentParser, Namespace
from multiprocessing import Pool
from pathlib import Path
from typing import Any

from configs.config import BaseConfig, ConfigDiffusion, ConfigLatentDiffusion
from configs.constants import DATASETS
from utils import data_utils
from utils.utils import load_config


def cli_main() -> Namespace:
    """Command-line interface for initializing and parsing arguments.

    Returns:
        Namespace: A namespace object containing parsed arguments.
    """
    parser = ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        help="Type of model",
        choices=["Diffusion", "LatentDiffusion"],
        type=str,
        required=True,
    )
    parser.add_argument(
        "-c",
        "--config",
        help="Filename for configs",
        type=str,
        default="base_gpu.yaml",
    )

    return parser.parse_args()


def __parallel_save_dataset(args: tuple) -> None:
    """Parallel function to save dataset.

    Args:
        args (tuple): Arguments for the save_dataset function.
    """
    dataset, file_paths, is_latent, map_writer_ids = args
    data_utils.save_dataset(
        dataset, file_paths, is_latent=is_latent, map_writer_ids=map_writer_ids
    )


def __create_dataset(args: tuple) -> Any:
    """Create a dataset based on the provided arguments.

    Args:
        args (tuple): Arguments for creating the dataset.

    Returns:
        Any: The created dataset.
    """
    dataset_class, kwargs = args

    return DATASETS[dataset_class](**kwargs)


def prepare_data(config: BaseConfig, args: Namespace) -> None:
    """Prepares and saves datasets for training and validation based on the specified model configurations.

    Args:
        config (BaseConfig): The loaded configuration object.
        args (Namespace): Parsed command-line arguments.

    Raises:
        FileNotFoundError: If specified dataset files do not exist and cannot be removed.
    """
    train_size = config.get("train_size", 0.8)
    val_size = 1.0 - train_size

    kwargs_dataset = {
        "config": config,
        "img_height": config.get("img_height", 90),
        "img_width": config.get("img_width", 1400),
        "max_text_len": config.max_text_len,
        "max_files": train_size * config.max_files,
        "use_gpu": True,
        "dataset_type": "train",
    }

    Path("./data/h5_dataset").mkdir(parents=True, exist_ok=True)

    h5_file_path_train, h5_file_path_val, json_file_path_train, json_file_path_val = (
        __get_file_paths(args.model)
    )

    if isinstance(config, ConfigDiffusion):
        kwargs_dataset |= {"max_seq_len": config.max_seq_len}

    data_utils.remove_existing_files([
        h5_file_path_train,
        h5_file_path_val,
        json_file_path_train,
        json_file_path_val,
    ])

    tasks = [
        (args.model, kwargs_dataset),
        (
            args.model,
            {
                **kwargs_dataset,
                "max_files": val_size * config.max_files,
                "dataset_type": "val",
            },
        ),
    ]

    with Pool(processes=os.cpu_count()) as pool:
        train_dataset, val_dataset = pool.map(__create_dataset, tasks)

    tasks = [
        (
            train_dataset.dataset,
            (h5_file_path_train, json_file_path_train),
            isinstance(config, ConfigLatentDiffusion),
            train_dataset.map_writer_id,
        ),
        (
            val_dataset.dataset,
            (h5_file_path_val, json_file_path_val),
            isinstance(config, ConfigLatentDiffusion),
            val_dataset.map_writer_id,
        ),
    ]

    with Pool(processes=os.cpu_count()) as pool:
        pool.map(__parallel_save_dataset, tasks)


def __get_file_paths(model: str) -> tuple[Path, Path, Path, Path]:
    """Get file paths for HDF5 and JSON files based on the model type.

    Args:
        model (str): The type of model.

    Returns:
        tuple[Path, Path, Path, Path]: Paths for train HDF5, val HDF5, train JSON, and val JSON files.
    """
    h5_file_path_train = Path(
        f"./data/h5_dataset/train_{"iamondb" if model == "Diffusion" else "iamdb"}.h5"
    ).resolve()
    h5_file_path_val = Path(
        f"./data/h5_dataset/val_{"iamondb" if model == "Diffusion" else "iamdb"}.h5"
    ).resolve()
    json_file_path_train = Path(
        f"./data/json_writer_ids/train_writer_ids_{"iamondb" if model == "Diffusion" else "iamdb"}.json"
    ).resolve()
    json_file_path_val = Path(
        f"./data/json_writer_ids/val_writer_ids_{"iamondb" if model == "Diffusion" else "iamdb"}.json"
    ).resolve()

    return (
        h5_file_path_train,
        h5_file_path_val,
        json_file_path_train,
        json_file_path_val,
    )


if __name__ == "__main__":
    args = cli_main()

    config_file = f"configs/{args.model}/{args.config}"
    config = load_config(config_file, args.model)

    prepare_data(config, args)
