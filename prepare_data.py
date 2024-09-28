import os
from argparse import ArgumentParser, Namespace
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from torch.utils.data.dataloader import Dataset

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
        dataset,
        file_paths,
        is_latent=is_latent,
        map_writer_ids=map_writer_ids,
    )


def __create_dataset(args: tuple) -> Dataset:
    """Create a dataset based on the provided arguments.

    Args:
        args (tuple): Arguments for creating the dataset.

    Returns:
        Dataset: The created dataset.

    """
    dataset_class, kwargs = args
    return DATASETS[dataset_class](**kwargs)


def prepare_data(config: BaseConfig, args: Namespace) -> None:
    """Prepare and save datasets for training and validation based on the specified model configurations.

    Args:
        config (BaseConfig): The loaded configuration object.
        args (Namespace): Parsed command-line arguments.

    """
    train_size = config.get("train_size", 0.8)
    val_size = 1.0 - train_size

    kwargs_dataset = {
        "config": config,
        "img_height": config.get("img_height", 90),
        "img_width": config.get("img_width", 1400),
        "max_text_len": config.max_text_len,
        "max_files": int(train_size * config.max_files),
        "use_gpu": True,
        "dataset_type": "train",
    }

    Path("./data/h5_dataset").mkdir(parents=True, exist_ok=True)

    (
        h5_file_path_train,
        h5_file_path_val,
        json_file_path_train,
        json_file_path_val,
    ) = __get_file_paths(args.model)

    if isinstance(config, ConfigDiffusion):
        kwargs_dataset["max_seq_len"] = config.max_seq_len

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

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        train_dataset, val_dataset = executor.map(__create_dataset, tasks)

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

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        executor.map(__parallel_save_dataset, tasks)


def __get_file_paths(model: str) -> tuple[Path, Path, Path, Path]:
    """Get file paths for HDF5 and JSON files based on the model type.

    Args:
        model (str): The type of model.

    Returns:
        tuple[Path, Path, Path, Path]: Paths for train HDF5, val HDF5,
                                       train JSON, and val JSON files.

    """
    suffix = "iamondb" if model == "Diffusion" else "iamdb"
    return (
        Path(f"./data/h5_dataset/train_{suffix}.h5").resolve(),
        Path(f"./data/h5_dataset/val_{suffix}.h5").resolve(),
        Path(
            f"./data/json_writer_ids/train_writer_ids_{suffix}.json",
        ).resolve(),
        Path(f"./data/json_writer_ids/val_writer_ids_{suffix}.json").resolve(),
    )


if __name__ == "__main__":
    args = cli_main()

    config_file = f"configs/{args.model}/{args.config}"
    config = load_config(config_file, args.model)

    prepare_data(config, args)
