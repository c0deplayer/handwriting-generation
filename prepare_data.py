import contextlib
import os
import sys
from argparse import ArgumentParser
from pathlib import Path

import yaml

from configs.config import ConfigLatentDiffusion
from configs.settings import CONFIGS, DATASETS
from data import utils


def cli_main():
    """
    Command-line interface for initializing and parsing arguments.

    Returns:
        Namespace: A namespace object containing parsed arguments.
    """

    parser = ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        help="Type of model",
        choices=["RNN", "Diffusion", "LatentDiffusion"],
        type=str,
        required=True,
    )
    parser.add_argument(
        "-cf",
        "--config-file",
        help="Filename for configs",
        type=str,
        default="base_gpu.yaml",
    )

    return parser.parse_args()


def prepare_data():
    """
    Prepares and saves datasets for training and validation based on the specified model configurations.

    The function identifies the model type, splits the dataset into training and validation sets,
    and saves them in HDF5 and JSON formats as required. It handles different configurations
    for models like LatentDiffusion, RNN, and Diffusion.

    Raises:
        FileNotFoundError: If specified dataset files do not exist and cannot be removed.
    """

    is_latent = isinstance(config, ConfigLatentDiffusion)
    train_size = config.get("train_size", 0.8)
    val_size = 1.0 - train_size
    kwargs_dataset = dict(
        config=config,
        img_height=config.get("img_height", 90),
        img_width=config.get("img_width", 1400),
        max_text_len=config.max_text_len,
        max_files=train_size * config.max_files,
        dataset_type="train",
    )
    h5_file_path_train = Path(
        f"./data/h5_dataset/train_{'iamondb' if args.config in ('Diffusion', 'RNN') else 'iamdb'}.h5"
    )
    h5_file_path_val = Path(
        f"./data/h5_dataset/val_{'iamondb' if args.config in ('Diffusion', 'RNN') else 'iamdb'}.h5"
    )

    if args.config in ("Diffusion", "RNN"):
        json_file_path_train = None
        json_file_path_val = None

        kwargs_dataset["max_seq_len"] = config.max_seq_len
    else:
        json_file_path_train = Path("data/json_writer_ids/train_writer_ids.json")
        json_file_path_val = Path("data/json_writer_ids/val_writer_ids.json")

    with contextlib.suppress(FileNotFoundError, TypeError):
        os.remove(h5_file_path_train)
        os.remove(h5_file_path_val)
        os.remove(json_file_path_train)
        os.remove(json_file_path_val)

    dataset = DATASETS[args.config](**kwargs_dataset)

    utils.save_dataset(
        dataset.dataset,
        (h5_file_path_train, json_file_path_train),
        is_latent=is_latent,
        map_writer_ids=dataset.map_writer_id if is_latent else None,
    )

    kwargs_dataset["max_files"] = val_size * config.max_files
    kwargs_dataset["dataset_type"] = "val"

    dataset = DATASETS[args.config](**kwargs_dataset)

    utils.save_dataset(
        dataset.dataset,
        (h5_file_path_val, json_file_path_val),
        is_latent=is_latent,
        map_writer_ids=dataset.map_writer_id if is_latent else None,
    )


if __name__ == "__main__":
    args = cli_main()

    if sys.version_info < (3, 8):
        raise SystemExit("Only Python 3.8 and above is supported")

    config_file = f"configs/{args.config}/{args.config_file}"

    config = CONFIGS[args.config].from_yaml_file(
        file=config_file, decoder=yaml.load, Loader=yaml.Loader
    )

    prepare_data()
