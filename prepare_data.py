import contextlib
import os
import sys
from argparse import ArgumentParser
from pathlib import Path

import yaml

from configs.settings import CONFIGS, DATASETS
from data import utils


def cli_main():
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
    if args.config in ("Diffusion", "RNN"):
        h5_file_path = Path(
            f"data/h5_dataset/train_val_iamondb_{args.config.lower()}.h5"
        )
        with contextlib.suppress(FileNotFoundError):
            os.remove(h5_file_path)

        kwargs_dataset = dict(
            config=config,
            img_height=config.get("img_height", 90),
            img_width=config.get("img_width", 1400),
            max_text_len=config.max_text_len,
            max_seq_len=config.max_seq_len,
            max_files=config.max_files,
        )
        dataset = DATASETS[args.config](**kwargs_dataset)
        utils.save_dataset(dataset.dataset, (h5_file_path, None))

    else:
        train_size = config.get("train_size", 0.85)
        val_size = 1.0 - train_size
        h5_file_path = Path("data/h5_dataset/train_iamdb.h5")
        json_file_path = Path("data/json_writer_ids/train_writer_ids.json")

        with contextlib.suppress(FileNotFoundError):
            os.remove(h5_file_path)
            os.remove(json_file_path)

        kwargs_dataset = dict(
            config=config,
            img_height=config.get("img_height", 90),
            img_width=config.get("img_width", 1400),
            max_text_len=config.max_text_len,
            max_files=train_size * config.max_files,
            dataset_type="train",
        )

        dataset = DATASETS[args.config](**kwargs_dataset)
        utils.save_dataset(
            dataset.dataset,
            (h5_file_path, json_file_path),
            latent=True,
            map_writer_ids=dataset.map_writer_id,
        )

        h5_file_path = Path("data/h5_dataset/val_iamdb.h5")
        json_file_path = Path("data/json_writer_ids/val_writer_ids.json")
        with contextlib.suppress(FileNotFoundError):
            os.remove(h5_file_path)
            os.remove(json_file_path)

        kwargs_dataset["max_files"] = val_size * config.max_files
        kwargs_dataset["dataset_type"] = "val"

        dataset = DATASETS[args.config](**kwargs_dataset)
        utils.save_dataset(
            dataset.dataset,
            (h5_file_path, json_file_path),
            latent=True,
            map_writer_ids=dataset.map_writer_id,
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
