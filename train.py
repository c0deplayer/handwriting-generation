import json
import os
from argparse import ArgumentParser, Namespace
from typing import Any

import lightning as L
import neptune
import torch
from dotenv import load_dotenv
from lightning.pytorch.callbacks import RichModelSummary, RichProgressBar
from lightning.pytorch.loggers.neptune import NeptuneLogger
from neptune import ModelVersion

from configs.config import BaseConfig, ConfigDiffusion, ConfigLatentDiffusion
from configs.constants import DATASETS, MODELS, MODELS_SN
from data.dataset import DataModule
from periodic_checkpoint import PeriodicCheckpoint
from utils import train_utils
from utils.utils import load_config


def cli_main() -> Namespace:
    """Command-line interface for initializing and parsing arguments for model training.

    Returns:
        Namespace: A namespace object containing parsed arguments.
    """
    parser = ArgumentParser(description="Model Training CLI")
    model_group = parser.add_argument_group("Model Options")
    model_group.add_argument(
        "-m",
        "--model",
        help="Type of model",
        choices=["Diffusion", "LatentDiffusion"],
        type=str,
        required=True,
    )
    model_group.add_argument(
        "-c", "--config", help="Filename for configs", type=str, default="base_gpu.yaml"
    )

    env_group = parser.add_argument_group("Environment Options")
    env_group.add_argument(
        "-r",
        "--remote",
        help="Flag indicating whether the model will be trained on a server with dedicated GPUs, such as the A100",
        action="store_true",
    )
    env_group.add_argument(
        "-n", "--neptune", help="Flag for using NeptuneLogger", action="store_true"
    )
    env_group.add_argument(
        "--gpu",
        help="GPU/MPS (Apple Silicone) device to use for training",
        action="store_true",
    )

    return parser.parse_args()


def get_model_params(config: BaseConfig) -> dict[str, Any]:
    """Get the model parameters based on the configuration and arguments.

    Args:
        config (ConfigDiffusion | ConfigLatentDiffusion): The loaded configuration object.

    Returns:
        dict[str, Any]: A dictionary containing model parameters.
    """
    match config:
        case ConfigDiffusion():
            return dict(
                diffusion_params=dict(
                    num_layers=config.num_layers,
                    c1=config.channels,
                    c2=config.channels * 3 // 2,
                    c3=config.channels * 2,
                    drop_rate=config.drop_rate,
                    vocab_size=config.vocab_size,
                ),
                use_ema=config.use_ema,
            )
        case ConfigLatentDiffusion():
            with open("data/json_writer_ids/train_writer_ids.json", mode="r") as fp:
                map_writer_id = json.load(fp)
                n_style_classes = len(map_writer_id)

            return dict(
                unet_params=dict(
                    in_channels=config.channels,
                    out_channels=config.channels,
                    channels=config.emb_dim,
                    res_layers=config.res_layers,
                    vocab_size=config.vocab_size,
                    attention_levels=config.attention_levels,
                    channel_multipliers=config.channel_multipliers,
                    heads=config.n_heads,
                    d_cond=config.d_cond,
                    n_style_classes=n_style_classes,
                    dropout=config.drop_rate,
                    max_seq_len=config.max_text_len + 2,
                ),
                autoencoder_path=config.autoencoder_path,
                n_steps=config.n_steps,
                beta_start=config.beta_start,
                beta_end=config.beta_end,
                img_size=(config.img_height, config.img_width),
            )
        case _:
            raise ValueError("Unsupported config type")


def get_trainer_params(config: BaseConfig, args: Namespace) -> dict[str, Any]:
    """Get the trainer parameters based on the configuration and arguments.

    Args:
        config (ConfigDiffusion | ConfigLatentDiffusion): The loaded configuration object.
        args (Namespace): Parsed command-line arguments.

    Returns:
        dict[str, Any]: A dictionary containing trainer parameters.
    """
    trainer_params = dict(
        accelerator=train_utils.get_device() if args.gpu else "cpu",
        default_root_dir=config.checkpoint_path,
        deterministic=True,
        detect_anomaly=False,
        max_epochs=config.max_epochs,
        callbacks=[
            RichModelSummary(max_depth=5),
            RichProgressBar(refresh_rate=1, leave=True),
            PeriodicCheckpoint(
                every_steps=10000,
                dirpath=config.checkpoint_path,
                filename="model-epoch{epoch:02d}-train_loss{train/loss:.2f}-val_loss{val/loss:.2f}",
            ),
        ],
    )

    if isinstance(config, ConfigDiffusion):
        trainer_params |= {
            "gradient_clip_val": config.clip_grad,
            "gradient_clip_algorithm": config.clip_algorithm,
            "max_steps": config.max_epochs,
        }

        del trainer_params["max_epochs"]

    if args.gpu and args.remote:
        torch.set_float32_matmul_precision("high")
        trainer_params |= {"devices": [0]}

    return trainer_params


def setup_neptune_logger(
    args: Namespace,
    model_params: dict[str, Any],
    config_file: str,
    model: Any,
    neptune_token: str,
) -> tuple[NeptuneLogger, ModelVersion]:
    """Set up the Neptune logger for logging model training.

    Args:
        args (Namespace): Parsed command-line arguments.
        model_params (dict[str, Any]): Model parameters.
        config_file (str): Path to the configuration file.
        model (Any): The model to be logged.
        neptune_token (str): Neptune API token.

    Returns:
        tuple[NeptuneLogger, ModelVersion]: The configured Neptune logger and model version.
    """
    neptune_logger = NeptuneLogger(
        project="codeplayer/handwriting-generation",
        api_token=neptune_token,
        log_model_checkpoints=False,
        source_files=["**/*.py", config_file],
    )

    model_version = neptune.init_model_version(
        model=f"HAN-{MODELS_SN[args.model]}",
        api_token=neptune_token,
        project="codeplayer/handwriting-generation",
    )

    neptune_logger.log_hyperparams(model_params)
    neptune_logger.log_model_summary(model=model, max_depth=-1)

    model_version["run/id"] = neptune_logger.run["sys/id"].fetch()
    model_version["run/url"] = neptune_logger.run.get_url()
    model_version["model/parameters"] = model_params
    model_version["model/config"].upload(config_file)

    return neptune_logger, model_version


def train_model(
    config: BaseConfig,
    config_file: str,
    args: Namespace,
    neptune_token: str,
) -> None:
    """Configures and initiates the training process for the selected model.

    This function handles the setup of the PyTorch Lightning Trainer, model initialization,
    data module preparation, and optional logging with Neptune. It supports different
    configurations for models like Diffusion, RNN, and LatentDiffusion.

    Args:
        config (BaseConfig): The loaded configuration object.
        config_file (str): Path to the configuration file.
        args (Namespace): Parsed command-line arguments.
        neptune_token (str): Neptune API token.
    """
    model_version = None
    model_params = get_model_params(config)
    trainer_params = get_trainer_params(config, args)

    model = MODELS[args.model](**model_params)
    dataset = DATASETS[args.model]
    data_module = DataModule(config=config, dataset=dataset, use_gpu=args.gpu)

    if args.neptune:
        neptune_logger, model_version = setup_neptune_logger(
            args, model_params, config_file, model, neptune_token
        )
        trainer_params["logger"] = neptune_logger

    trainer = L.Trainer(**trainer_params)
    trainer.fit(model=model, datamodule=data_module)
    trainer.save_checkpoint(filepath=f"{config.checkpoint_path}/model.ckpt")

    if model_version is not None:
        model_version["model/binary"].upload(f"{config.checkpoint_path}/model.ckpt")
        model_version.stop()


if __name__ == "__main__":
    args = cli_main()
    load_dotenv()
    neptune_token = os.environ["NEPTUNE_API_TOKEN"]

    config_file = f"configs/{args.model}/{args.config}"
    config = load_config(config_file, args.model)

    train_model(config, config_file, args, neptune_token)
