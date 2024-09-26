import json
import os
from argparse import ArgumentParser, Namespace
from typing import Any, Optional

import lightning as L
import neptune
import torch
from dotenv import load_dotenv
from lightning.pytorch.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
)
from lightning.pytorch.loggers.neptune import NeptuneLogger
from neptune import ModelVersion

from configs.config import (
    BaseConfig,
    ConfigConvNeXt,
    ConfigDiffusion,
    ConfigInception,
    ConfigLatentDiffusion,
)
from configs.constants import DATASETS, MODELS, MODELS_SN
from data.dataset import DataModule
from utils import utils
from utils.callbacks import PeriodicCheckpoint
from utils.utils import load_config


def cli_main() -> Namespace:
    """Command-line interface for initializing and parsing arguments for model training.

    Returns:
        Namespace: A namespace object containing parsed arguments.
    """
    parser = ArgumentParser(description="Model Training CLI")
    parser.add_argument(
        "-m",
        "--model",
        help="Model to train. Info: Inception and ConvNext are trained for style classification, "
        "while Diffusion and LatentDiffusion are trained for text-to-image synthesis.",
        choices=["Diffusion", "LatentDiffusion", "Inception", "ConvNeXt"],
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
    parser.add_argument(
        "-r",
        "--remote",
        help="Flag indicating whether the model will be trained on a server with dedicated GPUs, such as the A100",
        action="store_true",
    )
    parser.add_argument(
        "-n",
        "--neptune",
        help="Flag for using NeptuneLogger",
        action="store_true",
    )
    parser.add_argument(
        "--gpu",
        help="GPU/MPS (Apple Silicone) device to use for training",
        action="store_true",
    )

    return parser.parse_args()


def get_model_params(config: BaseConfig) -> dict[str, Any]:
    """Get the model parameters based on the configuration and arguments.

    Args:
        config (BaseConfig): The loaded configuration object.

    Returns:
        dict[str, Any]: A dictionary containing model parameters.
    """
    match config:
        case ConfigDiffusion():
            return {
                "diffusion_params": {
                    "num_layers": config.num_layers,
                    "c1": config.channels,
                    "c2": config.channels * 3 // 2,
                    "c3": config.channels * 2,
                    "drop_rate": config.drop_rate,
                    "vocab_size": config.vocab_size,
                },
                "use_ema": config.use_ema,
            }
        case ConfigLatentDiffusion():
            with open(
                "data/json_writer_ids/train_writer_ids.json", mode="r"
            ) as fp:
                map_writer_id = json.load(fp)
                n_style_classes = len(map_writer_id)

            return {
                "unet_params": {
                    "in_channels": config.channels,
                    "out_channels": config.channels,
                    "channels": config.emb_dim,
                    "res_layers": config.res_layers,
                    "vocab_size": config.vocab_size,
                    "attention_levels": config.attention_levels,
                    "channel_multipliers": config.channel_multipliers,
                    "heads": config.n_heads,
                    "d_cond": config.d_cond,
                    "n_style_classes": n_style_classes,
                    "dropout": config.drop_rate,
                    "max_seq_len": config.max_text_len + 2,
                },
                "autoencoder_path": config.autoencoder_path,
                "n_steps": config.n_steps,
                "beta_start": config.beta_start,
                "beta_end": config.beta_end,
                "img_size": (config.img_height, config.img_width),
            }
        case ConfigConvNeXt() | ConfigInception():
            suffix = (
                "iamondb" if config.gen_model_type == "Diffusion" else "iamdb"
            )
            with open(
                f"data/json_writer_ids/train_writer_ids_{suffix}.json", mode="r"
            ) as fp:
                map_writer_id = json.load(fp)
                num_class = len(map_writer_id)

            return {
                "num_class": num_class,
                "gen_model_type": config.gen_model_type.lower(),
            }
        case _:
            raise ValueError("Unsupported config type")


def get_trainer_params(config: BaseConfig, args: Namespace) -> dict[str, Any]:
    """Get the trainer parameters based on the configuration and arguments.

    Args:
        config (BaseConfig): The loaded configuration object.
        args (Namespace): Parsed command-line arguments.

    Returns:
        dict[str, Any]: A dictionary containing trainer parameters.
    """
    trainer_params = {
        "accelerator": utils.get_device() if args.gpu else "cpu",
        "default_root_dir": config.checkpoint_path,
        "deterministic": True,
        "detect_anomaly": False,
        "max_epochs": config.max_epochs,
        "callbacks": [
            RichModelSummary(max_depth=5),
            RichProgressBar(refresh_rate=1, leave=True),
        ],
    }

    match config:
        case ConfigDiffusion():
            trainer_params.update({
                "gradient_clip_val": config.clip_grad,
                "gradient_clip_algorithm": config.clip_algorithm,
                "max_steps": config.max_epochs,
            })
            trainer_params["callbacks"].append(
                PeriodicCheckpoint(
                    every_steps=10000,
                    dirpath=config.checkpoint_path,
                    filename="model-epoch{epoch:02d}-train_loss{train/loss:.2f}-val_loss{val/loss:.2f}",
                )
            )
            del trainer_params["max_epochs"]
        case ConfigLatentDiffusion():
            trainer_params["callbacks"].append(
                PeriodicCheckpoint(
                    every_steps=1000,
                    dirpath=config.checkpoint_path,
                    filename="model-epoch{epoch:02d}-train_loss{train/loss:.2f}-val_loss{val/loss:.2f}",
                )
            )
        case ConfigConvNeXt() | ConfigInception():
            trainer_params["callbacks"].extend([
                ModelCheckpoint(
                    dirpath=config.checkpoint_path,
                    filename="model-epoch{epoch:02d}-train_loss{train/loss:.2f}-val_loss{val/loss:.2f}",
                    mode="max",
                    monitor="val/accuracy",
                    save_weights_only=True,
                    auto_insert_metric_name=False,
                ),
                EarlyStopping(
                    monitor="val/accuracy",
                    mode="max",
                    patience=8,
                    verbose=True,
                    check_on_train_epoch_end=False,
                ),
            ])

    if args.gpu and args.remote:
        torch.set_float32_matmul_precision("high")
        trainer_params["devices"] = [0]

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
        mode="debug",
    )

    model_version = neptune.init_model_version(
        model=f"HAN-{MODELS_SN[args.model]}",
        api_token=neptune_token,
        project="codeplayer/handwriting-generation",
        mode="debug",
    )

    neptune_logger.log_hyperparams(model_params)
    neptune_logger.log_model_summary(model=model, max_depth=-1)

    model_version["run/id"] = neptune_logger.run["sys/id"].fetch()
    model_version["run/url"] = neptune_logger.run.get_url()
    model_version["model/parameters"] = model_params
    model_version["model/config"].upload(config_file)

    return neptune_logger, model_version


def train(
    config: BaseConfig,
    config_file: str,
    args: Namespace,
    neptune_token: Optional[str] = None,
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
    L.seed_everything(seed=42)

    model_version = None
    model_params = get_model_params(config)
    trainer_params = get_trainer_params(config, args)

    model = MODELS[args.model](**model_params)

    if hasattr(config, "gen_model_type") and config.gen_model_type:
        args.model = config.gen_model_type

    dataset = DATASETS[args.model]
    data_module = DataModule(config=config, dataset=dataset, use_gpu=args.gpu)

    if (
        args.neptune
        and neptune_token
        and isinstance(config, (ConfigDiffusion, ConfigLatentDiffusion))
    ):
        neptune_logger, model_version = setup_neptune_logger(
            args, model_params, config_file, model, neptune_token
        )
        trainer_params["logger"] = neptune_logger

    trainer = L.Trainer(**trainer_params)
    trainer.fit(model=model, datamodule=data_module)
    trainer.save_checkpoint(filepath=f"{config.checkpoint_path}/model.ckpt")

    if model_version:
        model_version["model/binary"].upload(
            f"{config.checkpoint_path}/model.ckpt"
        )
        model_version.stop()


if __name__ == "__main__":
    args = cli_main()

    load_dotenv()
    neptune_token = os.getenv("NEPTUNE_API_TOKEN")

    config_file = f"configs/{args.model}/{args.config}"
    config = load_config(config_file, args.model)

    train(config, config_file, args, neptune_token)
