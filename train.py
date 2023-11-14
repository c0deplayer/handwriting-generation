import json
import os
import sys
from argparse import ArgumentParser
from pathlib import Path

import lightning.pytorch as pl
import neptune
import torch
import yaml
from lightning.pytorch.callbacks import (
    RichModelSummary,
    RichProgressBar,
)
from lightning.pytorch.loggers.neptune import NeptuneLogger

from configs.settings import MODELS, MODELS_SN, CONFIGS, DATASETS
from data.dataset import DataModule
from periodic_checkpoint import PeriodicCheckpoint


def cli_main():
    """
    Command-line interface for initializing and parsing arguments for model training,
    including configurations and logging options.

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

    return parser.parse_args()


def train_model():
    """
    Configures and initiates the training process for the selected model.

    This function handles the setup of the PyTorch Lightning Trainer, model initialization,
    data module preparation, and optional logging with Neptune. It supports different
    configurations for models like Diffusion, RNN, and LatentDiffusion.

    Raises:
        SystemExit: If the Python version is below 3.8.
    """

    global dataset
    trainer_params = dict(
        accelerator=config.device,
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

    if args.config == "Diffusion":
        model_params = dict(
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

        trainer_params.update(
            {
                "gradient_clip_val": config.clip_grad,
                "gradient_clip_algorithm": config.clip_algorithm,
                "max_steps": config.max_epochs,
            }
        )

        del trainer_params["max_epochs"]

    elif args.config == "RNN":
        model_params = dict(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_window=config.num_window,
            num_mixture=config.num_mixture,
            vocab_size=config.vocab_size,
            bias=config.bias,
            clip_grads=(config.lstm_clip, config.mdn_clip),
        )

    else:
        with open("data/json_writer_ids/train_writer_ids.json", mode="r") as fp:
            map_writer_id = json.load(fp)
            n_style_classes = len(map_writer_id)
            del map_writer_id

        model_params = dict(
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

    # * Only for university servers that have two GPUs *
    if config.device == "cuda" and args.remote:
        torch.set_float32_matmul_precision("high")
        trainer_params["devices"] = [0]
        # trainer_params["precision"] = "bf16-mixed"

    model = MODELS[args.config](**model_params)
    dataset = DATASETS[args.config]
    data_module = DataModule(config=config, dataset=dataset)

    if args.neptune:
        source_files = [f"{os.getcwd()}/main.py"] + [
            f"{os.getcwd()}/{str(path)}"
            for path in Path(f"models/{args.config}").glob("*.py")
        ]

        neptune_logger = NeptuneLogger(
            project="codeplayer/handwriting-generation",
            log_model_checkpoints=False,
            mode="debug",
            capture_stdout=False,
            source_files=source_files,
            dependencies="infer",
        )

        model_version = neptune.init_model_version(
            model=f"HAN-{MODELS_SN[args.config]}",
            project="codeplayer/handwriting-generation",
            mode="debug",
        )

        trainer_params["logger"] = neptune_logger

        neptune_logger.log_hyperparams(model_params)
        neptune_logger.log_model_summary(model=model, max_depth=-1)

        model_version["run/id"] = neptune_logger.run["sys/id"].fetch()
        model_version["run/url"] = neptune_logger.run.get_url()
        model_version["model/parameters"] = model_params
        model_version["model/config"].upload(
            f"configs/{args.config}/{args.config_file}"
        )

    trainer = pl.Trainer(**trainer_params)
    trainer.fit(model=model, datamodule=data_module)
    trainer.save_checkpoint(filepath=f"{config.checkpoint_path}/model.ckpt")

    if args.neptune:
        model_version["model/binary"].upload(f"{config.checkpoint_path}/model.ckpt")
        model_version.stop()


if __name__ == "__main__":
    args = cli_main()
    dataset = None

    if sys.version_info < (3, 8):
        raise SystemExit("Only Python 3.8 and above is supported")

    config_file = f"configs/{args.config}/{args.config_file}"

    config = CONFIGS[args.config].from_yaml_file(
        file=config_file, decoder=yaml.load, Loader=yaml.Loader
    )

    train_model()
