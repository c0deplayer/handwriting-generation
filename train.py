import os
import sys
from argparse import ArgumentParser

import lightning.pytorch as pl
import numpy as np
import torch
import yaml
from lightning.pytorch.callbacks import (
    RichModelSummary,
    RichProgressBar,
    EarlyStopping,
    ModelCheckpoint,
)
from lightning.pytorch.plugins import MixedPrecisionPlugin

from configs.config import ConfigDiffusion, ConfigRNN, ConfigLatentDiffusion
from data.dataset import DataModule, IAMDataset, IAMonDataset
from models.Diffusion.model import DiffusionModel
from models.LatentDiffusion.model import LatentDiffusionModel
from models.RNN.model import RNNModel

MODELS = {
    "Diffusion": DiffusionModel,
    "RNN": RNNModel,
    "LatentDiffusion": LatentDiffusionModel,
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


def set_random_seed(
    seed: int = 42,
    precision: int = 10,
    deterministic: bool = False,
) -> None:
    # random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_printoptions(precision=precision)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True  # noqa
        torch.backends.cudnn.benchmark = False  # noqa


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

    parser.add_argument(
        "-r",
        "--remote",
        help="Flag for training on remote server",
        action="store_true",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = cli_main()

    if sys.version_info < (3, 8):
        raise SystemExit("Only Python 3.8 and above is supported")

    config_file = f"configs/{args.config}/{args.config_file}"

    config = CONFIGS[args.config].from_yaml_file(
        file=config_file, decoder=yaml.load, Loader=yaml.Loader
    )

    kwargs_trainer = dict(
        accelerator=config.device,
        default_root_dir=config.checkpoint_path,
        max_epochs=config.max_epochs,
        callbacks=[
            RichModelSummary(max_depth=3),
            RichProgressBar(refresh_rate=1),
            EarlyStopping(monitor="val_loss", patience=25),
            ModelCheckpoint(
                dirpath=config.checkpoint_path,
                monitor="val_loss",
                filename="{epoch}-{loss:.2f}-{val_loss:.2f}",
                save_top_k=5,
                # verbose=True,
            ),
        ],
    )

    if args.config == "Diffusion":
        kwargs_model = dict(
            num_layers=config.num_layers,
            c1=config.channels,
            c2=config.channels * 3 // 2,
            c3=config.channels * 2,
            drop_rate=config.drop_rate,
            vocab_size=config.vocab_size,
        )

        kwargs_trainer.update(
            {
                "gradient_clip_val": config.clip_grad,
                "gradient_clip_algorithm": config.clip_algorithm,
            }
        )

    elif args.config == "RNN":
        kwargs_model = dict(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_window=config.num_window,
            num_mixture=config.num_mixture,
            vocab_size=config.vocab_size,
            bias=config.bias,
            clip_grads=(config.lstm_clip, config.mdn_clip),
        )

        # * The diffusion model requires at least 32-bit precision, otherwise it generates NaN values. *
        if config.device == "cuda":
            plugins = [MixedPrecisionPlugin(precision="16-mixed", device=config.device)]
            kwargs_trainer["plugins"] = plugins
    else:
        kwargs_model = dict(
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
                n_style_classes=340,  # TODO: This should not be hardcoded !
                dropout=config.drop_rate,
                max_seq_len=config.max_text_len,
            ),
            autoencoder_path=config.autoencoder_path,
            n_steps=config.n_steps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            img_size=(config.img_height, config.img_width),
        )

    # * Only for university servers that have two GPUs *
    if config.device == "cuda" and args.remote:
        # * Only for A100 GPU on university server *
        # torch.set_float32_matmul_precision("medium")

        kwargs_trainer["devices"] = [1]

    set_random_seed()
    model = MODELS[args.config](**kwargs_model)
    dataset = DATASETS[args.config]
    data_module = DataModule(config=config, dataset=dataset)
    trainer = pl.Trainer(**kwargs_trainer)

    trainer.fit(model=model, datamodule=data_module)
    trainer.save_checkpoint(filepath=f"{config.checkpoint_path}/model.ckpt")

    # model = HandwritingSynthesisDiffusion.load_from_checkpoint(
    #     checkpoint_path=f"{config.checkpoint_path}/model.ckpt",
    #     # map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    #     map_location=torch.device("cpu"),
    # )
    #
    # model.eval()
    #
    # style_path = Path(f"assets/{os.listdir('assets')[0]}")
    #
    # model.generate("Handwriting Synthesis ", style_path)
