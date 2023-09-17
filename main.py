import contextlib
import os
import random
import sys
from argparse import ArgumentParser
from pathlib import Path

import lightning.pytorch as pl
import torch
import yaml
from lightning.pytorch.callbacks import (
    RichModelSummary,
    RichProgressBar,
    ModelCheckpoint,
)

from configs.config import ConfigDiffusion, ConfigRNN, ConfigLatentDiffusion
from data import utils
from data.dataset import IAMDataset, IAMonDataset, DataModule
from models.Diffusion.model import DiffusionWrapper
from models.LatentDiffusion.model import LatentDiffusionModel
from models.RNN.model import RNNModel

MODELS = {
    "Diffusion": DiffusionWrapper,
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
    deterministic: bool = False,
) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
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
    parser.add_argument(
        "--prepare-data",
        help="Flag for saving preprocessed dataset to H5 file",
        action="store_true",
    )
    parser.add_argument("--generate", help="Generate handwriting", action="store_true")
    parser.add_argument(
        "-t",
        "--text",
        help="Text to generate",
        type=str,
        default="Handwriting Synthesis in Python",
    )
    parser.add_argument(
        "-w",
        "--writer",
        help="Writer style. If not provided, the default writer is selected randomly",
        type=int,
        default=random.randint(0, 339),
    )

    parser.add_argument("--train", help="Train selected model", action="store_true")

    return parser.parse_args()


def prepare_data():
    global dataset

    if args.config in ("Diffusion", "RNN"):
        h5_file_path = Path("data/h5_dataset/train_val_iamondb.h5")
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
        utils.save_dataset_to_h5(dataset.dataset, h5_file_path)

    else:
        train_size = config.get("train_size", 0.85)
        val_size = 1.0 - train_size
        h5_file_path = Path("data/h5_dataset/train_iamdb.h5")

        with contextlib.suppress(FileNotFoundError):
            os.remove(h5_file_path)

        kwargs_dataset = dict(
            config=config,
            img_height=config.get("img_height", 90),
            img_width=config.get("img_width", 1400),
            max_text_len=config.max_text_len,
            max_files=train_size * config.max_files,
            dataset_type="train",
        )

        dataset = DATASETS[args.config](**kwargs_dataset)
        utils.save_dataset_to_h5(
            dataset.dataset,
            h5_file_path,
            latent=True,
            map_writer_ids=dataset.map_writer_id,
        )

        h5_file_path = Path("data/h5_dataset/val_iamdb.h5")
        with contextlib.suppress(FileNotFoundError):
            os.remove(h5_file_path)

        kwargs_dataset["max_files"] = val_size * config.max_files
        kwargs_dataset["dataset_type"] = "val"

        dataset = DATASETS[args.config](**kwargs_dataset)
        utils.save_dataset_to_h5(
            dataset.dataset,
            h5_file_path,
            latent=True,
            map_writer_ids=dataset.map_writer_id,
        )


def train_model():
    global dataset
    # * ModelCheckpoint to save 9 (most of the time) latest models from epochs *
    kwargs_trainer = dict(
        accelerator=config.device,
        default_root_dir=config.checkpoint_path,
        max_epochs=config.max_epochs,
        callbacks=[
            RichModelSummary(max_depth=3),
            RichProgressBar(refresh_rate=1, leave=True),
            # EarlyStopping(monitor="val_loss", patience=25),
            ModelCheckpoint(
                dirpath=config.checkpoint_path,
                monitor="loss",
                filename="{epoch}-{loss:.2f}-{val_loss:.2f}",
                save_top_k=9,
            ),
        ],
    )

    if args.config == "Diffusion":
        kwargs_model = dict(
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
        kwargs_trainer["devices"] = [1]

    set_random_seed()
    model = MODELS[args.config](**kwargs_model)
    dataset = DATASETS[args.config]
    data_module = DataModule(config=config, dataset=dataset)

    trainer = pl.Trainer(**kwargs_trainer)
    trainer.fit(model=model, datamodule=data_module)
    trainer.save_checkpoint(filepath=f"{config.checkpoint_path}/model.ckpt")


def generate_handwriting() -> None:
    model = MODELS[args.config].load_from_checkpoint(
        checkpoint_path=f"{config.checkpoint_path}/model.ckpt",
        map_location=torch.device(config.device),
    )
    model.eval()

    if args.config == "LatentDiffusion":
        print(f"Selected writer id: {args.writer}")
        model.generate(
            args.text,
            vocab=config.vocab,
            writer_id=args.writer,
            save_path="handwriting.png",
        )
    else:
        model.generate(args.text, save_path="handwriting.png", vocab=config.vocab)


if __name__ == "__main__":
    args = cli_main()
    dataset = None

    if sys.version_info < (3, 8):
        raise SystemExit("Only Python 3.8 and above is supported")

    config_file = f"configs/{args.config}/{args.config_file}"

    config = CONFIGS[args.config].from_yaml_file(
        file=config_file, decoder=yaml.load, Loader=yaml.Loader
    )

    if args.prepare_data:
        prepare_data()
    elif args.generate:
        generate_handwriting()
    elif args.train:
        train_model()
    else:
        raise ValueError("The selected argument does not exist")
