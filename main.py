import contextlib
import datetime
import json
import math
import os
import random
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import List, Union

import PIL.Image as ImageModule
import lightning.pytorch as pl
import neptune
import numpy as np
import torch
import torchvision.transforms
import yaml
from PIL.Image import Image
from einops import rearrange
from lightning.pytorch.callbacks import (
    RichModelSummary,
    RichProgressBar,
    ModelCheckpoint,
    EarlyStopping,
)
from lightning.pytorch.loggers.neptune import NeptuneLogger
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance

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

MODELS_SN = {
    "Diffusion": "DM",
    "RNN": "RNN",
    "LatentDiffusion": "LDM",
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
        "-n",
        "--neptune",
        help="Flag for using NeptuneLogger",
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
    parser.add_argument(
        "-s",
        "--style_path",
        help="Filename for style. If not provided, the default style is selected randomly",
        type=str,
        default=f"{os.listdir('assets')[np.random.randint(0, len(os.listdir('assets')))]}",
    )

    parser.add_argument(
        "--fid",
        help="Calculate FID between original and generated handwriting",
        action="store_true",
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


def train_model():
    global dataset
    patience = math.ceil(config.max_epochs / config.batch_size)
    print(f"Patience set to {patience} epoch")
    # * ModelCheckpoint to save 9 (most of the time) latest models from epochs *
    trainer_params = dict(
        accelerator=config.device,
        default_root_dir=config.checkpoint_path,
        max_epochs=config.max_epochs,
        callbacks=[
            RichModelSummary(max_depth=5),
            RichProgressBar(refresh_rate=1, leave=True),
            EarlyStopping(monitor="val/loss", patience=patience),
            ModelCheckpoint(
                dirpath=config.checkpoint_path,
                monitor="val/loss",
                filename="epoch{epoch:02d}-train_loss{train/loss:.2f}-val_loss{val/loss:.2f}",
                auto_insert_metric_name=False,
                save_top_k=8,
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
            }
        )

    elif args.config == "RNN":
        model_params = dict(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_window=config.num_window,
            num_mixture=config.num_mixture,
            vocab_size=config.vocab_size,
            bias=config.bias,
            clip_grads=(config.gru_clip, config.mdn_clip),
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
            mode="async",
            capture_stdout=False,
            source_files=source_files,
            dependencies="infer",
        )

        model_version = neptune.init_model_version(
            model=f"HAN-{MODELS_SN[args.config]}",
            project="codeplayer/handwriting-generation",
            mode="async",
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
        # noinspection PyUnboundLocalVariable
        model_version["model/binary"].upload(f"{config.checkpoint_path}/model.ckpt")
        model_version.stop()


def generate_handwriting() -> Union[Image, plt.Figure, List[Image], List[plt.Figure]]:
    model = MODELS[args.config].load_from_checkpoint(
        checkpoint_path=f"{config.checkpoint_path}/model.ckpt",
        map_location=torch.device(config.device),
    )
    model.eval()

    save_path = (
        None
        if args.fid
        else f"{os.getcwd()}/images/{args.config}/{args.text.replace(' ', '-')}.jpeg"
    )
    if args.config == "LatentDiffusion":
        print(f"Selected writer id: {args.writer}")

        return model.generate(
            args.text,
            vocab=config.vocab,
            writer_id=args.writer,
            save_path=save_path,
            color="black",
            is_fid=args.fid,
        )

    return model.generate(
        args.text,
        save_path=save_path,
        vocab=config.vocab,
        color="black",
        style_path=args.style_path,
        is_fid=args.fid,
    )


def full_sampling_for_fid():
    # TODO: Implement IS score
    # TODO: Implement for RNN models
    fid = FrechetInceptionDistance(normalize=True)
    fid.set_dtype(torch.float64)

    device = torch.device(config.device)
    if args.config == "LatentDiffusion":
        normalize_undo = utils.NormalizeInverse((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        fid_dataset = IAMDataset(
            config=config,
            img_height=config.img_height,
            img_width=config.img_width,
            max_text_len=config.max_text_len,
            max_files=0,
            dataset_type="train",
        )

        fid_dataloader = DataLoader(
            fid_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            pin_memory=True,
        )

        for i, (writer_ids, images, labels) in enumerate(fid_dataloader, start=1):
            print(
                f"""
            ===================================
                Step {i}/{math.ceil(len(fid_dataset) / config.batch_size)}
            ===================================
                """
            )
            writer_ids, labels = writer_ids.to(device), labels.to(device)
            real_samples = normalize_undo(images)
            args.text, args.writer = labels, writer_ids

            fake_samples = generate_handwriting()

            fake_samples = [
                rearrange(
                    torchvision.transforms.ToTensor()(fake_sample), "c w h -> 1 c w h"
                )
                for fake_sample in fake_samples
            ]
            fake_samples = torch.cat(fake_samples, dim=0)

            fid.update(real_samples, real=True)
            fid.update(fake_samples, real=False)

    elif args.config == "Diffusion":
        fid_dataset = IAMonDataset(
            config=config,
            img_height=config.img_height,
            img_width=config.img_width,
            max_text_len=config.max_text_len,
            max_seq_len=config.max_seq_len,
            max_files=0,
        )

        fid_dataloader = DataLoader(
            fid_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            pin_memory=True,
        )

        for i, (_, text, style, real_samples) in enumerate(fid_dataloader, start=1):
            print(
                f"""
            ===================================
                Step {i}/{math.ceil(len(fid_dataset) / config.batch_size)}
            ===================================
                """
            )
            text, style = text.to(device), style.to(device)
            args.text, args.style_path = text, style

            fake_samples = generate_handwriting()

            fake_samples = [
                ImageModule.frombytes(
                    "RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb()
                )
                for fig in fake_samples
            ]

            fake_samples = [
                rearrange(torchvision.transforms.ToTensor()(image), "c h w -> 1 c h w")
                for image in fake_samples
            ]

            fake_samples = torch.cat(fake_samples, dim=0)

            fid.update(real_samples, real=True)
            fid.update(fake_samples, real=False)
    else:
        raise NotImplementedError()

    fid_value = fid.compute().item()
    print(f"FID value: {fid_value}")

    with open(f"{config.checkpoint_path}/metrics.txt", mode="w") as f:
        f.write(
            f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
            f"|| Model: {MODELS_SN[args.config]} "
            f"|| FID value: {fid_value} "
            f"|| IS value: None"
        )


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
    elif args.fid:
        full_sampling_for_fid()
    else:
        raise ValueError("The selected argument does not exist")
