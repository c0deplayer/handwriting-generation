import contextlib
import math
import os
import shutil
import sys
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from typing import List, Union

import PIL.Image as ImageModule
import torch
import torchvision
import yaml
from cleanfid import fid
from einops import rearrange, repeat
from matplotlib import pyplot as plt
from PIL import ImageOps
from PIL.Image import Image
from torch import Tensor
from torch.utils.data import DataLoader
from torchmetrics.image.inception import InceptionScore
from torchvision.utils import save_image

from configs.settings import CALCULATION_BASE_DIR, CONFIGS, MODELS, MODELS_SN
from data import utils
from data.dataset import IAMDataset, IAMonDataset


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
        choices=["Diffusion", "LatentDiffusion"],
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
        "--strict",
        help="Strict mode for a dataset that excludes OOV words",
        action="store_true",
    )

    return parser.parse_args()


def full_sampling():
    """
    Conducts full sampling for generated and real images to calculate
    metrics like Inception Score (IS), Frechet Inception Distance (FID)
    and Kernel Inception Distance (KID).

    This function iterates over a dataset, generates handwriting samples using a model,
    and calculates IS, FID and KID for these samples compared to real images. It handles different
    configurations for models like LatentDiffusion and others. The results are saved and printed.
    """

    num_of_image = 1
    isc_fake = InceptionScore(normalize=False)
    isc_real = InceptionScore(normalize=False)
    kwargs_dataset = dict(
        config=config,
        img_height=config.img_height,
        img_width=config.img_width,
        max_text_len=config.max_text_len,
        max_files=config.max_files,
        dataset_type="train",
        strict=args.strict,
    )

    device = torch.device(config.device)
    if args.config == "LatentDiffusion":
        normalize_undo = utils.NormalizeInverse((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        fid_dataset = IAMDataset(**kwargs_dataset)

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

            fake_samples = generate_handwriting(labels, writer_ids)

            fake_samples = [
                rearrange(
                    torchvision.transforms.ToTensor()(fake_sample), "c w h -> 1 c w h"
                )
                for fake_sample in fake_samples
            ]
            fake_samples = torch.cat(fake_samples, dim=0)

            for real, fake in zip(images.unbind(0), fake_samples.unbind(0)):
                save_image(
                    real,
                    fp=f"{CALCULATION_BASE_DIR}/real_samples/img-{num_of_image}.jpeg",
                )
                save_image(
                    fake,
                    fp=f"{CALCULATION_BASE_DIR}/fake_samples/img-{num_of_image}.jpeg",
                )
                num_of_image += 1

            fake_samples = normalize_undo(fake_samples).type(torch.uint8)
            real_samples = normalize_undo(images).type(torch.uint8)

            isc_fake.update(fake_samples)
            isc_real.update(real_samples)

    else:
        kwargs_dataset["max_seq_len"] = config.max_seq_len
        fid_dataset = IAMonDataset(**kwargs_dataset)

        fid_dataloader = DataLoader(
            fid_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            pin_memory=True,
        )

        for i, (_, text, style, img) in enumerate(fid_dataloader, start=1):
            print(
                f"""
            ===================================
                Step {i}/{math.ceil(len(fid_dataset) / config.batch_size)}
            ===================================
                """
            )
            text, style = text.to(device), style.to(device)
            args.text, args.style_path = text, style

            fake_samples = generate_handwriting(text, style)

            fake_samples = [
                ImageModule.frombytes(
                    "RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb()
                )
                for fig in fake_samples
            ]

            total_width, total_height = fake_samples[0].size
            for sample in fake_samples[1:]:
                total_width = max(total_width, sample.width)
                total_height = max(total_height, sample.height)

            fake_samples_fixed = []
            for sample in fake_samples:
                img_fix = sample.convert("L")

                bbox = ImageOps.invert(img_fix).getbbox()
                img_fix = img_fix.crop(bbox)

                img_fix = ImageOps.pad(
                    image=img_fix,
                    size=(total_width, total_height),
                    method=ImageModule.LANCZOS,
                    color="white",
                    centering=(0.0, 0.5),
                )

                img_fix = img_fix.convert("RGB")

                fake_samples_fixed.append(
                    rearrange(
                        torchvision.transforms.ToTensor()(img_fix), "c h w -> 1 c h w"
                    )
                )

            fake_samples = torch.cat(fake_samples_fixed, dim=0)

            for real, fake in zip(img.unbind(0), fake_samples.unbind(0)):
                save_image(
                    real,
                    fp=f"{CALCULATION_BASE_DIR}/real_samples/img-{num_of_image}.jpeg",
                )
                save_image(
                    fake,
                    fp=f"{CALCULATION_BASE_DIR}/fake_samples/img-{num_of_image}.jpeg",
                )
                num_of_image += 1

            fake_samples = fake_samples.type(torch.uint8)
            real_samples = repeat(
                img.type(torch.uint8), "b c h w -> b (repeats c) h w", repeats=3
            )

            isc_fake.update(fake_samples)
            isc_real.update(real_samples)

    isc_value_fake = isc_fake.compute()
    isc_value_real = isc_real.compute()
    fid_value = fid.compute_fid(
        fdir1=f"{CALCULATION_BASE_DIR}/real_samples",
        fdir2=f"{CALCULATION_BASE_DIR}/fake_samples",
        batch_size=config.batch_size // 2,
        num_workers=os.cpu_count() // 4,
        device=device,
    )
    kid_value = fid.compute_kid(
        fdir1=f"{CALCULATION_BASE_DIR}/real_samples",
        fdir2=f"{CALCULATION_BASE_DIR}/fake_samples",
        batch_size=config.batch_size // 2,
        num_workers=os.cpu_count() // 4,
        device=device,
    )

    print(
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
        f"|| Model: {MODELS_SN[args.config]} "
        f"|| FID value: {fid_value} "
        f"|| KID value: {kid_value} "
        f"|| IS value: {isc_value_fake[0]} +- {isc_value_fake[1]} "
        f"|| (Dataset) IS value: {isc_value_real[0]} +- {isc_value_real[1]}\n"
    )

    with open(f"{config.checkpoint_path}/metrics.txt", mode="a+") as f:
        f.write(
            f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
            f"|| Model: {MODELS_SN[args.config]} "
            f"|| FID value: {fid_value} "
            f"|| KID value: {kid_value} "
            f"|| IS value: {isc_value_fake[0]} +- {isc_value_fake[1]} "
            f"|| (Dataset) IS value: {isc_value_real[0]} +- {isc_value_real[1]}\n"
        )


def generate_handwriting(
    text: Tensor, style: Tensor
) -> Union[Image, plt.Figure, List[Image], List[plt.Figure]]:
    """
    Generates handwriting samples based on given text and style using the specified model.

    Args:
        text (Tensor): Input tensor representing text.
        style (Tensor): Style tensor representing handwriting style or writer ID.

    Returns:
        Union[Image, plt.Figure, List[Image], List[plt.Figure]]: Generated handwriting images or figures.
    """

    if args.config == "LatentDiffusion":
        return model.generate(
            text,
            vocab=config.vocab,
            max_text_len=config.max_text_len,
            writer_id=style,
            save_path=None,
            color="black",
            is_fid=True,
        )

    return model.generate(
        text,
        save_path=None,
        vocab=config.vocab,
        max_text_len=config.max_text_len,
        color="black",
        style_path=style,
        is_fid=True,
    )


if __name__ == "__main__":
    args = cli_main()

    if sys.version_info < (3, 8):
        raise SystemExit("Only Python 3.8 and above is supported")

    config_file = f"configs/{args.config}/{args.config_file}"

    config = CONFIGS[args.config].from_yaml_file(
        file=config_file, decoder=yaml.load, Loader=yaml.Loader
    )

    model = MODELS[args.config].load_from_checkpoint(
        checkpoint_path=f"{config.checkpoint_path}/model.ckpt",
        map_location=torch.device(config.device),
    )
    model.eval()

    with contextlib.suppress(FileNotFoundError):
        shutil.rmtree(CALCULATION_BASE_DIR)

    Path(f"{CALCULATION_BASE_DIR}/real_samples").mkdir(parents=True)
    Path(f"{CALCULATION_BASE_DIR}/fake_samples").mkdir(parents=True)

    full_sampling()
