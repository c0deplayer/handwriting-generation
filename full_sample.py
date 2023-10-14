import math
import sys
from argparse import ArgumentParser
from datetime import datetime
from typing import Union, List

import PIL.Image as ImageModule
import torch
import torchvision
import yaml
from PIL.Image import Image
from einops import rearrange
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance

from configs.settings import MODELS, MODELS_SN, CONFIGS
from data import utils
from data.dataset import IAMDataset, IAMonDataset


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


def full_sampling():
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
            f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
            f"|| Model: {MODELS_SN[args.config]} "
            f"|| FID value: {fid_value} "
            f"|| IS value: None"
        )


def generate_handwriting() -> Union[Image, plt.Figure, List[Image], List[plt.Figure]]:
    if args.config == "LatentDiffusion":
        return model.generate(
            args.text,
            vocab=config.vocab,
            writer_id=args.writer,
            save_path=None,
            color="black",
            is_fid=args.fid,
        )

    return model.generate(
        args.text,
        save_path=None,
        vocab=config.vocab,
        color="black",
        style_path=args.style_path,
        is_fid=args.fid,
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

    full_sampling()
