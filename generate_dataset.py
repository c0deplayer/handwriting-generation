import contextlib
import json
import math
import os
import shutil
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import List, Union

import PIL.Image as ImageModule
import torch
import yaml
from PIL.Image import Image
from matplotlib import pyplot as plt
from torch import Tensor
from torch.utils.data import DataLoader

from configs.settings import CONFIGS, MODELS, GEN_DATASET_DIR
from data.dataset import IAMDataset, IAMonDataset
from models.Diffusion.text_style import StyleExtractor


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


def generate_dataset():
    num_of_image = 1
    kwargs_dataset = dict(
        config=config,
        img_height=config.img_height,
        img_width=config.img_width,
        max_text_len=config.max_text_len,
        max_files=config.max_files,
        dataset_type="train",
        strict=args.strict,
    )
    map_images = {}

    device = torch.device(config.device)
    if args.config == "LatentDiffusion":
        dataset = IAMDataset(**kwargs_dataset)

        fid_dataloader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=os.cpu_count(),
        )

        for i, (writer_ids, images, labels) in enumerate(fid_dataloader, start=1):
            print(
                f"""
            ===================================
                Step {i}/{math.ceil(len(dataset) / config.batch_size)}
            ===================================
                """
            )
            writer_ids, labels = writer_ids.to(device), labels.to(device)

            generated_images = generate_handwriting(labels, writer_ids)

            for writer_id, gen_img in zip(writer_ids.unbind(0), generated_images):
                gen_img.save(f"{GEN_DATASET_DIR}/{args.config}/img-{num_of_image}.png")
                map_images[f"img-{num_of_image}.png"] = writer_id.item()
                num_of_image += 1

    else:
        kwargs_dataset["max_seq_len"] = config.max_seq_len
        kwargs_dataset["inception"] = True
        dataset = IAMonDataset(**kwargs_dataset)
        style_extractor = StyleExtractor(device=torch.device(config.device))

        fid_dataloader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=False,
            pin_memory=True,
        )

        for i, (writer_ids, img, text) in enumerate(fid_dataloader, start=1):
            print(
                f"""
            ===================================
                Step {i}/{math.ceil(len(dataset) / config.batch_size)}
            ===================================
                """
            )
            writer_image = img * 255
            with torch.inference_mode():
                style = style_extractor(writer_image)

            text, style = text.to(device), style.to(device)

            generated_images = generate_handwriting(text, style)

            generated_images = [
                ImageModule.frombytes(
                    "RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb()
                )
                for fig in generated_images
            ]

            for writer_id, gen_img in zip(writer_ids.unbind(0), generated_images):
                gen_img.save(f"{GEN_DATASET_DIR}/{args.config}/img-{num_of_image}.png")
                map_images[f"img-{num_of_image}.png"] = writer_id.item()
                num_of_image += 1

    with open(f"{GEN_DATASET_DIR}/{args.config}/writer_id_info.json", mode="w") as fp:
        json.dump(map_images, fp, indent=4)


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
        shutil.rmtree(f"{GEN_DATASET_DIR}/{args.config}")

    Path(f"{GEN_DATASET_DIR}/{args.config}").mkdir(parents=True)

    generate_dataset()
