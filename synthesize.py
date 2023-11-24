import os
import random
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import List, Union

import numpy as np
import torch
import yaml
from PIL.Image import Image
from matplotlib import pyplot as plt

from configs.settings import CONFIGS, MODELS


def cli_main():
    """
    Command-line interface for initializing and parsing arguments related to model configuration, text input,
     and style settings.

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
        default=random.randint(0, 329),
    )

    parser.add_argument(
        "--color",
        help="Handwriting color. If not provided, the default color is black",
        type=str,
        default="black",
    )

    parser.add_argument(
        "-s",
        "--style_path",
        help="Filename for style. If not provided, the default style is selected randomly",
        type=str,
        default=f"{os.listdir('assets')[np.random.randint(0, len(os.listdir('assets')))]}",
    )

    return parser.parse_args()


def generate_handwriting() -> Union[Image, plt.Figure, List[Image], List[plt.Figure]]:
    """
    Generates handwriting based on the specified model and input parameters.

    Depending on the model type (LatentDiffusion, RNN, or Diffusion), the function
    loads the appropriate model checkpoint, configures it, and generates handwriting
    images based on the provided text, style, and color.

    Returns:
        Union[Image, plt.Figure, List[Image], List[plt.Figure]]: Generated handwriting images or figures.
    """

    model = MODELS[args.config].load_from_checkpoint(
        checkpoint_path=f"{config.checkpoint_path}/model.ckpt",
        map_location=torch.device(config.device),
    )
    model.eval()
    # seed_everything(seed=42)

    if args.config == "LatentDiffusion":
        print(f"Selected writer id: {args.writer}")

        save_path = Path(
            f"{os.getcwd()}/images/{args.config}/{args.text.replace(' ', '-')}-w{args.writer}.jpeg"
        )

        return model.generate(
            args.text,
            vocab=config.vocab,
            max_text_len=config.max_text_len,
            writer_id=args.writer,
            save_path=save_path,
            color=args.color,
        )

    save_path = Path(
        f"{os.getcwd()}/images/{args.config}/{args.text.replace(' ', '-')}.jpeg"
    )

    if args.config == "Diffusion":
        return model.generate(
            args.text,
            save_path=save_path,
            max_text_len=config.max_text_len,
            vocab=config.vocab,
            color=args.color,
            style_path=args.style_path,
        )
    else:
        return model.generate(
            args.text,
            save_path=save_path,
            max_text_len=config.max_text_len,
            vocab=config.vocab,
            color=args.color,
        )


if __name__ == "__main__":
    args = cli_main()

    if sys.version_info < (3, 8):
        raise SystemExit("Only Python 3.8 and above is supported")

    config_file = f"configs/{args.config}/{args.config_file}"

    config = CONFIGS[args.config].from_yaml_file(
        file=config_file, decoder=yaml.load, Loader=yaml.Loader
    )

    generate_handwriting()
