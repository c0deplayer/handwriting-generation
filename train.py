import os
from argparse import ArgumentParser

import lightning.pytorch as pl
import numpy as np
import torch
import yaml
from lightning.pytorch.callbacks import RichModelSummary, RichProgressBar
from lightning.pytorch.plugins import MixedPrecisionPlugin

from configs.config import ConfigDiffusion, ConfigRNN
from data.dataset import DataModule, IAMDataset, IAMonDataset
from models.Diffusion.model import DiffusionModel
from models.RNN.model import RecurrentNeuralNetwork

MODELS = {"Diffusion": DiffusionModel, "RNN": RecurrentNeuralNetwork}
CONFIGS = {"Diffusion": ConfigDiffusion, "RNN": ConfigRNN}
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

    config_file = f"configs/{args.config}/{args.config_file}"

    config = CONFIGS[args.config].from_yaml_file(
        file=config_file, decoder=yaml.load, Loader=yaml.Loader
    )
    print(config.vocab_size)

    kwargs_trainer = dict(
        accelerator=config.device,
        default_root_dir=config.checkpoint_path,
        max_epochs=config.max_epochs,
        enable_checkpointing=True,
        enable_model_summary=True,
        callbacks=[
            RichModelSummary(max_depth=2),
            RichProgressBar(refresh_rate=1),
        ],
    )

    if args.config == "Diffusion":
        kwargs_model = dict(
            device=config.device,
            num_layers=config.num_layers,
            c1=config.channels,
            c2=config.channels * 3 // 2,
            c3=config.channels * 2,
            drop_rate=config.drop_rate,
            vocab_size=config.vocab_size,
        )

        kwargs_trainer |= dict(
            gradient_clip_val=config.clip_grad,
            gradient_clip_algorithm=config.clip_algorithm,
        )

    elif args.config == "RNN":
        kwargs_model = dict(
            input_size=config.input_size,
            device=config.device,
            hidden_size=config.hidden_size,
            num_window=config.num_window,
            num_mixture=config.num_mixture,
            vocab_size=config.vocab_size,
            bias=config.bias,
            clip_grads=(config.lstm_clip, config.mdn_clip),
        )

        # * The diffusion model requires at least 32-bit precision, otherwise it generates NaN values. *
        if config.device == "cuda":
            kwargs_trainer |= dict(
                plugins=[
                    MixedPrecisionPlugin(precision="16-mixed", device=config.device)
                ],
            )
    else:
        raise NotImplemented

    # * Only for university's server, which have two GPUs *
    if config.device == "cuda" and args.remote:
        kwargs_trainer |= dict(
            devices=[1],
        )

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
