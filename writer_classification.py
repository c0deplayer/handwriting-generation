import json
import os
from argparse import ArgumentParser

import torch
import torch.nn as nn
import yaml
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, random_split

from configs.settings import CONFIGS, DATASETS
from models.ConvNeXt.model import ConvNeXt_M
from models.InceptionV3.model import InceptionV3_M


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
        "--new-model",
        help="Whether to initialize a new model (ConvNeXt_M) or load an old one (InceptionV3_M)",
        action="store_true",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = cli_main()

    config_file = f"configs/{args.config}/{args.config_file}"

    config = CONFIGS[args.config].from_yaml_file(
        file=config_file, decoder=yaml.load, Loader=yaml.Loader
    )
    device = torch.device(config.device)

    with open(
        f"data/json_writer_ids/train_writer_ids_{'iamondb' if args.config == 'Diffusion' else 'iamdb'}.json",
        mode="r",
    ) as fp:
        map_writer_id = json.load(fp)
        num_class = len(map_writer_id)

        del map_writer_id

    if args.new_model:
        from models.ConvNeXt.utils import EarlyStopper, train_loop

        model = ConvNeXt_M(num_class, device=device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    else:
        from models.InceptionV3.utils import EarlyStopper, train_loop

        model = InceptionV3_M(num_class, device=device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    scheduler = StepLR(optimizer, step_size=15, gamma=0.1)
    early_stopper = EarlyStopper(patience=8, verbose=True, mode="max")
    criterion = nn.CrossEntropyLoss()

    kwargs_dataset = dict(
        config=config,
        img_height=config.get("img_height", 90),
        img_width=config.get("img_width", 1400),
        max_text_len=config.max_text_len,
        max_files=config.train_size * config.max_files,
        dataset_type="train",
    )

    if args.config == "Diffusion":
        kwargs_dataset["max_seq_len"] = config.max_seq_len
        kwargs_dataset["inception"] = True

    dataset = DATASETS[args.config](**kwargs_dataset)

    train_size = (
        int(0.85 * len(dataset))
        if args.config == "Diffusion"
        else int(0.9 * len(dataset))
    )
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size // 4,
        shuffle=True,
        pin_memory=True,
        num_workers=os.cpu_count() // 4,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size // 4,
        shuffle=False,
        pin_memory=True,
        num_workers=os.cpu_count() // 4,
    )

    train_config = dict(
        num_class=num_class,
        model=args.config.lower(),
        train_dataset_len=len(train_dataset),
        val_dataset_len=len(val_dataset),
    )

    train_loop(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        num_epochs=50,
        train_config=train_config,
        device=device,
        scheduler=scheduler,
        early_stopper=early_stopper,
    )
