import json
import os
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from typing import Union, Tuple, Dict, List, Any

import torch
import torchvision.transforms.v2 as transforms
import yaml
from einops import repeat
from rich.progress import track
from torch import Tensor, nn
from torch.utils.data import Dataset, DataLoader

from configs.config import ConfigDiffusion, ConfigLatentDiffusion
from configs.settings import GEN_DATASET_DIR, CONFIGS
from data.utils import get_image
from models.ConvNeXt.model import ConvNeXt_M
from models.InceptionV3.model import InceptionV3_M


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
        "--new-model",
        help="Whether to initialize a new model (ConvNeXt_M) or load an old one (InceptionV3_M)",
        action="store_true",
    )

    return parser.parse_args()


class ImageDataset(Dataset):
    def __init__(
        self,
        config: Union[ConfigDiffusion, ConfigLatentDiffusion],
    ) -> None:
        super().__init__()

        self.__config = config
        self._diffusion = isinstance(config, ConfigDiffusion)
        self.transforms = transforms.Compose(
            [transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)]
        )

        self.img_height = config.get("img_height", 90)
        self.img_width = config.get("img_width", 1400)

        self.__load_dataset__()
        print(
            f"Size of dataset: {len(self.dataset)} || Length of writer styles -- {len(self.map_writer_id)}"
        )

    def __load_dataset__(self) -> None:
        json_file_path = Path(
            f"./data/json_writer_ids/train_writer_ids_{'iamondb' if self._diffusion else 'iamdb'}.json"
        )
        json_writer_id_info_path = Path(
            f"{GEN_DATASET_DIR}/{'Diffusion' if self._diffusion else 'LatentDiffusion'}/writer_id_info.json"
        )

        with open(json_file_path, mode="r") as fp:
            self.__map_writer_id = json.load(fp)

        with open(json_writer_id_info_path, mode="r") as fp:
            map_images = json.load(fp)

        centering = (0.0, 0.5) if self._diffusion else (0.5, 0.5)
        dataset = []

        for img_name, writer_id in track(
            map_images.items(), description="Loading generated dataset..."
        ):
            img_path = Path(
                f"{GEN_DATASET_DIR}/{'Diffusion' if self._diffusion else 'LatentDiffusion'}/{img_name}"
            )

            image = get_image(
                img_path,
                width=self.img_width,
                height=self.img_height,
                latent=self._diffusion,
                centering=centering,
            )

            dataset.append(
                {
                    "writer": writer_id,
                    "image": image,
                }
            )

        self.__dataset = dataset

    @property
    def config(self) -> Union[ConfigDiffusion, ConfigLatentDiffusion]:
        return self.__config

    @property
    def dataset(self) -> List[Dict[str, Any]]:
        return self.__dataset

    @property
    def map_writer_id(self) -> Dict[str, int]:
        return self.__map_writer_id

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        image = self.transforms(self.dataset[index]["image"])
        writer_id = torch.tensor(self.dataset[index]["writer"], dtype=torch.int32)

        if not self._diffusion:
            image = repeat(image, "1 h w -> 3 h w")

        return writer_id, image

    def __len__(self) -> int:
        return len(self.dataset)


def evaluate_model(classifier: nn.Module, data: DataLoader) -> float:
    classifier.eval()
    performance_score = 0.0

    with torch.inference_mode():
        for batch in track(data):
            writer_id, image = batch

            image = image.to(device)
            writer_id = writer_id.type(torch.LongTensor).to(device)

            outputs = classifier(image)

            performance_score += compute_accuracy(outputs, writer_id)

    performance_score /= len(data)

    return performance_score


def compute_accuracy(predictions: Tensor, labels: Tensor) -> float:
    classes = torch.argmax(predictions, dim=1)
    return torch.mean((classes == labels).float()).item()


if __name__ == "__main__":
    args = cli_main()

    config_file = f"configs/{args.config}/{args.config_file}"

    config = CONFIGS[args.config].from_yaml_file(
        file=config_file, decoder=yaml.load, Loader=yaml.Loader
    )
    device = torch.device(config.device)

    dataset = ImageDataset(config=config)

    num_class = len(dataset.map_writer_id)

    if args.new_model:
        model = ConvNeXt_M(num_class, device=device)
        model.load_state_dict(
            torch.load(
                f"./model_checkpoints/ConvNeXt/ConvNeXt_M-{args.config.lower()}.pth",
                map_location=device,
            )
        )
    else:
        model = InceptionV3_M(num_class, device=device)
        model.load_state_dict(
            torch.load(
                f"./model_checkpoints/InceptionV3/InceptionV3_M-{args.config.lower()}.pth",
                map_location=device,
            )
        )

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=os.cpu_count(),
    )

    accuracy = evaluate_model(model, dataloader)

    print(
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
        f"|| Model: {'ConvNeXt' if args.new_model else 'InceptionV3'} "
        f"|| Model Type: {args.config} "
        f"|| Accuracy: {accuracy * 100:.4f}%\n"
    )

    with open(
        f"./model_checkpoints/{'ConvNeXt' if args.new_model else 'InceptionV3'}/results.txt",
        mode="a+",
    ) as file:
        file.write(
            f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
            f"|| Model: {'ConvNeXt' if args.new_model else 'InceptionV3'} "
            f"|| Model Type: {args.config} "
            f"|| Accuracy: {accuracy * 100:.4f}%\n"
        )
