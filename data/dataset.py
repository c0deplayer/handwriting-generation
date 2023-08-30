import os
import random
from pathlib import Path
from typing import Any, Literal

import lightning.pytorch as pl
import torch
import torchvision.transforms
from einops import rearrange
from rich.progress import track
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import Compose

from configs.config import ConfigDiffusion, ConfigLatentDiffusion, ConfigRNN
from models.Diffusion.text_style import StyleExtractor
# noinspection PyPackages
from . import utils
# noinspection PyPackages
from .tokenizer import Tokenizer


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset: Dataset,
        config: ConfigDiffusion | ConfigRNN | ConfigLatentDiffusion,
    ) -> None:
        """
        _summary_

        Parameters
        ----------
        dataset: _type_
            _description_
        config : ConfigDiffusion | ConfigRNN | ConfigLatentDiffusion
            _description_
        """

        super().__init__()

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.dataset = dataset
        self.__config = config
        self.batch_size = config.batch_size
        self.max_text_len = config.max_text_len
        self.max_files = config.max_files

        # TODO: temporary solution, find better one
        if isinstance(config, ConfigDiffusion):
            self.img_height = config.img_height
            self.img_width = config.img_width
            self.train_size = config.train_size
            self.val_size = 1.0 - self.train_size
            self.max_seq_len = config.max_seq_len
        elif isinstance(config, ConfigLatentDiffusion):
            self.img_height = config.img_height
            self.img_width = config.img_width
            # TODO: temporary solution, find better one
            if self.max_files:
                self.train_size = int(config.max_files * config.train_size)
                self.val_size = config.max_files - self.train_size
            else:
                self.train_size, self.val_size = 0, 0

        elif isinstance(config, ConfigRNN):
            self.img_height = 90
            self.img_width = 1400
            self.train_size = config.train_size
            self.val_size = 1.0 - self.train_size
            self.max_seq_len = config.max_seq_len
        else:
            raise RuntimeError(
                f"Expected ConfigDiffusion | ConfigRNN | ConfigLatentDiffusion, got {str(config)}"
            )

    # noinspection PyCallingNonCallable
    def setup(self, stage: str) -> None:
        if stage == "fit":
            if isinstance(self.__config, ConfigLatentDiffusion):
                transforms = torchvision.transforms.Compose(
                    [
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Normalize(
                            (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
                        ),
                    ]
                )
                self.train_dataset = self.dataset(
                    config=self.__config,
                    img_height=self.img_height,
                    img_width=self.img_width,
                    max_text_len=self.max_text_len,
                    max_files=self.train_size,
                    dataset_type="train",
                    transforms=transforms,
                )
                self.val_dataset = self.dataset(
                    config=self.__config,
                    img_height=self.img_height,
                    img_width=self.img_width,
                    max_files=self.val_size,
                    max_text_len=self.max_text_len,
                    dataset_type="val",
                    transforms=transforms,
                )
            else:
                iam_full = self.dataset(
                    config=self.__config,
                    img_height=self.img_height,
                    img_width=self.img_width,
                    max_text_len=self.max_text_len,
                    max_seq_len=self.max_seq_len,
                    max_files=self.max_files,
                )
                self.train_dataset, self.val_dataset = random_split(
                    iam_full, [self.train_size, self.val_size]
                )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            # num_workers=os.cpu_count(),
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            # num_workers=os.cpu_count(),
            pin_memory=True,
        )


class IAMonDataset(Dataset):
    def __init__(
        self,
        img_height: int,
        img_width: int,
        max_text_len: int,
        max_seq_len: int,
        max_files: int,
        config: ConfigDiffusion | ConfigRNN,
        **kwargs,
    ) -> None:
        """
        _summary_

        Parameters
        ----------
        img_height : int
            _description_
        img_width : int
            _description_
        max_text_len : int
            _description_
        max_seq_len : int
            _description_
        max_files : int
            _description_
        config : ConfigDiffusion | ConfigRNN
            _description_
        """

        super().__init__()

        self.__config = config
        self.img_height = img_height
        self.img_width = img_width
        self.max_seq_len = (
            max_seq_len
            if max_seq_len > 0
            else utils.get_max_seq_len(Path(f"{config.data_path}/lineStrokes"))
        )
        self.max_text_len = max_text_len
        self.max_files = max_files
        self.diffusion = isinstance(config, ConfigDiffusion)

        self.style_extractor = StyleExtractor(device=torch.device(config.device))
        self.tokenizer = Tokenizer(config.vocab)

        with open(f"{config.data_path}/dataset.txt", mode="r") as f:
            self.dataset_txt = f.readlines()

        self.__load_data()
        self._mean = utils.compute_mean(self.dataset)
        self._std = utils.compute_std(self.dataset)

    def __load_data(self) -> None:
        transform = torchvision.transforms.Compose(
            [torchvision.transforms.PILToTensor()]
        )
        dataset = []
        raw_data_path = Path(self.config.data_path)
        ascii_path = raw_data_path / "ascii"
        strokes_path = raw_data_path / "lineStrokes"
        img_path = raw_data_path / "lineImages"

        for line in track(
            self.dataset_txt, description="Preparing IAM Online DataLoader..."
        ):
            writer_id, idx = line.strip().split(",")
            path_txt = ascii_path / f"{idx[:3]}/{idx[:7]}/{idx}.txt"

            path_images = img_path / f"{idx[:3]}/{idx[:7]}"
            images = os.listdir(path_images)
            shuffled_images = images.copy()
            random.shuffle(shuffled_images)

            transcription = utils.get_transcription(path_txt)

            for file, raw_text in transcription.items():
                if len(raw_text) >= self.max_text_len:
                    shuffled_images.pop(0)
                    continue

                path_file_xml = strokes_path / f"{idx[:3]}/{idx[:7]}/{file}.xml"
                path_file_tif = path_images / shuffled_images.pop(0)

                strokes = utils.get_line_stroke(path_file_xml, diffusion=self.diffusion)
                strokes = utils.pad_stroke(strokes, max_length=self.max_seq_len)

                eye = torch.eye(self.tokenizer.get_vocab_size())
                text = self.tokenizer.encode(raw_text)
                text = utils.fill_text(text, max_len=self.max_text_len)
                onehot = eye[text].numpy()

                image = utils.get_image(
                    path_file_tif, width=self.img_width, height=self.img_height
                )

                if strokes is None or image.size[0] >= self.img_width:
                    continue

                image = utils.pad_image(
                    image, width=self.img_width, height=self.img_height
                )
                writer_image = transform(image).to(torch.float32)
                if self.diffusion:
                    writer_image = rearrange(writer_image, "1 h w -> 1 1 h w")

                    with torch.no_grad():
                        style = self.style_extractor(writer_image)
                else:
                    style = writer_image

                style = rearrange(style, "1 h w -> w h")

                dataset.append(
                    {
                        "writer": writer_id,
                        "file": file,
                        "raw_text": raw_text,
                        "strokes": strokes,
                        "text": text,
                        "onehot": onehot,
                        "image": image,
                        "style": style,
                    }
                )

                if self.max_files and len(dataset) >= self.max_files:
                    self.__dataset = dataset
                    print(f"Size of dataset: {len(self.dataset)}")
                    return

        self.__dataset = dataset
        print(f"Size of dataset: {len(self.dataset)}")

    @property
    def config(self) -> ConfigDiffusion | ConfigRNN:
        return self.__config

    @property
    def dataset(self) -> list[dict[str, Any]]:
        return self.__dataset

    @property
    def mean(self) -> Tensor:
        return self._mean

    @property
    def std(self) -> Tensor:
        return self._std

    def normalize(self, strokes: Tensor) -> Tensor:
        return (strokes - self.mean) / self.std

    def denormalize(self, strokes: Tensor) -> Tensor:
        return strokes * self.std + self.mean

    def __getitem__(self, index: int) -> tuple[Tensor, ...]:
        if self.diffusion:
            strokes = torch.tensor(self.dataset[index]["strokes"], dtype=torch.float32)
            text = torch.tensor(self.dataset[index]["text"], dtype=torch.int32)
            style = self.dataset[index]["style"]

            strokes = self.normalize(strokes)

            return strokes, text, style

        else:
            strokes = torch.tensor(self.dataset[index]["strokes"], dtype=torch.float32)
            text = torch.tensor(self.dataset[index]["onehot"], dtype=torch.float16)

            strokes = self.normalize(strokes)

            return strokes, text

    def __len__(self) -> int:
        return len(self.dataset)


class IAMDataset(Dataset):
    def __init__(
        self,
        config: ConfigLatentDiffusion,
        img_height: int,
        img_width: int,
        max_text_len: int,
        max_files: int,
        dataset_type: Literal["train", "val", "test"],
        transforms: Compose,
        **kwargs,
    ) -> None:
        self.__config = config
        self.max_text_len = max_text_len
        self.max_files = max_files
        self.transforms = transforms
        self.img_height = img_height
        self.img_width = img_width
        self.dataset_type = dataset_type

        self.tokenizer = Tokenizer(config.vocab)

        type_dict = {
            "train": "iam_tr_va1.filter",
            "val": "iam_va2.filter",
            "test": "iam_test.filter",
        }

        with open(f"{config.data_path}/{type_dict[dataset_type]}", mode="r") as f:
            self.dataset_txt = f.readlines()

        self.__map_writer_id = {}
        self.__load_data()
        print(f"Length of writer styles -- {len(self.map_writer_id)}")

    def __load_data(self) -> None:
        dataset = []
        raw_data_path = Path(self.config.data_path)

        for line in track(self.dataset_txt, description="Preparing IAM Dataloader..."):
            parts = line.split(" ")
            writer_id, image_id = parts[0].split(",")[0], parts[0].split(",")[1]
            label = parts[1].rstrip()
            image_parts = image_id.split("-")
            f_folder, s_folder = (
                image_parts[0],
                f"{image_parts[0]}-{image_parts[1]}",
            )

            img_path = raw_data_path / f"words/{f_folder}/{s_folder}/{image_id}.png"

            image = utils.get_image(
                img_path,
                width=self.img_width,
                height=self.img_height,
                latent=True,
            )
            if image is None:
                continue

            image = utils.pad_image(image, width=self.img_width, height=self.img_height)
            if self.transforms is not None:
                image = self.transforms(image)

            label = self.tokenizer.encode(label)
            label = utils.fill_text(label, max_len=self.max_text_len)

            if label is None:
                continue

            dataset.append({"writer": writer_id, "image": image, "label": label})

            if writer_id not in self.__map_writer_id.keys():
                self.__map_writer_id[writer_id] = len(self.__map_writer_id)

            if self.max_files and len(dataset) >= self.max_files:
                self.__dataset = dataset
                return

        self.__dataset = dataset

    @property
    def config(self) -> ConfigLatentDiffusion:
        return self.__config

    @property
    def dataset(self) -> list[dict[str, Any]]:
        return self.__dataset

    @property
    def map_writer_id(self) -> dict[str, int]:
        return self.__map_writer_id

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        w_index = self.dataset[index]["writer"]

        writer_id = torch.tensor(self.map_writer_id[w_index], dtype=torch.int32)
        image = self.dataset[index]["image"]
        label = torch.tensor(self.dataset[index]["label"], dtype=torch.long)

        return writer_id, image, label
