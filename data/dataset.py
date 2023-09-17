import os
import random
from pathlib import Path
from typing import Any, Literal, Union, Dict, List, Tuple

import lightning.pytorch as pl
import torch
import torchvision.transforms
from einops import rearrange
from rich.progress import track
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, random_split

from configs.config import ConfigDiffusion, ConfigLatentDiffusion, ConfigRNN
from models.Diffusion.text_style import StyleExtractor
from . import utils
from .tokenizer import Tokenizer


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset: Dataset,
        config: Union[ConfigDiffusion, ConfigRNN, ConfigLatentDiffusion],
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

        if not isinstance(config, (ConfigDiffusion, ConfigRNN, ConfigLatentDiffusion)):
            raise TypeError(
                "Expected config to be ConfigDiffusion, ConfigRNN or ConfigLatentDiffusion, "
                f"got {type(config).__name__}"
            )

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.dataset = dataset
        self.__config = config
        self.batch_size = config.batch_size
        self.max_text_len = config.max_text_len
        self.max_files = config.max_files
        self.img_height = config.get("img_height", 90)
        self.img_width = config.get("img_width", 1400)
        self.max_seq_len = config.get("max_seq_len", 0)
        self.train_size = config.get("train_size", 0.85)
        self.val_size = 1.0 - self.train_size

    # noinspection PyCallingNonCallable
    def setup(self, stage: str) -> None:
        if stage == "fit":
            if isinstance(self.__config, ConfigLatentDiffusion):
                self.train_dataset = self.dataset(
                    config=self.__config,
                    img_height=self.img_height,
                    img_width=self.img_width,
                    max_text_len=self.max_text_len,
                    max_files=self.train_size * self.max_files,
                    dataset_type="train",
                )
                self.val_dataset = self.dataset(
                    config=self.__config,
                    img_height=self.img_height,
                    img_width=self.img_width,
                    max_text_len=self.max_text_len,
                    max_files=self.val_size * self.max_files,
                    dataset_type="val",
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


class DummyDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()

    def __load_dataset(self) -> None:
        pass

    def preprocess_data(self) -> List[Dict[str, Any]]:
        pass

    def __getitem__(self, index: int) -> Tuple[Any, ...]:
        pass

    def __len__(self) -> int:
        pass


class IAMonDataset(Dataset):
    def __init__(
        self,
        img_height: int,
        img_width: int,
        max_text_len: int,
        max_seq_len: int,
        max_files: int,
        config: Union[ConfigDiffusion, ConfigRNN],
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
        config : Union[ConfigDiffusion, ConfigRNN]
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

        self.__load_dataset__()
        print(f"Size of dataset: {len(self.dataset)}")

    def __load_dataset__(self) -> None:
        h5_file_path = Path("data/h5_dataset/train_val_iamondb.h5")

        if h5_file_path.is_file():
            self.__dataset, _ = utils.load_dataset_from_h5(h5_file_path, self.max_files)
        else:
            self.__dataset = self.preprocess_data()

    def preprocess_data(self) -> List[Dict[str, Any]]:
        """
        _summary_

        Returns
        -------
        List[Dict[str, Any]]
            _description_
        """

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
                if len(raw_text) > self.max_text_len:
                    shuffled_images.pop(0)
                    continue

                path_file_xml = strokes_path / f"{idx[:3]}/{idx[:7]}/{file}.xml"
                path_file_tif = path_images / shuffled_images.pop(0)

                strokes = utils.get_line_strokes(path_file_xml, self.max_seq_len)

                one_hot, text = utils.get_encoded_text_with_one_hot_encoding(
                    raw_text, self.tokenizer, self.max_text_len
                )

                image = utils.get_image(path_file_tif, self.img_width, self.img_height)

                if strokes is None or image is None:
                    continue

                writer_image = torchvision.transforms.PILToTensor()(image).to(
                    torch.float32
                )
                writer_image = rearrange(writer_image, "1 h w -> 1 1 h w")
                with torch.no_grad():
                    style = self.style_extractor(writer_image)

                dataset.append(
                    {
                        "writer": writer_id,
                        "file": file,
                        "raw_text": raw_text,
                        "strokes": strokes,
                        "text": text,
                        "one_hot": one_hot,
                        "image": image,
                        "style": style,
                    }
                )

                if self.max_files and len(dataset) >= self.max_files:
                    return dataset

        return dataset

    @property
    def config(self) -> Union[ConfigDiffusion, ConfigRNN]:
        return self.__config

    @property
    def dataset(self) -> List[Dict[str, Any]]:
        return self.__dataset

    def __getitem__(self, index: int) -> Tuple[Tensor, ...]:
        if self.diffusion:
            strokes = torch.tensor(self.dataset[index]["strokes"], dtype=torch.float32)
            text = torch.tensor(self.dataset[index]["text"])
            style = self.dataset[index]["style"]

            return strokes, text, style

        else:
            strokes = torch.tensor(self.dataset[index]["strokes"], dtype=torch.float32)
            text = torch.tensor(self.dataset[index]["one_hot"], dtype=torch.float32)

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
    ) -> None:
        self.__config = config
        self.max_text_len = max_text_len
        self.max_files = max_files
        self.transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
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

        self.__load_data__()
        print(f"Length of writer styles -- {len(self.map_writer_id)}")

    def __load_data__(self) -> None:
        h5_file_path = Path(f"data/h5_dataset/{self.dataset_type}_iamdb.h5")

        if h5_file_path.is_file():
            self.__dataset, self.__map_writer_id = utils.load_dataset_from_h5(
                h5_file_path, self.max_files, latent=True
            )
        else:
            self.__dataset, self.__map_writer_id = self.preprocess_data()

    def preprocess_data(self) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        """
        _summary_

        Returns
        -------
        Tuple[List[Dict[str, Any]], Dict[str, int]]
            _description_
        """

        dataset, map_writer_id = [], {}
        raw_data_path = Path(self.config.data_path)

        for line in track(
            self.dataset_txt,
            description=f"Preparing {self.dataset_type} IAM Dataloader...",
        ):
            parts = line.split(" ")
            writer_id, image_id = parts[0].split(",")[0], parts[0].split(",")[1]
            label = parts[1].rstrip()

            if len(label) > self.max_text_len:
                continue

            image_parts = image_id.split("-")
            f_folder, s_folder = (
                image_parts[0],
                f"{image_parts[0]}-{image_parts[1]}",
            )

            img_path = raw_data_path / f"words/{f_folder}/{s_folder}/{image_id}.png"

            image = utils.get_image(
                img_path, self.img_width, self.img_height, latent=True
            )

            if image is None:
                continue

            if self.transforms is not None:
                image = self.transforms(image)

            _, label = utils.get_encoded_text_with_one_hot_encoding(
                label, self.tokenizer, self.max_text_len
            )

            dataset.append({"writer": writer_id, "image": image, "label": label})

            if writer_id not in map_writer_id.keys():
                map_writer_id[writer_id] = len(map_writer_id)

            if self.max_files and len(dataset) >= self.max_files:
                return dataset, map_writer_id

        return dataset, map_writer_id

    @property
    def config(self) -> ConfigLatentDiffusion:
        return self.__config

    @property
    def dataset(self) -> List[Dict[str, Any]]:
        return self.__dataset

    @property
    def map_writer_id(self) -> Dict[str, int]:
        return self.__map_writer_id

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, Tensor]:
        w_index = self.dataset[index]["writer"]

        writer_id = torch.tensor(self.map_writer_id[w_index], dtype=torch.int32)
        image = self.dataset[index]["image"]
        label = torch.tensor(self.dataset[index]["label"], dtype=torch.long)

        return writer_id, image, label
