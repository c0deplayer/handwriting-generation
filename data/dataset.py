from pathlib import Path
from typing import Any, Literal, Optional, Type

import lightning as L
import torch
from rich.progress import track
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import v2

from configs.config import (
    BaseConfig,
    ConfigConvNeXt,
    ConfigDiffusion,
    ConfigInception,
    ConfigLatentDiffusion,
)
from data.tokenizer import Tokenizer
from models.Diffusion.text_style import StyleExtractor
from utils import data_utils
from utils.utils import get_device


class DataModule(L.LightningDataModule):
    """
    This DataModule class is designed to handle dataset management and data loading for PyTorch Lightning-based
    deep learning applications. It provides train and validation data loaders, allowing you to separate the
    training and validation data seamlessly.

    Args:
        dataset (Type[Dataset]): The dataset class to be used for loading data.
        config (BaseConfig): A configuration object specifying dataset parameters and other relevant settings.
        use_gpu (bool): Whether to use GPU for data loading.

    Attributes:
        train_dataset (Dataset): The training dataset, initially set to None.
        val_dataset (Dataset): The validation dataset, initially set to None.
        dataset (Type[Dataset]): The dataset class used for loading data.
        __config (BaseConfig): The configuration object specifying dataset parameters and settings.
        batch_size (int): The batch size used for data loading.
        max_text_len (int): The maximum length of text data.
        max_files (int): The maximum number of data files.
        img_height (int): The height of image data.
        img_width (int): The width of image data.
        max_seq_len (int): The maximum sequence length (if applicable).
        train_size (float): The proportion of data used for training.
        val_size (float): The proportion of data used for validation.
        use_gpu (bool): Whether to use GPU for data loading.

    Methods:
        setup(stage: str) -> None:
            Prepare the dataset for training and validation.

        train_dataloader() -> DataLoader:
            Return a DataLoader for the training dataset.

        val_dataloader() -> DataLoader:
            Return a DataLoader for the validation dataset.
    """

    def __init__(
        self,
        dataset: Type[Dataset],
        config: BaseConfig,
        use_gpu: bool = False,
    ) -> None:
        super().__init__()

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
        self.max_seq_len = self.max_seq_len - (self.max_seq_len % 8) + 8
        self.use_gpu = use_gpu

    def setup(self, stage: str) -> None:
        """
        Prepare the dataset for training and validation.

        Args:
            stage (str): The stage of the training process ('fit', 'validate', or 'test').
        """
        if stage == "fit":
            if isinstance(
                self.__config, (ConfigDiffusion, ConfigLatentDiffusion)
            ):
                self.train_dataset = self._create_dataset(
                    "train", self.train_size
                )
                self.val_dataset = self._create_dataset("val", self.val_size)
            else:
                dataset = self._create_dataset("train", self.train_size)
                train_length = int(self.train_size * len(dataset))
                val_length = len(dataset) - train_length

                self.train_dataset, self.val_dataset = random_split(
                    dataset, [train_length, val_length]
                )

    def _create_dataset(self, dataset_type: str, size: float) -> Dataset:
        """
        Create a dataset for the specified type and size.

        Args:
            dataset_type (str): The type of dataset ('train' or 'val').
            size (float): The proportion of data to use for the dataset.

        Returns:
            Dataset: The created dataset.
        """
        kwargs_dataset = {
            "config": self.__config,
            "img_height": self.img_height,
            "img_width": self.img_width,
            "max_text_len": self.max_text_len,
            "max_files": int(size * self.max_files),
            "dataset_type": dataset_type,
            "use_gpu": self.use_gpu,
        }
        if isinstance(
            self.__config, (ConfigDiffusion, ConfigConvNeXt, ConfigInception)
        ):
            kwargs_dataset["max_seq_len"] = self.max_seq_len

        return self.dataset(**kwargs_dataset)

    def train_dataloader(self) -> DataLoader:
        """
        Return a DataLoader for the training dataset.

        Returns:
            DataLoader: The DataLoader for the training dataset.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        """
        Return a DataLoader for the validation dataset.

        Returns:
            DataLoader: The DataLoader for the validation dataset.
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
        )


class IAMonDataset(Dataset):
    """
    Dataset class for loading and preprocessing data from the IAM Online handwriting recognition dataset.

    Args:
        img_height (int): The height of images in the dataset.
        img_width (int): The width of images in the dataset.
        max_text_len (int): The maximum allowable length of text data.
        max_seq_len (int): The maximum sequence length (if applicable).
        max_files (int): The maximum number of data files to load.
        config (ConfigDiffusion): A configuration object specifying dataset parameters and settings.
        dataset_type (Literal["train", "val", "test"]): The type of dataset to load ("train," "val," or "test").
        use_gpu (bool): Whether to use GPU for data loading.
        strict (bool, optional): Whether to enforce strict vocabulary constraints (default: False).
        inception (bool, optional): Whether to use inception-style preprocessing (default: False).
    """

    def __init__(
        self,
        img_height: int,
        img_width: int,
        max_text_len: int,
        max_seq_len: int,
        max_files: int,
        config: ConfigDiffusion,
        dataset_type: Literal["train", "val", "test"],
        *,
        use_gpu: bool = False,
        strict: bool = False,
        inception: bool = False,
    ) -> None:
        super().__init__()

        self.config = config
        self.strict = strict
        self.inception = inception
        self.img_height = img_height
        self.img_width = img_width
        self.max_seq_len = (
            max_seq_len if max_seq_len > 0 else self.__get_max_seq_len()
        )
        self.max_text_len = max_text_len
        self.max_files = max_files
        self.transforms = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ])

        self.style_extractor = StyleExtractor(
            device=torch.device(get_device() if use_gpu else "cpu")
        )
        self.tokenizer = Tokenizer(config.vocab)
        self.dataset_type = dataset_type
        self.dataset_txt = self.__load_dataset_txt()
        self.dataset, self.map_writer_id = self.__load_dataset()

        print(
            f"Size of dataset: {len(self.dataset)} || Length of writer styles -- {len(self.map_writer_id)}"
        )

    def __load_dataset_txt(self) -> list[str]:
        type_dict = {
            "train": "iamondb_tr_va1.filter",
            "val": "iamondb_va2.filter",
            "test": "iamondb_test.filter",
        }
        dataset_file = (
            Path(self.config.data_path) / type_dict[self.dataset_type]
        )
        with dataset_file.open("r") as f:
            return f.readlines()

    def __get_max_seq_len(self) -> int:
        return data_utils.get_max_seq_len(
            Path(f"{self.config.data_path}/lineStrokes")
        )

    def __load_dataset(
        self,
    ) -> tuple[list[dict[str, Any]], Optional[dict[str, int]]]:
        h5_file_path = Path(f"./data/h5_dataset/{self.dataset_type}_iamondb.h5")
        json_file_path = Path(
            f"./data/json_writer_ids/{self.dataset_type}_writer_ids_iamondb.json"
        )

        if h5_file_path.is_file() and not self.strict:
            return data_utils.load_dataset(
                (h5_file_path, json_file_path), self.max_files
            )
        else:
            return self.__preprocess_data()

    def __preprocess_data(self) -> tuple[list[dict[str, Any]], dict[str, int]]:
        dataset, map_writer_id = [], {}
        raw_data_path = Path(self.config.data_path).resolve()
        ascii_path = raw_data_path / "ascii"
        strokes_path = raw_data_path / "lineStrokes"
        img_path = raw_data_path / "lineImages"
        original_path = raw_data_path / "original-xml"

        for line in track(
            self.dataset_txt,
            description=f"[cyan]Preparing {self.dataset_type} IAM Online DataLoader...",
        ):
            idx = line.strip()
            writer_id, transcription = self.__get_writer_and_transcription(
                idx, ascii_path, original_path
            )

            for file, raw_text in transcription.items():
                if not self.__is_valid_text(raw_text):
                    continue

                strokes, image, style = self.__get_strokes_image_style(
                    idx, file, strokes_path, img_path
                )
                if strokes is None or image is None:
                    continue

                one_hot, text = (
                    data_utils.get_encoded_text_with_one_hot_encoding(
                        raw_text, self.tokenizer, self.max_text_len
                    )
                )

                dataset.append({
                    "writer": writer_id,
                    "file": file,
                    "raw_text": raw_text,
                    "strokes": strokes,
                    "text": text,
                    "one_hot": one_hot,
                    "image": image,
                    "style": style,
                })

                map_writer_id.setdefault(writer_id, len(map_writer_id))

                if self.max_files and len(dataset) >= self.max_files:
                    return dataset, map_writer_id

        self.style_extractor = self.style_extractor.to(device="cpu")
        return dataset, map_writer_id

    def __get_writer_and_transcription(
        self, idx: str, ascii_path: Path, original_path: Path
    ) -> tuple[int, dict[str, str]]:
        path_txt = ascii_path / f"{idx[:3]}/{idx[:7]}/{idx}.txt"
        path_file_original_xml = original_path / f"{idx[:3]}/{idx[:7]}"
        writer_id = data_utils.get_writer_id(path_file_original_xml, idx)
        transcription = data_utils.get_transcription(path_txt)

        return writer_id, transcription

    def __is_valid_text(self, text: str) -> bool:
        return len(text) <= self.max_text_len and (
            not self.strict or all(c in self.config.vocab for c in text)
        )

    def __get_strokes_image_style(
        self, idx: str, file: str, strokes_path: Path, img_path: Path
    ) -> tuple[Any, Any, Any]:
        path_file_xml = strokes_path / f"{idx[:3]}/{idx[:7]}/{file}.xml"
        path_file_tif = img_path / f"{idx[:3]}/{idx[:7]}/{file}.tif"

        strokes = data_utils.get_line_strokes(path_file_xml, self.max_seq_len)
        image = data_utils.get_image(
            path_file_tif, self.img_width, self.img_height, centering=(0.0, 0.5)
        )

        if image is not None:
            writer_image = (
                v2.PILToTensor()(image).unsqueeze(0).to(torch.float32)
            )
            with torch.inference_mode():
                style = self.style_extractor(writer_image).squeeze(0)
        else:
            style = None

        return strokes, image, style

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[Tensor, ...]:
        item = self.dataset[index]
        if isinstance(self.config, (ConfigConvNeXt, ConfigInception)):
            image = self.transforms(item["image"])
            writer_id = torch.tensor(
                self.map_writer_id[item["writer"]], dtype=torch.int32
            )
            text = torch.tensor(item["text"])
            return writer_id, image, text
        else:
            strokes = torch.tensor(item["strokes"], dtype=torch.float32)
            text = torch.tensor(item["text"])
            style = item["style"]
            image = self.transforms(item["image"])
            return strokes, text, style, image


class IAMDataset(Dataset):
    """
    Handles the IAM Database handwriting dataset, providing methods for loading, preprocessing, and retrieving data samples.

    Args:
        config (ConfigLatentDiffusion): Configuration object specifying dataset parameters and settings.
        img_height (int): The height of the images in the dataset.
        img_width (int): The width of the images in the dataset.
        max_text_len (int): The maximum length of the text data.
        max_files (int): The maximum number of files to load.
        dataset_type (Literal["train", "val", "test"]): Type of the dataset - train, validation, or test.
        strict (bool, optional): Whether to apply strict constraints on data filtering. Defaults to False.
    """

    def __init__(
        self,
        config: ConfigLatentDiffusion,
        img_height: int,
        img_width: int,
        max_text_len: int,
        max_files: int,
        dataset_type: Literal["train", "val", "test"],
        *,
        strict: bool = False,
    ) -> None:
        self.config = config
        self.strict = strict
        self.max_text_len = max_text_len
        self.max_files = max_files
        self.img_height = img_height
        self.img_width = img_width
        self.dataset_type = dataset_type

        self.transforms = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        self.tokenizer = Tokenizer(config.vocab)
        self.dataset_txt = self.__load_dataset_txt()
        self.dataset, self.map_writer_id = self.__load_data()

        print(
            f"Size of dataset: {len(self.dataset)} || Length of writer styles -- {len(self.map_writer_id)}"
        )

    def __load_dataset_txt(self) -> list[str]:
        type_dict = {
            "train": "iam_tr_va1.filter",
            "val": "iam_va2.filter",
            "test": "iam_test.filter",
        }
        dataset_file = (
            Path(self.config.data_path) / type_dict[self.dataset_type]
        )
        with dataset_file.open("r") as f:
            return f.readlines()

    def __load_data(
        self,
    ) -> tuple[list[dict[str, Any]], Optional[dict[str, int]]]:
        h5_file_path = Path(f"./data/h5_dataset/{self.dataset_type}_iamdb.h5")
        json_file_path = Path(
            f"./data/json_writer_ids/{self.dataset_type}_writer_ids_iamdb.json"
        )

        if (
            h5_file_path.is_file()
            and json_file_path.is_file()
            and not self.strict
        ):
            return data_utils.load_dataset(
                (h5_file_path, json_file_path), self.max_files, latent=True
            )
        else:
            return self.__preprocess_data()

    def __preprocess_data(self) -> tuple[list[dict[str, Any]], dict[str, int]]:
        dataset, map_writer_id = [], {}
        raw_data_path = Path(self.config.data_path)

        for line in track(
            self.dataset_txt,
            description=f"Preparing {self.dataset_type} IAM Dataloader...",
        ):
            writer_id, image_id, label = self.__parse_line(line)
            if not self.__is_valid_label(label):
                continue

            img_path = self.__get_image_path(raw_data_path, image_id)
            image = data_utils.get_image(
                img_path,
                self.img_width,
                self.img_height,
                latent=True,
                centering=(0.5, 0.5),
            )

            if image is None:
                continue

            image = self.transforms(image)
            _, encoded_label = (
                data_utils.get_encoded_text_with_one_hot_encoding(
                    label, self.tokenizer, self.max_text_len
                )
            )

            dataset.append({
                "writer": writer_id,
                "image": image,
                "label": encoded_label,
            })
            map_writer_id.setdefault(writer_id, len(map_writer_id))

            if self.max_files and len(dataset) >= self.max_files:
                break

        return dataset, map_writer_id

    def __parse_line(self, line: str) -> tuple[str, str, str]:
        parts = line.split(" ")
        writer_id, image_id = parts[0].split(",")
        label = parts[1].rstrip()

        return writer_id, image_id, label

    def __is_valid_label(self, label: str) -> bool:
        return len(label) <= self.max_text_len and (
            not self.strict or all(c in self.config.vocab for c in label)
        )

    def __get_image_path(self, raw_data_path: Path, image_id: str) -> Path:
        image_parts = image_id.split("-")
        f_folder, s_folder = (
            image_parts[0],
            f"{image_parts[0]}-{image_parts[1]}",
        )
        return raw_data_path / f"words/{f_folder}/{s_folder}/{image_id}.png"

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        item = self.dataset[index]
        writer_id = torch.tensor(
            self.map_writer_id[item["writer"]], dtype=torch.int32
        )
        image = item["image"]
        label = torch.tensor(item["label"], dtype=torch.long)
        return writer_id, image, label
