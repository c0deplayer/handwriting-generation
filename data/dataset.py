import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Tuple, Union

import lightning.pytorch as pl
import torch
import torchvision.transforms.v2 as transforms
from einops import rearrange
from rich.progress import track
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from configs.config import ConfigDiffusion, ConfigLatentDiffusion
from models.Diffusion.text_style import StyleExtractor
from . import utils
from .tokenizer import Tokenizer


class DataModule(pl.LightningDataModule):
    """
    This DataModule class is designed to handle dataset management and data loading for PyTorch Lightning-based
    deep learning applications. It provides train and validation data loaders, allowing you to separate the
    training and validation data seamlessly.

    Args:
        dataset (Dataset): The dataset class to be used for loading data.
        config (Union[ConfigDiffusion, ConfigLatentDiffusion]): A configuration object specifying
            dataset parameters and other relevant settings.

    Raises:
        TypeError: If the `config` object is not an instance of `ConfigDiffusion`,
                   or `ConfigLatentDiffusion`.

    Attributes:
        train_dataset (DataLoader): The training dataset, initially set to None.
        val_dataset (DataLoader): The validation dataset, initially set to None.
        dataset (Dataset): The dataset class used for loading data.
        __config (Union[ConfigDiffusion, ConfigLatentDiffusion]):
                The configuration object specifying dataset parameters and settings.
        batch_size (int): The batch size used for data loading.
        max_text_len (int): The maximum length of text data.
        max_files (int): The maximum number of data files.
        img_height (int): The height of image data.
        img_width (int): The width of image data.
        max_seq_len (int): The maximum sequence length (if applicable).
        train_size (float): The proportion of data used for training.
        val_size (float): The proportion of data used for validation.

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
        dataset: Dataset,
        config: Union[ConfigDiffusion, ConfigLatentDiffusion],
    ) -> None:
        super().__init__()

        if not isinstance(config, (ConfigDiffusion, ConfigLatentDiffusion)):
            raise TypeError(
                "Expected config to be ConfigDiffusion, or ConfigLatentDiffusion, "
                f"got {type(config).__name__}"
            )

        self.train_dataset = None
        self.val_dataset = None

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

    # noinspection PyCallingNonCallable
    def setup(self, stage: str) -> None:
        if stage == "fit":
            kwargs_dataset = dict(
                config=self.__config,
                img_height=self.img_height,
                img_width=self.img_width,
                max_text_len=self.max_text_len,
                max_files=self.train_size * self.max_files,
                dataset_type="train",
            )
            if isinstance(self.__config, ConfigDiffusion):
                kwargs_dataset["max_seq_len"] = self.max_seq_len

            self.train_dataset = self.dataset(**kwargs_dataset)

            kwargs_dataset["max_files"] = self.val_size * self.max_files
            kwargs_dataset["dataset_type"] = "val"
            self.val_dataset = self.dataset(**kwargs_dataset)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=os.cpu_count() // 4,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=os.cpu_count() // 4,
            pin_memory=True,
        )


class DummyDataset(Dataset):
    """
    This `DummyDataset` class is intended to be used as a template for creating custom dataset classes in PyTorch.
    It provides placeholder methods and serves as a starting point for implementing your own dataset.

    Args:
        None

    Methods:
        __init__(self) -> None:
            Initialize a new instance of the `DummyDataset` class.

        __load_dataset__(self) -> None:
            Placeholder method for loading the dataset. Override this method to load your own data.

        preprocess_data(self) -> List[Dict[str, Any]]:
            Placeholder method for data preprocessing. Override this method to preprocess your dataset.

        __getitem__(self, index: int) -> Tuple[Any, ...]:
            Placeholder method for retrieving an item from the dataset.
            Override this method to define how data is retrieved.

        __len__(self) -> int:
            Placeholder method for getting the length of the dataset.
            Override this method to specify the dataset's length.
    """

    def __init__(self) -> None:
        super().__init__()

    def __load_dataset__(self) -> None:
        pass

    def preprocess_data(self) -> List[Dict[str, Any]]:
        pass

    def __getitem__(self, index: int) -> Tuple[Any, ...]:
        pass

    def __len__(self) -> int:
        pass


class IAMonDataset(Dataset):
    """
    This dataset class is designed for loading and preprocessing data from the IAM Online handwriting recognition dataset.
    It is intended to be used with PyTorch data loaders for training, validation, and testing purposes.

    Args:
        img_height (int): The height of images in the dataset.
        img_width (int): The width of images in the dataset.
        max_text_len (int): The maximum allowable length of text data.
        max_seq_len (int): The maximum sequence length (if applicable).
        max_files (int): The maximum number of data files to load.
        config (ConfigDiffusion): A configuration object specifying dataset parameters and settings.
        dataset_type (Literal["train", "val", "test"]): The type of dataset to load ("train," "val," or "test").
        strict (bool, optional): Whether to enforce strict vocabulary constraints (default: False).

    Attributes:
        img_height (int): The height of the images in the dataset.
        img_width (int): The width of the images in the dataset.
        max_seq_len (int): The maximum sequence length of the data.
        max_text_len (int): The maximum length of the text data.
        max_files (int): The maximum number of files to load.
        style_extractor (StyleExtractor): Object for extracting style features.
        tokenizer (Tokenizer): Tokenizer for text encoding.
        dataset_type (Literal["train", "val", "test"]): Type of the dataset.
        dataset_txt (List[str]): List of dataset filenames.

    Methods:
        __load_dataset__(self) -> None:
            Load the dataset from an HDF5 file or preprocess it if necessary.

        preprocess_data(self) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
            Preprocess the dataset by loading and formatting data from the raw IAM Online dataset.

        __getitem__(self, index: int) -> Tuple[Tensor, ...]:
            Get a data item from the dataset.

        __len__(self) -> int:
            Get the length of the dataset.

    Properties:
        config (ConfigDiffusion): The configuration object.
        dataset (List[Dict[str, Any]]): The loaded dataset.
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
        strict: bool = False,
        inception: bool = False,
    ) -> None:
        super().__init__()

        self.__config = config
        self._strict = strict
        self._inception = inception
        self.img_height = img_height
        self.img_width = img_width
        self.max_seq_len = (
            max_seq_len
            if max_seq_len > 0
            else utils.get_max_seq_len(Path(f"{config.data_path}/lineStrokes"))
        )
        self.max_text_len = max_text_len
        self.max_files = max_files
        self._diffusion = isinstance(config, ConfigDiffusion)
        self.transforms = transforms.Compose(
            [transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)]
        )

        self.style_extractor = StyleExtractor(device=torch.device(config.device))
        self.tokenizer = Tokenizer(config.vocab)

        self.dataset_type = dataset_type

        type_dict = {
            "train": "iamondb_tr_va1.filter",
            "val": "iamondb_va2.filter",
            "test": "iamondb_test.filter",
        }

        with open(f"{config.data_path}/{type_dict[dataset_type]}", mode="r") as f:
            self.dataset_txt = f.readlines()

        self.__load_dataset__()
        print(
            f"Size of dataset: {len(self.dataset)} || Length of writer styles -- {len(self.map_writer_id)}"
        )

        # if not self.diffusion:
        #     self._mean = utils.compute_mean(self.dataset)
        #     self._std = utils.compute_std(self.dataset)

    def __load_dataset__(self) -> None:
        h5_file_path = Path(f"./data/h5_dataset/{self.dataset_type}_iamondb.h5")
        json_file_path = Path(
            f"./data/json_writer_ids/{self.dataset_type}_writer_ids_iamondb.json"
        )
        if h5_file_path.is_file() and not self._strict:
            self.__dataset, self.__map_writer_id = utils.load_dataset(
                (h5_file_path, json_file_path), self.max_files
            )
        else:
            self.__dataset, self.__map_writer_id = self.preprocess_data()

    def preprocess_data(self) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        dataset, map_writer_id = [], {}
        raw_data_path = Path(self.config.data_path)
        ascii_path = raw_data_path / "ascii"
        strokes_path = raw_data_path / "lineStrokes"
        img_path = raw_data_path / "lineImages"
        original_path = raw_data_path / "original-xml"

        for line in track(
            self.dataset_txt,
            description=f"Preparing {self.dataset_type} IAM Online DataLoader...",
        ):
            idx = line.strip()
            path_txt = ascii_path / f"{idx[:3]}/{idx[:7]}/{idx}.txt"
            path_file_original_xml = original_path / f"{idx[:3]}/{idx[:7]}"

            writer_id = utils.get_writer_id(path_file_original_xml, idx)

            transcription = utils.get_transcription(path_txt)

            for file, raw_text in transcription.items():
                if len(raw_text) > self.max_text_len:
                    continue

                if self._strict and not all(c in self.config.vocab for c in raw_text):
                    continue

                path_file_xml = strokes_path / f"{idx[:3]}/{idx[:7]}/{file}.xml"
                path_file_tif = img_path / f"{idx[:3]}/{idx[:7]}/{file}.tif"

                strokes = utils.get_line_strokes(path_file_xml, self.max_seq_len)

                one_hot, text = utils.get_encoded_text_with_one_hot_encoding(
                    raw_text, self.tokenizer, self.max_text_len
                )

                image = utils.get_image(
                    path_file_tif, self.img_width, self.img_height, centering=(0.0, 0.5)
                )

                if strokes is None or image is None:
                    continue

                writer_image = transforms.PILToTensor()(image).to(torch.float32)
                writer_image = rearrange(writer_image, "1 h w -> 1 1 h w")
                with torch.inference_mode():
                    style = rearrange(
                        self.style_extractor(writer_image), "1 h w -> h w"
                    )

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

                if writer_id not in map_writer_id.keys():
                    map_writer_id[writer_id] = len(map_writer_id)

                if self.max_files and len(dataset) >= self.max_files:
                    return dataset, map_writer_id

        return dataset, map_writer_id

    @property
    def config(self) -> ConfigDiffusion:
        return self.__config

    @property
    def dataset(self) -> List[Dict[str, Any]]:
        return self.__dataset

    @property
    def map_writer_id(self) -> Dict[str, int]:
        return self.__map_writer_id

    # @property
    # def mean(self) -> Tensor:
    #     return self._mean
    #
    # @property
    # def std(self) -> Tensor:
    #     return self._std
    #
    # def normalize(self, strokes: Tensor) -> Tensor:
    #     return (strokes - self.mean) / self.std
    #
    # def denormalize(self, strokes: Tensor) -> Tensor:
    #     return strokes * self.std + self.mean

    def __getitem__(self, index: int) -> Tuple[Tensor, ...]:
        if self._inception:
            image = self.transforms(self.dataset[index]["image"])
            w_index = str(self.dataset[index]["writer"])

            writer_id = torch.tensor(self.map_writer_id[w_index], dtype=torch.int32)
            text = torch.tensor(self.dataset[index]["text"])

            return writer_id, image, text
        elif self._diffusion:
            strokes = torch.tensor(self.dataset[index]["strokes"], dtype=torch.float32)
            text = torch.tensor(self.dataset[index]["text"])
            style = self.dataset[index]["style"]
            image = self.transforms(self.dataset[index]["image"])

            return strokes, text, style, image
        else:
            strokes = torch.tensor(self.dataset[index]["strokes"], dtype=torch.float32)
            text = torch.tensor(self.dataset[index]["one_hot"], dtype=torch.float32)

            # strokes[1:, :] = self.normalize(strokes[1:, :])

            return strokes, text

    def __len__(self) -> int:
        return len(self.dataset)


class IAMDataset(Dataset):
    """
    This class is designed to handle the IAM Database handwriting dataset. It provides methods for loading
    and preprocessing the data, and for retrieving data samples for training or validation.

    Args:
        config (ConfigLatentDiffusion): Configuration object specifying dataset parameters and settings.
        img_height (int): The height of the images in the dataset.
        img_width (int): The width of the images in the dataset.
        max_text_len (int): The maximum length of the text data.
        max_files (int): The maximum number of files to load.
        dataset_type (Literal["train", "val", "test"]): Type of the dataset - train, validation, or test.
        strict (bool, optional): Whether to apply strict constraints on data filtering. Defaults to False.

    Attributes:
        img_height (int): The height of the images in the dataset.
        img_width (int): The width of the images in the dataset.
        max_text_len (int): The maximum length of the text data.
        max_files (int): The maximum number of files to load.
        transforms (torchvision.transforms.Compose): Image transformations for data preprocessing.
        dataset_type (Literal["train", "val", "test"]): Type of the dataset.
        tokenizer (Tokenizer): Tokenizer for text encoding.
        dataset_txt (List[str]): List of dataset filenames.

    Properties:
        config (ConfigLatentDiffusion): The configuration object.
        dataset (List[Dict[str, Any]]): The loaded dataset.
        map_writer_id (Dict[str, int]): A mapping of writer IDs to integer indices.

    Methods:
        __init__(...): Initializes an instance of the IAMDataset class.
        __load_data__() -> None: Loads the dataset into memory.
        preprocess_data() -> Tuple[List[Dict[str, Any]], Dict[str, int]]: Preprocesses the data.
        __getitem__(index: int) -> Tuple[Tensor, Tensor, Tensor]: Retrieves a specific item from the dataset.
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
        self.__config = config
        self._strict = strict
        self.max_text_len = max_text_len
        self.max_files = max_files
        self.transforms = transforms.Compose(
            [
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
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
        print(
            f"Size of dataset: {len(self.dataset)} || Length of writer styles -- {len(self.map_writer_id)}"
        )

    def __load_data__(self) -> None:
        h5_file_path = Path(f"./data/h5_dataset/{self.dataset_type}_iamdb.h5")
        json_file_path = Path(
            f"./data/json_writer_ids/{self.dataset_type}_writer_ids_iamdb.json"
        )

        if h5_file_path.is_file() and json_file_path.is_file() and not self._strict:
            self.__dataset, self.__map_writer_id = utils.load_dataset(
                (h5_file_path, json_file_path), self.max_files, latent=True
            )
        else:
            self.__dataset, self.__map_writer_id = self.preprocess_data()

    def preprocess_data(self) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
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

            if self._strict and not all(c in self.config.vocab for c in label):
                continue

            image_parts = image_id.split("-")
            f_folder, s_folder = (
                image_parts[0],
                f"{image_parts[0]}-{image_parts[1]}",
            )

            img_path = raw_data_path / f"words/{f_folder}/{s_folder}/{image_id}.png"

            image = utils.get_image(
                img_path,
                self.img_width,
                self.img_height,
                latent=True,
                centering=(0.5, 0.5),
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
