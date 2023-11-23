import json
import os
import warnings
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
import PIL
import torch
import torch.nn.functional as F
from numpy.linalg import norm
from PIL import Image as ImageModule
from PIL import ImageOps
from PIL.Image import Image
from rich.progress import track
from torch import Tensor
from torchvision.transforms import Normalize

from .tokenizer import Tokenizer


def get_transcription(path: Path) -> Dict[str, str]:
    """
    This function reads transcription data from a file specified by the provided path. The file is expected
    to contain text data with specific formatting, where each line represents a transcription entry.

    Args:
        path (Path): The path to the file containing transcription data.

    Returns:
        Dict[str, str]: A dictionary where keys are composed of the filename stem and a numerical index,
        and values are the transcribed text for each entry.

    Example:
        If the file contains the following lines:
        ```
        CSR:
        Line 1
        Line 2
        Line 3
        ```

        The function would return:
        ```
        {
            'filename-01': 'Line 1',
            'filename-02': 'Line 2',
            'filename-03': 'Line 3'
        }
        ```

    """

    with path.open(mode="r") as f:
        transcription = f.readlines()

    result = map(str.rstrip, transcription)
    transcription = list(filter(None, list(result)))
    start_index = transcription.index("CSR:") + 1
    lines = transcription[start_index:]

    return {
        f"{path.stem}-{str(z).zfill(2)}": line for z, line in enumerate(lines, start=1)
    }


def get_line_strokes(path: Path, max_length: int) -> Optional[np.array]:
    """
    Get line strokes from an XML file.

    Args:
        path (Path): Path to the XML file.
        max_length (int): Maximum length of the stroke sequence.

    Returns:
        Optional[np.array]: An array containing line strokes if successful, or None if the maximum value of the stroke
        coordinates exceeds 15.
    """

    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as e:
        raise ET.ParseError(f"Failed to parse file {path}\n{str(e)}") from e

    stroke_set_tag = root.find("StrokeSet")
    if stroke_set_tag is None:
        raise ValueError("No StrokeSet element found in XML file")

    strokes_cord = []
    prev_cord = None

    for stroke in stroke_set_tag.findall("Stroke"):
        for point in stroke.findall("Point"):
            x, y, time = (
                float(point.attrib["x"]),
                float(point.attrib["y"]),
                float(point.attrib["time"]),
            )

            if prev_cord is not None:
                dx, dy = x - prev_cord[0], -y - prev_cord[1]
                strokes_cord.append([dx, dy, 0.0, time])

            prev_cord = [x, -y]

        if strokes_cord:
            strokes_cord[-1][2] = 1.0
        else:
            strokes_cord.append([prev_cord[0], prev_cord[1], 1.0, time])

    strokes = np.array(strokes_cord)
    strokes = strokes[np.argsort(strokes[:, 3])][:, :3]

    # if diffusion:
    strokes[:, :2] /= np.std(strokes[:, :2])

    for _ in range(3):
        strokes = __combine_strokes(strokes, int(len(strokes) * 0.20))

    if np.amax(np.abs(strokes)) > 15:
        return None
    # else:
    #     strokes = np.insert(strokes, 0, [0, 0, 0.0], axis=0)

    return __pad_strokes(strokes, max_length)


def __combine_strokes(strokes: np.ndarray, n: int) -> np.ndarray:
    """
    Given an array representing stroke segments, this function combines stroke pairs to reduce their count
    by a specified amount 'n'. It sorts stroke segments based on their combination potential and merges
    the most compatible strokes together.

    Args:
        strokes (np.ndarray): An array representing stroke segments, where each row represents a stroke segment.
        n (int): The number of stroke segments to be reduced.

    Returns:
        np.ndarray: The modified array after combining the strokes.

    Raises:
        UserWarning: If the number of strokes is odd, the last stroke will be ignored during the process.

    Note:
        This function modifies the input strokes array in place.

    """

    s, s_neighbors = strokes[::2, :2], strokes[1::2, :2]

    if len(strokes) % 2:
        warnings.warn(
            "The number of strokes is odd, so the last stroke will be ignored.",
            UserWarning,
        )
        s = s[:-1]

    values = (
        norm(s, axis=-1) + norm(s_neighbors, axis=-1) - norm(s + s_neighbors, axis=-1)
    )
    indices = np.argsort(values)[:n]

    strokes[indices * 2] += strokes[indices * 2 + 1]
    strokes[indices * 2, 2] = np.greater(strokes[indices * 2, 2], 0)
    strokes = np.delete(strokes, indices * 2 + 1, axis=0)
    strokes[:, :2] /= np.std(strokes[:, :2])

    if not strokes[-1, 2]:
        strokes[-1, 2] = 1.0

    return strokes


def get_image(
    path: Path,
    width: int,
    height: int,
    *,
    latent: bool = False,
    centering: Tuple[float, float] = (0.0, 0.0),
) -> Union[Image, None]:
    """
    This function loads an image from the specified file path and applies preprocessing based on the provided parameters.
    It can convert the image to grayscale (L mode) or RGB mode based on the 'latent' parameter and crop/resize the image
    to the specified dimensions.

    Args:
        path (Path): The path to the image file to be loaded.
        width (int): The desired width of the output image.
        height (int): The desired height of the output image.
        latent (bool, optional): If True, converts the image to RGB mode; otherwise, to grayscale (L mode).
                                 Defaults to False.
        centering (Tuple[float, float], optional): The centering position for resizing. Defaults to (0.0, 0.0).

    Returns:
        Union[Image, None]: The loaded and preprocessed image as a PIL Image object,
                            or None if the image cannot be loaded.

    Raises:
        FileNotFoundError: If the image file is not found at the specified path.

    Example:
        - To load and resize an RGB image:
        ```python
        image = get_image('path/to/image.jpg', width=200, height=150, latent=True)
        ```

        - To load and resize a grayscale image:
        ```python
        image = get_image('path/to/image.png', width=100, height=100)
        ```

    """

    try:
        img = ImageModule.open(path)
    except PIL.UnidentifiedImageError:
        return None
    except FileNotFoundError as e:
        raise f"The image was not found: {path}" from e

    if latent:
        img = img.convert("RGB")
    else:
        img = img.convert("L")

        bbox = ImageOps.invert(img).getbbox()
        img = img.crop(bbox)

    return ImageOps.pad(
        image=img,
        method=ImageModule.LANCZOS,
        size=(width, height),
        color="white",
        centering=centering,
    )


def get_encoded_text_with_one_hot_encoding(
    text: str, tokenizer: Tokenizer, max_len: int, *, pad_value: int = 0
) -> Tuple[np.array, np.array]:
    """
    This function encodes the input text using a provided tokenizer and converts it into one-hot encoding.
    It also ensures that the resulting encoding has a maximum length specified by 'max_len' by padding with
    the 'pad_value'.

    Args:
        text (str): The input text to be encoded.
        tokenizer (Tokenizer): The tokenizer used for character-to-token mapping.
        max_len (int): The desired maximum length of the encoding.
        pad_value (int, optional): The padding value to use when extending the encoding. Defaults to 0.

    Returns:
        Tuple[np.array, np.array]: A tuple containing the one-hot encoded text and the text encoding.

    Example:
        ```python
        text = "Hello"
        tokenizer = Tokenizer("abcdefghijklmnopqrstuvwxyz")
        max_len = 10
        pad_value = 0
        one_hot_encoding, text_encoding = get_encoded_text_with_one_hot_encoding(text, tokenizer, max_len, pad_value=pad_value)
        ```

    """

    text = tokenizer.encode(text)

    text_len = len(text)
    max_len += 2

    if text_len < max_len:
        padded_text = np.full((max_len - text_len,), pad_value)
        text = np.concatenate((text, padded_text))

    labels = torch.as_tensor(text)
    one_hot = F.one_hot(labels, num_classes=tokenizer.get_vocab_size())

    return one_hot.numpy(), text


def __pad_strokes(
    strokes: np.array, max_length: int, *, fill_value: float = 0
) -> Optional[np.array]:
    """
    This function takes a stroke array and either pads it with a specified fill value.

    Args:
        strokes (np.array): The input stroke array to be padded..
        max_length (int): The desired maximum length for the stroke array.
        fill_value (float, optional): The value to use for padding when extending the array. Defaults to 0.

    Returns:
        Optional[np.array]: The padded stroke array, or None if the input length exceeds the maximum length.
    """

    if max_length:
        stroke_len = len(strokes)

        if stroke_len > max_length:
            return None

        padded_strokes = np.full((max_length, 3), fill_value, dtype=np.float32)
        padded_strokes[:, 2] = 1.0
        padded_strokes[:stroke_len, :] = strokes[:, :]
    else:
        padded_strokes = strokes

    return padded_strokes


def get_writer_id(path: Path, idx: str) -> int:
    """
    This function searches for a writer ID associated with a specific data entry specified by 'idx' within
    a directory of XML files at the provided 'path'. It returns the writer ID if found, or 0 if no matching
    entry is located.

    Args:
        path (Path): The path to the directory containing XML files.
        idx (str): The identifier of the data entry to find the writer ID for.

    Returns:
        int: The writer ID associated with the specified data entry, or 0 if not found.

    Example:
        ```
        path_to_xml_dir = Path('path/to/xml_files')
        data_entry_id = '12345'
        writer_id = get_writer_id(path_to_xml_dir, data_entry_id)
        ```
    """

    writer_id = 0

    for file in path.glob("*.xml"):
        root = ET.parse(file).getroot()

        general_tag = root.find("General")
        if general_tag is None:
            continue

        form_id = general_tag[0].attrib.get("id", None)

        if form_id is None or form_id != idx:
            continue

        writer_id = int(general_tag[0].attrib.get("writerID", "0"))
        break

    return writer_id


def get_max_seq_len(strokes_path: Path) -> int:
    """
    This function calculates the maximum sequence length of stroke data in XML files located within the specified
    directory 'strokes_path' and its subdirectories. It iterates through the XML files, extracting stroke sequences
    and keeping track of the maximum length.

    Args:
        strokes_path (Path): The path to the directory containing XML files.

    Returns:
        int: The maximum sequence length of stroke data among all XML files.
    """

    max_length = 0
    xml_file = tuple(strokes_path.rglob("*.xml"))

    for path_xml_file in track(
        xml_file, description="Calculating max sequence length..."
    ):
        strokes = get_line_strokes(path_xml_file, 0)
        max_length = max(max_length, len(strokes))

    print(f"Max sequence length is {max_length}")
    return max_length


# def compute_mean(dataset: List[Dict[str, Any]]) -> Tensor:
#     strokes = []
#
#     for item in track(dataset, description="Computing mean..."):
#         strokes.extend(iter(item["strokes"][1:, :]))
#
#     strokes = torch.tensor(strokes, dtype=torch.float32)
#
#     mean = strokes.mean(axis=0)
#     mean[2] = 0.0
#
#     return mean
#
#
# def compute_std(dataset: List[Dict[str, Any]]) -> Tensor:
#     strokes = []
#
#     for item in track(dataset, description="Computing std..."):
#         strokes.extend(iter(item["strokes"][1:, :]))
#
#     strokes = torch.tensor(strokes, dtype=torch.float32)
#
#     std = strokes.std(axis=0)
#     std[2] = 1.0
#
#     return std


def load_dataset(
    path: Tuple[Path, Optional[Path]], max_files: int, *, latent: bool = False
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, int]]]:
    """
    This function loads a dataset from an H5 file and, optionally, a JSON file containing writer ID mappings.
    It retrieves and organizes data into a list of dictionaries and returns it. The 'latent' parameter controls
    the type of data loading (latent or non-latent).

    Args:
        path (Tuple[Path, Optional[Path]]): A tuple containing the path to the H5 file and, optionally,
                                            the path to a JSON file.
        max_files (int): The maximum number of files to load from the dataset.

    Returns:
        Tuple[List[Dict[str, Any], Optional[Dict[str, int]]]: A tuple containing a list of dictionaries
                                        representing the loaded data and, if latent, a dictionary mapping writer IDs.

    Example:
        ```
        h5_file_path = Path('path/to/dataset.h5')
        json_file_path = Path('path/to/writer_ids.json')
        max_files_to_load = 100
        latent_data = True
        dataset, writer_id_mapping = load_dataset((h5_file_path, json_file_path), max_files_to_load, latent=latent_data)
        ```
    """

    map_writer_id = None
    with open(path[1], mode="r") as fp:
        map_writer_id = json.load(fp)

    with h5py.File(path[0], mode="r") as f:
        dataset = []

        for group_name in track(
            f["dataset_group"], description="Loading dataset from H5 file..."
        ):
            group = f["dataset_group"][group_name]
            writer_id = group.attrs["writer"]

            if latent:
                data_dict = {
                    "writer": writer_id,
                    "image": torch.tensor(np.array(group["image"])),
                    "label": np.array(group["label"]),
                }

            else:
                data_dict = {
                    "writer": writer_id,
                    "file": group.attrs["file"],
                    "raw_text": group.attrs["raw_text"],
                    "strokes": np.array(group["strokes"]),
                    "text": np.array(group["text"]),
                    "one_hot": np.array(group["one_hot"]),
                    "image": ImageModule.fromarray(np.array(group["image"])),
                    "style": torch.tensor(np.array(group["style"])),
                }

            dataset.append(data_dict)

            if max_files and len(dataset) >= max_files:
                return dataset, map_writer_id

        return dataset, map_writer_id


def save_dataset(
    dataset: List[Dict[str, Any]],
    path: Tuple[Path, Optional[Path]],
    *,
    is_latent: bool = False,
    map_writer_ids: Dict[str, int],
) -> None:
    """
    This function saves a dataset to an H5 file and, optionally, a JSON file containing writer ID mappings.
    The 'is_latent' parameter controls the type of data being saved (latent or non-latent).

    Args:
        dataset (List[Dict[str, Any]]): A list of dictionaries representing the dataset to be saved.
        path (Tuple[Path, Optional[Path]]): A tuple containing the path to the H5 file and, optionally,
                                            the path to a JSON file.
        is_latent (bool, optional): If True, save latent data; otherwise, save non-latent data. Defaults to False.
        map_writer_ids (Optional[Dict[str, int]], optional): A dictionary mapping writer IDs to be saved in a JSON file.
                                                             Defaults to None.

    Returns:
        None

    Example:
        ```python
        dataset_to_save = [...]  # List of dictionaries representing the dataset
        h5_file_path = Path('path/to/dataset.h5')
        json_file_path = Path('path/to/writer_ids.json')
        is_data_latent = True
        writer_id_mapping = {...}  # Dictionary mapping writer IDs
        save_dataset(dataset_to_save, (h5_file_path, json_file_path), is_latent=is_data_latent, map_writer_ids=writer_id_mapping)
        ```
    """

    with open(path[1], mode="w") as fp:
        json.dump(map_writer_ids, fp, indent=4)

    with h5py.File(path[0], mode="w") as f:
        dataset_h5 = f.create_group("dataset_group")

        for i, data_dict in track(enumerate(dataset), description="Saving dataset..."):
            writer_id = data_dict["writer"]
            group = dataset_h5.create_group(f"dataset_{i}")
            group.attrs["writer"] = writer_id

            if is_latent:
                image_data = data_dict["image"].numpy()
                group.create_dataset("image", data=image_data)
                group.create_dataset("label", data=data_dict["label"])

            else:
                group.attrs["file"] = data_dict["file"]
                group.attrs["raw_text"] = data_dict["raw_text"]

                group.create_dataset("strokes", data=data_dict["strokes"])
                group.create_dataset("text", data=data_dict["text"])
                group.create_dataset("one_hot", data=data_dict["one_hot"])

                image_data = np.array(data_dict["image"])
                group.create_dataset("image", data=image_data)
                group.create_dataset("style", data=data_dict["style"].numpy())


def uniquify(path: Path) -> Path:
    """
    This function takes a 'path' and checks if a file with the same name exists. If it does, it appends a counter
    in parentheses to the file name to make it unique. It continues to increment the counter until a unique path is found.

    Args:
        path (Path): The original path to be made unique.

    Returns:
        Path: A unique path that does not clash with existing files.

    Example:
        ```
        original_path = Path('path/to/file.txt')
        unique_path = uniquify(original_path)
        ```
    """

    filename, extension = os.path.splitext(path)
    counter = 1

    while os.path.exists(path):
        path = f"{filename}({str(counter)}){extension}"
        counter += 1

    return Path(path)


class NormalizeInverse(Normalize):
    """
    This class represents the inverse of a normalization transform. It can be used to revert the
    normalization of an image tensor back to its original values by applying the inverse
    normalization.

    Args:
        mean (Sequence): The mean values used for the original normalization.
        std (Sequence): The standard deviation values used for the original normalization.

    Example:
        ```
        mean_values = [0.485, 0.456, 0.406]
        std_values = [0.229, 0.224, 0.225]
        normalize_transform = Normalize(mean=mean_values, std=std_values)
        inverse_transform = NormalizeInverse(mean=mean_values, std=std_values)

        # Normalize an image tensor
        normalized_image = normalize_transform(image_tensor)

        # Revert the normalization to get the original image tensor
        original_image = inverse_transform(normalized_image)
        ```
    """

    def __init__(self, mean: Sequence, std: Sequence) -> None:
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)

        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv

        super().__init__(mean_inv, std_inv, inplace=True)

    def __call__(self, tensor: Tensor) -> Tensor:
        return super().__call__(tensor.clone())
