import json
import os
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Union, Dict, List, Tuple, Any

import PIL
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image as ImageModule, ImageOps
from PIL.Image import Image
from numpy.linalg import norm
from rich.progress import track

from .tokenizer import Tokenizer


def get_transcription(path: Path) -> Dict[str, str]:
    """
    _summary_

    Parameters
    ----------
    path : Path
        _description_

    Returns
    -------
    Dict[str, str]
        _description_
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


def get_line_strokes(path: Path, max_length: int) -> np.array:
    """
    _summary_

    Parameters
    ----------
    path : Path
        _description_
    max_length : int
        _description_

    Returns
    -------
    np.array
        _description_

    Raises
    ------
    ET.ParseError
        _description_
    ValueError
        _description_
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
    strokes[:, :2] /= np.std(strokes[:, :2])

    for _ in range(3):
        strokes = __combine_strokes(strokes, int(len(strokes) * 0.15))

    return __pad_strokes(strokes, max_length)


def __combine_strokes(strokes: np.ndarray, n: int) -> np.ndarray:
    """
    _summary_
    Parameters
    ----------
    strokes : np.ndarray
        _description_
    n : int
        _description_
    Returns
    -------
    np.ndarray
        _description_
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
    _summary_

    Parameters
    ----------
    path : Path
        _description_
    width : int
        _description_
    height : int
        _description_
    latent : bool, optional
        _description_, by default False
    centering : Tuple[float, float], optional
        _description_, by default (0.0, 0.0)

    Returns
    -------
    Union[Image, None]
        _description_

    Raises
    ------
    FileNotFoundError
        _description_
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

    w, h = img.size

    img = img.resize(size=(w * height // h, height), resample=ImageModule.LANCZOS)
    w, h = img.size

    if w > width:
        img = img.resize(size=(width, height), resample=ImageModule.LANCZOS)

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
    _summary_

    Parameters
    ----------
    text : str
        _description_
    tokenizer : Tokenizer
        _description_
    max_len : int
        _description_
    pad_value : int, optional
        _description_, by default 0

    Returns
    -------
    Tuple[np.array, np.array]
        _description_
    """

    text = tokenizer.encode(text)

    text_len = len(text)
    max_len += 2

    if text_len < max_len:
        padded_text = np.full((max_len - text_len,), pad_value)
        # noinspection PyTypeChecker
        text = np.concatenate((text, padded_text))

    labels = torch.as_tensor(text)
    one_hot = F.one_hot(labels, num_classes=tokenizer.get_vocab_size())

    return one_hot.numpy(), text


def __pad_strokes(
    strokes: np.array, max_length: int, *, fill_value: float = 0
) -> Union[np.array, None]:
    """
    _summary_

    Parameters
    ----------
    strokes : np.array
        _description_
    max_length : int
        _description_
    fill_value : float, optional
        _description_, by default 0

    Returns
    -------
    np.array | None
        _description_
    """

    if max_length:
        stroke_len = len(strokes)

        if stroke_len > max_length or np.amax(np.abs(strokes)) > 15:
            return None

        padded_strokes = np.full((max_length, 3), fill_value, dtype=np.float32)
        padded_strokes[:, 2] = 1.0
        padded_strokes[:stroke_len, :] = strokes[:, :]
    else:
        padded_strokes = strokes

    return padded_strokes


def get_max_seq_len(strokes_path: Path) -> int:
    """
    _summary_

    Parameters
    ----------
    strokes_path : Path
        _description_

    Returns
    -------
    int
        _description_
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


def load_dataset(
    path: Tuple[Path, Union[Path, None]], max_files: int, *, latent: bool = False
) -> Tuple[List[Dict[str, Any]], Union[None, Dict[str, int]]]:
    """
    _summary_

    Parameters
    ----------
    path : Tuple[Path, Union[Path, None]]
        _description_
    max_files : int
        _description_
    latent : bool, optional
        _description_, by default False

    Returns
    -------
    Tuple[List[Dict[str, Any]], Union[None, Dict[str, int]]]
        _description_
    """
    if latent:
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
    path: Tuple[Path, Union[Path, None]],
    *,
    latent: bool = False,
    map_writer_ids: Dict[str, int] = None,
) -> None:
    """
    _summary_

    Parameters
    ----------
    dataset : List[Dict[str, Any]]
        _description_
    path : Tuple[Path, Union[Path, None]]
        _description_
    latent : bool, optional
        _description_, by default False
    map_writer_ids : Dict[str, int], optional
        _description_, by default None
    """
    if map_writer_ids is not None:
        with open(path[1], mode="w") as fp:
            json.dump(map_writer_ids, fp, indent=4)

    with h5py.File(path[0], mode="w") as f:
        dataset_h5 = f.create_group("dataset_group")

        for i, data_dict in track(enumerate(dataset), description="Saving dataset..."):
            writer_id = data_dict["writer"]
            group = dataset_h5.create_group(f"dataset_{i}")
            group.attrs["writer"] = writer_id

            if latent:
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


def uniquify(path: Union[Path, str]) -> str:
    filename, extension = os.path.splitext(path)
    counter = 1

    while os.path.exists(path):
        path = f"{filename}({str(counter)}){extension}"
        counter += 1

    return path
