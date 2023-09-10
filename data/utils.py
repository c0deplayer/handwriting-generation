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
from rich.progress import track
from scipy.signal import savgol_filter

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

    for stroke in stroke_set_tag.findall("Stroke"):
        strokes_cord.extend(
            [float(point.attrib["x"]), -float(point.attrib["y"]), 0.0]
            for point in stroke.findall("Point")
        )
        strokes_cord[-1][2] = 1.0

    strokes = np.array(strokes_cord, dtype=np.float32)
    strokes = __denoise_strokes(strokes)
    # strokes[:, 2] = np.roll(strokes[:, 2], 1)
    strokes[:, :2] /= np.std(strokes[:, :2])

    return __pad_strokes(strokes, max_length)


def __denoise_strokes(strokes: np.array) -> np.array:
    """
    _summary_

    Parameters
    ----------
    strokes : np.array
        _description_

    Returns
    -------
    np.array
        _description_
    """

    stroke_segments = np.split(strokes, np.where(strokes[:, 2] == 1)[0] + 1, axis=0)
    n_strokes = []

    for strokes in stroke_segments:
        if len(strokes) != 0:
            x_smooth = savgol_filter(
                strokes[:, 0], window_length=7, polyorder=3, mode="nearest"
            )
            y_smooth = savgol_filter(
                strokes[:, 1], window_length=7, polyorder=3, mode="nearest"
            )
            smoothed_stroke = np.column_stack((x_smooth, y_smooth, strokes[:, 2]))
            n_strokes.append(smoothed_stroke)

    strokes = np.vstack(n_strokes)
    diffs = np.concatenate(
        [strokes[1:, :2] - strokes[:-1, :2], strokes[1:, 2:3]], axis=1
    )

    return np.vstack((np.array([[0.0, 0.0, 0.0]]), diffs))


def get_image(
    path: Path, width: int, height: int, *, latent: bool = False
) -> Union[Tuple[Image, Image], Tuple[None, None]]:
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

    Returns
    -------
    Union[Tuple[Image, Image], Tuple[None, None]]
        _description_

    Raises
    ------
    FileNotFoundError
        _description_
    """

    try:
        img = ImageModule.open(path)
    except PIL.UnidentifiedImageError:
        return None, None
    except FileNotFoundError as e:
        raise f"The image was not found: {path}" from e

    if not latent:
        if img.mode != "L":
            img = img.convert("L")

        # noinspection PyTypeChecker
        img = __remove_whitespaces(np.array(img).astype("uint8"), threshold=127)
        img = ImageModule.fromarray(img, mode="L")

        w, h = img.size
        img = img.resize(size=(w * height // h, height), resample=ImageModule.BILINEAR)

    else:
        if img.mode != "RGB":
            img = img.convert("RGB")

        w, h = img.size

        img = img.resize(size=(w * height // h, height), resample=ImageModule.LANCZOS)
        w, h = img.size

        if w > width:
            img = img.resize(size=(width, height), resample=ImageModule.LANCZOS)

    return __pad_image(img, width, height)


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

    if text_len <= max_len:
        padded_text = np.full((max_len - text_len,), pad_value)
        # noinspection PyTypeChecker
        text = np.concatenate((text, padded_text))

    labels = torch.as_tensor(text)
    one_hot = F.one_hot(labels, num_classes=tokenizer.get_vocab_size())

    return one_hot.numpy(), text


def __remove_whitespaces(img: np.array, *, threshold: int) -> np.array:
    """
    _summary_

    Parameters
    ----------
    img : np.array
        _description_
    threshold : int
        _description_

    Returns
    -------
    np.array
        _description_
    """

    row_mins = np.amin(img, axis=1)
    col_mins = np.amin(img, axis=0)

    rows = np.where(row_mins < threshold)[0]
    cols = np.where(col_mins < threshold)[0]

    return img[rows[0] : rows[-1], cols[0] : cols[-1]]


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


def __pad_image(img: Image, width: int, height: int) -> Tuple[Image, Image]:
    """
    _summary_

    Parameters
    ----------
    img : Image
        _description_
    width : int
        _description_
    height : int
        _description_

    Returns
    -------
    Tuple[Image, Image]
        _description_
    """

    if img.size[0] > width:
        return img, img
    else:
        return ImageOps.pad(image=img, size=(width, height), color="white"), img


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


def load_dataset_from_h5(
    path: Path, max_files: int, *, latent: bool = False
) -> Tuple[List[Dict[str, Any]], Union[None, Dict[str, int]]]:
    """
    _summary_

    Parameters
    ----------
    path : Path
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

    with h5py.File(path, mode="r") as f:
        dataset = []
        map_writer_id = {}

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
                if writer_id not in map_writer_id.keys():
                    map_writer_id[writer_id] = group.attrs["writer_map"]

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


def save_dataset_to_h5(
    dataset: List[Dict[str, Any]],
    path: Path,
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
    path : Path
        _description_
    latent : bool, optional
        _description_, by default False
    map_writer_ids : Dict[str, int], optional
        _description_, by default None
    """

    with h5py.File(path, mode="w") as f:
        dataset_h5 = f.create_group("dataset_group")

        for i, data_dict in track(
            enumerate(dataset), description="Saving dataset for later usage..."
        ):
            writer_id = data_dict["writer"]
            group = dataset_h5.create_group(f"dataset_{i}")
            group.attrs["writer"] = writer_id

            if latent:
                group.attrs["writer_map"] = map_writer_ids[writer_id]
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
