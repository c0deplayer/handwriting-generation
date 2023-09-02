import warnings
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Union, Dict, List

import PIL
import numpy as np
import torch
from PIL import Image as image, ImageOps
from PIL.Image import Image
from numpy.linalg import norm
from rich.progress import track
from torch import Tensor


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


def get_line_stroke(path: Path, *, diffusion: bool) -> np.ndarray:
    """
    _summary_

    Parameters
    ----------
    path : Path
        _description_
    diffusion : bool
        _description_,

    Returns
    -------
    np.ndarray
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
            x, y = float(point.attrib["x"]), float(point.attrib["y"])

            if prev_cord is not None:
                dx, dy = x - prev_cord[0], -y - prev_cord[1]
                strokes_cord.append([dx, dy, 0.0])

            prev_cord = [x, -y]

        if strokes_cord:
            strokes_cord[-1][-1] = 1.0
        else:
            strokes_cord.append([prev_cord[0], prev_cord[1], 1.0])

    strokes = np.array(strokes_cord)
    strokes[:, 2] = np.roll(strokes[:, 2], 1)

    # TODO: Test if combine_strokes will make RNN better
    # ? Apply std based on on set of strokes or whole dataset ?
    # if diffusion:
    #     strokes[:, :2] /= np.std(strokes[:, :2], axis=0)
    #
    #     for _ in range(3):
    #         strokes = __combine_strokes(strokes, int(len(strokes) * 0.2))

    return strokes


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

    if len(strokes) % 2 != 0:
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

    return strokes


def get_image(
    path: Path, *, width: int, height: int, latent: bool = False
) -> Union[Image, None]:
    """
    _summary_

    Parameters
    ----------
    path : Path
        _description_
    width : int
        _description_
    height : iny
        _description_
    latent : bool
        _description_

    Returns
    -------
    Image | None
        _description_

    Raises
    ------
    FileNotFoundError
        _description_
    """

    if not path.is_file():
        raise FileNotFoundError(f"The image was not found: {path}")
    try:
        img = image.open(path)
    except PIL.UnidentifiedImageError:
        return None

    if not latent:
        if img.mode != "L":
            img = img.convert("L")

        # noinspection PyTypeChecker
        img = __remove_whitespaces(np.array(img).astype("uint8"), threshold=127)
        img = image.fromarray(img, mode="L")

        w, h = img.size
        img = img.resize(size=(w * height // h, height), resample=image.BILINEAR)

    else:
        if img.mode != "RGB":
            img = img.convert("RGB")

        w, h = img.size

        img = img.resize(size=(w * height // h, height), resample=image.LANCZOS)
        w, h = img.size

        if w > width:
            img = img.resize(size=(width, height), resample=image.LANCZOS)

    return img


def __remove_whitespaces(img: np.ndarray, *, threshold: int) -> np.ndarray:
    """
    _summary_

    Parameters
    ----------
    img : np.ndarray
        _description_
    threshold : int
        _description_

    Returns
    -------
    np.ndarray
        _description_
    """

    row_mins = np.amin(img, axis=1)
    col_mins = np.amin(img, axis=0)

    rows = np.where(row_mins < threshold)[0]
    cols = np.where(col_mins < threshold)[0]

    return img[rows[0] : rows[-1], cols[0] : cols[-1]]


def pad_stroke(
    stroke: np.ndarray, *, max_length: int, fill_value: float = 0
) -> Union[np.ndarray, None]:
    """
    _summary_

    Parameters
    ----------
    stroke : np.ndarray
        _description_
    max_length : int
        _description_
    fill_value : float, optional
        _description_, by default 0

    Returns
    -------
    np.ndarray | None
        _description_
    """

    stroke_len = len(stroke)
    stroke_tmp = stroke[:, :2] / np.std(stroke[:, :2], axis=0)

    # ? Hu ?
    if stroke_len > max_length or np.amax(np.abs(stroke_tmp)) > 15:
        return None

    padded_strokes = np.full((max_length, 3), fill_value, dtype=np.float32)
    padded_strokes[:, 2] = 1
    padded_strokes[:stroke_len, :2] = stroke[:, :2]
    padded_strokes[:stroke_len, 2] = stroke[:, 2]

    return padded_strokes


def fill_text(
    text: List[int], *, max_len: int, pad_value: int = 0
) -> Union[np.ndarray, None]:
    """
    _summary_

    Parameters
    ----------
    text : List[int]
        _description_
    max_len : int
        _description_
    pad_value : int, optional
        _description_, by default 0

    Returns
    -------
    np.ndarray
        _description_
    """

    text_len = len(text)
    if text_len > max_len:
        return None

    padded_text = np.full((max_len - text_len,), pad_value)
    # noinspection PyTypeChecker
    text = np.concatenate((text, padded_text))

    return text


def pad_image(img: Image, *, width: int, height: int) -> Image:
    """
    _summary_

    Parameters
    ----------
    img : np.ndarray
        _description_
    width : int
        _description_
    height : int
        _description_

    Returns
    -------
    np.ndarray
        _description_
    """
    if img.size[0] > width:
        return img
    else:
        return ImageOps.pad(image=img, size=(width, height), color="white")


def compute_mean(dataset: List[Dict[str, Any]]) -> Tensor:
    """
    _summary_

    Parameters
    ----------
    dataset : List[Dict[str, Any]]
        _description_

    Returns
    -------
    Tensor
        _description_
    """

    strokes = []

    for item in track(dataset, description="Computing mean..."):
        strokes.extend(iter(item["strokes"]))

    strokes = torch.tensor(strokes, dtype=torch.float32)

    mean = strokes.mean(axis=0)
    mean[2] = 0.0

    return mean


def compute_std(dataset: List[Dict[str, Any]]) -> Tensor:
    """
    _summary_

    Parameters
    ----------
    dataset : List[Dict[str, Any]]
        _description_

    Returns
    -------
    Tensor
        _description_
    """

    strokes = []

    for item in track(dataset, description="Computing std..."):
        strokes.extend(iter(item["strokes"]))

    strokes = torch.tensor(strokes, dtype=torch.float32)

    std = strokes.std(axis=0)
    std[2] = 1.0

    return std


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
        strokes = get_line_stroke(path_xml_file, diffusion=True)
        max_length = max(max_length, len(strokes))

    print(f"Max sequence length is {max_length}")
    return max_length
