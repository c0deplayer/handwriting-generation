import sys

sys.path.append(".")

import os
import string
from pathlib import Path
from xml.etree import ElementTree as ET

from rich.progress import track

from configs.config import ConfigDiffusion
from utils.utils import load_config


def read_lines(file_path: Path, skip_lines: int = 0) -> list[str]:
    """Reads lines from a file, optionally skipping the first few lines.

    Args:
        file_path (Path): The path to the file.
        skip_lines (int): The number of lines to skip at the beginning.

    Returns:
        list[str]: A list of stripped lines from the file.
    """
    with file_path.open(mode="r") as f:
        lines = f.readlines()[skip_lines:]

    return [line.strip() for line in lines]


def write_data_to_file(
    file_path: Path, data: list[str], description: str
) -> None:
    """Writes a list of strings to a file, with a progress description.

    Args:
        file_path (Path): The path to the output file.
        data (list[str]): The data to write to the file.
        description (str): The description for the progress tracker.
    """
    with file_path.open(mode="w") as output:
        for row in track(data, description=description):
            output.write(f"{row}\n")


def create_form_dict(form_lines: list[str]) -> dict[str, str]:
    """Creates a dictionary mapping form IDs to writer IDs.

    Args:
        form_lines (list[str]): The lines from the forms file.

    Returns:
        dict[str, str]: A dictionary mapping form IDs to writer IDs.
    """
    return {line.split(" ")[0]: line.split(" ")[1] for line in form_lines}


def filter_data_iam_dataset(filter_punctuation: bool = True) -> None:
    """Filters the IAM dataset based on punctuation and splits the data into train, validation, and test sets.

    Args:
        filter_punctuation (bool): Whether to filter out words with punctuation. Defaults to True.

    Raises:
        IndexError: If a writer ID is not found in the form dictionary.
    """
    words_path = data_path_tmp / "ascii/words.txt"
    forms_path = data_path_tmp / "ascii/forms.txt"
    data_split_paths = [
        data_path_tmp / f"raw_data_split/{set_name}.txt"
        for set_name in [
            "trainset",
            "validationset1",
            "validationset2",
            "testset",
        ]
    ]

    train_files, val_files, test_files = [], [], []
    split_txt_lines = [read_lines(path) for path in data_split_paths]

    word_lines = read_lines(words_path, skip_lines=23)
    form_lines = read_lines(forms_path, skip_lines=16)

    form_dict = create_form_dict(form_lines)

    for line in track(word_lines, description="Filtering IAMDB..."):
        parts = line.split()
        idx_full, word = parts[0], parts[-1]
        idx_parts = idx_full.split("-")
        idx_two = f"{idx_parts[0]}-{idx_parts[1]}"

        if filter_punctuation and any(p in word for p in string.punctuation):
            continue

        writer_id = form_dict.get(idx_two)
        if writer_id is None:
            raise IndexError("Writer_id is empty!")

        split_index = next(
            (
                i
                for i, split in enumerate(split_txt_lines)
                if idx_full[:-3] in split
            ),
            None,
        )
        match split_index:
            case 0 | 1:
                train_files.append(f"{writer_id},{idx_full} {word}")
            case 2:
                val_files.append(f"{writer_id},{idx_full} {word}")
            case 3:
                test_files.append(f"{writer_id},{idx_full} {word}")

    file_data_mapping = {
        "iam_tr_va1.filter": (
            train_files,
            "Saving train set for IAM Dataset...",
        ),
        "iam_va2.filter": (
            val_files,
            "Saving validation set for IAM Dataset...",
        ),
        "iam_test.filter": (test_files, "Saving test set for IAM Dataset..."),
    }

    for file_name, (data, description) in file_data_mapping.items():
        write_data_to_file(data_path_tmp / file_name, data, description)


def filter_data_iam_on_dataset(
    config_diff: ConfigDiffusion, *, force: bool = False
) -> None:
    """Filters the IAMonDB dataset and saves the filtered data to a file.

    Args:
        force (bool): Whether to force re-filtering even if the output file already exists. Defaults to False.

    Raises:
        ET.ParseError: If there is an error parsing an XML file.
    """
    if not force and os.path.isfile(txt_file_iam_on_db):
        print(f"TXT file already exists: {txt_file_iam_on_db}")
        return

    ascii_files = []
    xml_files = {}
    xml_path = data_path_diff / "original-xml"
    ascii_path = data_path_diff / "ascii"

    for file in track(
        xml_path.rglob("*.xml"),
        description="Collecting information about IAMonDB...",
    ):
        filename = file.stem
        dir_name = str(file.parent).split("/")[-1]

        if filename in config_diff.blacklist:
            continue

        xml_files.setdefault(dir_name, [])
        xml_files[dir_name].append(filename)

    for file in track(
        ascii_path.rglob("*.txt"), description="Filtering IAMonDB..."
    ):
        filename = file.stem
        dir_name = str(file.parent).split("/")[-1]
        xml_list = xml_files.get(dir_name)

        if filename in config_diff.blacklist or xml_list is None:
            continue

        for xml_file in xml_list:
            path = xml_path / f"{dir_name[:3]}/{dir_name}/{xml_file}.xml"

            try:
                root = ET.parse(path).getroot()
            except (ET.ParseError, FileNotFoundError) as e:
                raise ET.ParseError(
                    f"Failed to parse file {path}\n{str(e)}"
                ) from e

            general_tag = root.find("General")
            writer_id = (
                int(general_tag[0].attrib.get("writerID", "0"))
                if general_tag is not None
                else 0
            )

            if (filename[-1] == xml_file[-1]) or (
                filename[-1].isdigit() and xml_file[-1] == "s"
            ):
                ascii_files.append(f"{writer_id},{filename}")

    write_data_to_file(
        txt_file_iam_on_db, ascii_files, description="Saving IAMonDB..."
    )


if __name__ == "__main__":
    config_diff = load_config("./configs/Diffusion/base_gpu.yaml", "Diffusion")

    data_path_tmp = Path("./raw_data/IAMDB/").resolve()
    data_path_diff = Path(config_diff.data_path).resolve()

    txt_file_iam_on_db = data_path_diff / "dataset.txt"

    filter_data_iam_dataset(filter_punctuation=True)
    filter_data_iam_on_dataset(config_diff, force=True)
