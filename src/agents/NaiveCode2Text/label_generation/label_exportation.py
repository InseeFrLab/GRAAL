import logging
import re

import pandas as pd
import s3fs

logger = logging.getLogger(__name__)


def create_file_name(
        output_path: str,
        output_format: str,
        temperature: float,
        language: str,
        exhaustive_sampling: bool = False,
        use_fewshot: bool = False,
        n_fewshot: int = 0,
        model_name: str = None,
        model: str = None,
        ) -> str:
    """
    Create the full output_path with an automatic name representing the config inputs.

    Args:
        output_path (str): the folder to upload the file in
        output_format (str): the format (.txt or .parquet)
        temperature: float,
        language: str,
        exhaustive_sampling: bool = False,
        use_fewshot: bool = False,
        n_fewshot: int = 0,
        model_name: str = None,
        model: str = None,

    Returns:
        bool: True if the file has been correctly saved.
    """

    temp_string = f"_temp{temperature}".replace(".", "")

    if model_name is None and model_name:
        model_string = "_" + model
        if model_name is None:
            model_string = ""
    else:
        model_string = "_" + model_name

    model_string = re.sub(r"[^a-zA-Z0-9_-]", "-", model_string)

    if use_fewshot:
        if n_fewshot > 0:
            fewshot_string = f"_fewshot{n_fewshot}"
        else:
            fewshot_string = "_fewshot"
    else:
        fewshot_string = ""

    if exhaustive_sampling:
        exhaust_string = "_exhaustive"
    else:
        exhaust_string = ""

    file_name = "generation" + model_string + temp_string + fewshot_string + exhaust_string

    assert output_format in [".txt", ".parquet"], "Output format should be either .txt or .parquet"

    file_name += output_format

    final_path = output_path + file_name + output_format

    return final_path


def export_to_txt(
        codes: list,
        names: list,
        labels: list,
        file_path: str,
        generation_time: float
        ) -> bool:
    """
    Save results to file .txt
    The number of codes used for generation should not exceed 100.
    Else, upload in .parquet format

    Args:
        codes (list): List of codes generated.
        names (list): List of names for the codes generated
        labels (list of lists): List of labels for each code
        file_path (str): The .txt file path.
        generation_time (float): The time it took to generate.

    Returns:
        bool: True if the file has been correctly saved.
    """
    if len(labels) == 0 or len(labels) > 100:
        return False

    nb_labels = len(labels) * len(labels[0])

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(f"{nb_labels} wordings have been generated in {generation_time:.2f} sec.\n\n")
        f.write("=" * 36 + "\n")

        for code, name, generated_labels in zip(codes, names, labels):
            f.write(f"Code: {code}\n")
            f.write(f"Name: {name}\n")
            f.write("Result:\n")

            for j, label in enumerate(generated_labels):
                f.write(f"{j}. {label}\n")

            f.write("\n" + "=" * 36 + "\n")

    return True


def export_to_parquet(
        codes: list,
        names: list,
        labels: list,
        file_path: str,
        fs: s3fs.S3FileSystem
        ) -> bool:
    """
    Save results to file .txt
    The number of codes used for generation should not exceed 100.
    Else, upload in .parquet format

    Args:
        codes (list): List of codes used for the generation.
        names (list): List of names for the codes used for the generation.
        labels (list of lists): List of generated labels for each code.
        file_path (str): The .parquet file path.
        fs (float): The filesystem for exportation.

    Returns:
        bool: True if the file has been correctly saved.
    """

    # Basic validation
    assert (len(codes) == len(names) == len(labels)), \
        "Codes, names, and labels must have the same length."

    assert len(codes) != 0, "Empty input: nothing to export."

    # Flatten structure
    rows = []

    for code, name, generated_labels in zip(codes, names, labels):

        assert isinstance(generated_labels, list), "Labels must be stored in a list."

        for label in generated_labels:
            rows.append({
                "code": code,
                "name": name,
                "label": label
            })

    # Save to parquet
    df = pd.DataFrame(rows)

    if fs is None:
        try:
            existing_df = pd.read_parquet(file_path)
            df = pd.concat([existing_df, df], ignore_index=True)
        except FileNotFoundError:
            logger.info(f"No file found at location{file_path}, creating...")
    else:
        try:
            with fs.open(file_path, 'rb') as f:
                existing_df = pd.read_parquet(f)
            df = pd.concat([existing_df, df], ignore_index=True)
        except FileNotFoundError:
            logger.info(f"No file found at location{file_path}, creating...")

    df.to_parquet(
        file_path,
        engine="pyarrow",
        index=False,
        filesystem=fs
    )

    return True
