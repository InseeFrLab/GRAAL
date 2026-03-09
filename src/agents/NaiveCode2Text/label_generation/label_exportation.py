import logging

import pandas as pd
import s3fs

logger = logging.getLogger(__name__)


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
    if not (len(codes) == len(names) == len(labels)):
        raise "Codes, names, and labels must have the same length."
        return False

    if len(codes) == 0:
        raise "Empty input: nothing to export."
        return False

    # Flatten structure
    rows = []

    for code, name, generated_labels in zip(codes, names, labels):

        if not isinstance(generated_labels, list):
            raise "Labels must be stored in a list."
            return False

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
