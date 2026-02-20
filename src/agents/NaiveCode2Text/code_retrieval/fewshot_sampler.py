import polars as pl
import s3fs
import numpy as np


def sample_fewshot_lazy(
        fs: s3fs.S3FileSystem,
        population_path: str,
        code_column: str,
        code: str,
        label_column: str,
        n_fewshot: int
        ) -> np.ndarray:
    """
    Sample examples of labels of a certain code using lazyframes from Polars.

    Args:
        fs (S3FileSystem): The filesystem for importation.
        population_path (str): The path of the parquet file of the population.
        code_column (str): The name of the column for codes.
        code (str): The code to sample examples from.
        label_column (str): The name of the column for labels.
        n_fewshot (int): The number of examples to sample.

    Returns:
        numpy.ndarray: An array of n_fewshot labels of code sampled without replacement.
    """

    with fs.open(population_path, "rb") as f:
        lf = (
            pl.scan_parquet(f)
            .filter(pl.col(code_column) == code)
            .select(label_column)
            .with_row_index("row_id")
        )

        n_available = lf.select(pl.len()).collect().item()

        n_fewshot = min(n_available, n_fewshot)

        random_ids = (
            pl.Series("row_id", range(n_available))
            .sample(n=n_fewshot, with_replacement=False)
            .to_frame()
            .lazy()
        )

        sampled = lf.join(random_ids, on="row_id", how="inner")

    df = sampled.collect()

    return df[label_column].to_numpy()
