import polars as pl
import s3fs
import numpy as np


def sample_codes_lazy(
        fs: s3fs.S3FileSystem,
        population_path: str,
        code_column: str,
        n_codes: int
        ) -> np.ndarray:
    """
    Sample codes with replacement using lazyframes from Polars.

    Args:
        fs (S3FileSystem): The filesystem for importation.
        population_path (str): The path of the parquet file of the population.
        code_column (str): The name of the column for codes.
        n_codes (int): The number of codes to sample.

    Returns:
        numpy.ndarray: An array of n_codes codes sampled with replacement.
    """

    with fs.open(population_path, 'rb') as f:
        lf = (
            pl.scan_parquet(f)
            .with_row_index("row_id")
        )

        total_rows = lf.select(pl.len()).collect().item()

        random_ids = (
            pl.Series("row_id", range(total_rows))
            .sample(n=n_codes, with_replacement=True)
            .to_frame()
            .lazy()
        )

        sampled = lf.join(random_ids, on="row_id", how="inner")

    df = sampled.collect()

    return df[code_column].to_numpy()
