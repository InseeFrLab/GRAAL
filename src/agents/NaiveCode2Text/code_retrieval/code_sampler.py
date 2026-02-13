import polars as pl
import s3fs
import numpy as np


def sample_codes(
        fs: s3fs.S3FileSystem,
        population_path: str,
        code_column: str,
        n_codes: str
        ) -> np.ndarray:
    """
    Sample codes with replacement using dataframes from Polars.

    Args:
        fs: S3FileSystem configuré
    """

    with fs.open(population_path, 'rb') as f:
        df = pl.read_parquet(f)

    sampled = df.select(code_column).sample(n=n_codes, with_replacement=True)

    return sampled[code_column].to_numpy()


def sample_codes_lazy(
        fs: s3fs.S3FileSystem,
        population_path: str,
        code_column: str,
        n_codes: str
        ) -> np.ndarray:
    """
    Sample codes with replacement using lazyframes from Polars.
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
