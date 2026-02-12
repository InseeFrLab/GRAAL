import polars as pl
import s3fs


def sample_codes(fs: s3fs.S3FileSystem, population_path: str, code_column: str, n_codes: int):
    """
    Sample codes using Polars from S3.

    Args:
        fs: S3FileSystem configuré
        population_path: chemin S3 (avec ou sans s3://)
        code_column: nom de la colonne
        n_codes: nombre de codes à échantillonner
    """

    with fs.open(population_path, 'rb') as f:
        df = pl.read_parquet(f)

    sampled = df.select(code_column).sample(n=n_codes, with_replacement=True)

    return sampled[code_column].to_numpy()


def sample_codes_lazy(fs, population_path: str, code_column: str, n_codes: int):
    """
    Sample codes from a population with replacement.
    The population must be located in a parquet file and have a column for codes.
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
