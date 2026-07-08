"""Construction d'un jeu d'évaluation stratifié à partir d'un parquet labellisé.

Stratification par préfixe de code (2 caractères = division NAF par défaut)
pour couvrir les classes rares comme les sections à fort volume, avec un
tirage reproductible (seed fixée).

Usage :
    uv run -m src.evaluation.build_eval_set \
        --input projet-ape/data/08112022_27102024/naf2025/split/df_test.parquet \
        --output data/eval/eval_set.parquet \
        --n-per-stratum 10

Le chemin d'entrée est lu en local s'il existe, sinon sur S3 (Datalab/Onyxia,
identifiants AWS_* dans l'environnement).
"""

import argparse
import logging
import os

import polars as pl

from src.evaluation.config import PATH_EVAL_INPUT, PATH_EVAL_OUTPUT
from src.evaluation.metrics import normalize_code
from src.utils.logging import configure_logging

configure_logging()
logger = logging.getLogger(__name__)


def load_dataframe(path: str) -> pl.DataFrame:
    """Charge un parquet depuis le disque local ou S3 (Datalab)."""
    if os.path.exists(path):
        logger.info(f"Loading local parquet: {path}")
        return pl.read_parquet(path)

    import s3fs

    logger.info(f"Loading parquet from S3: {path}")
    fs = s3fs.S3FileSystem(
        client_kwargs={"endpoint_url": os.environ["AWS_ENDPOINT_URL"]},
        key=os.environ["AWS_ACCESS_KEY_ID"],
        secret=os.environ["AWS_SECRET_ACCESS_KEY"],
        token=os.environ["AWS_SESSION_TOKEN"],
    )
    with fs.open(path, "rb") as f:
        return pl.read_parquet(f)


def stratified_sample(
    df: pl.DataFrame,
    code_column: str,
    n_per_stratum: int,
    stratum_depth: int = 2,
    seed: int = 42,
) -> pl.DataFrame:
    """Tire au plus `n_per_stratum` lignes par strate de code.

    Args:
        df: Données labellisées.
        code_column: Colonne contenant le code de nomenclature.
        n_per_stratum: Nombre maximal de lignes par strate (les strates plus
            petites sont prises en entier).
        stratum_depth: Longueur du préfixe de code normalisé définissant la
            strate (2 = division NAF).
        seed: Graine du tirage, pour un jeu reproductible.

    Le sur-échantillonnage des strates rares casse la fréquence réelle des
    codes : une exactitude moyennée sans pondération sur le jeu obtenu répond
    à « quelle est la performance moyenne par code », pas « quelle serait la
    performance sur le trafic réel ». `ipw_weight` (population de la strate /
    lignes effectivement tirées) permet de reconstruire cette seconde lecture
    a posteriori (cf. `metrics.evaluate(..., weights=...)`). `eval_stratum`
    est conservée pour que le rééchantillonnage stratifié (bootstrap) n'ait
    pas à redériver les strates depuis le code, avec un risque de dérive si
    `stratum_depth` change.
    """
    df = (
        df.with_columns(
            pl.col(code_column)
            .map_elements(normalize_code, return_dtype=pl.Utf8)
            .alias("_norm_code")
        )
        .drop_nulls("_norm_code")
        .with_columns(pl.col("_norm_code").str.slice(0, stratum_depth).alias("_stratum"))
    )

    n_strata = df["_stratum"].n_unique()
    logger.info(f"{n_strata} strata found at depth {stratum_depth}")

    df = df.with_columns(pl.len().over("_stratum").alias("_population_count"))
    sampled = df.filter(pl.int_range(pl.len()).shuffle(seed=seed).over("_stratum") < n_per_stratum)
    sampled = sampled.with_columns(
        pl.len().over("_stratum").alias("_sampled_count"),
    ).with_columns(
        (pl.col("_population_count") / pl.col("_sampled_count")).alias("ipw_weight"),
    )

    logger.info(f"Sampled {len(sampled)} rows out of {len(df)}")
    return sampled.rename({"_stratum": "eval_stratum"}).drop(
        ["_norm_code", "_population_count", "_sampled_count"]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a stratified evaluation set")
    parser.add_argument(
        "--input", default=PATH_EVAL_INPUT, help="Input parquet (local path or S3 key)"
    )
    parser.add_argument("--output", default=PATH_EVAL_OUTPUT, help="Output parquet (local path)")
    parser.add_argument(
        "--code-column", default="apet2025", help="Label column (default: apet2025)"
    )
    parser.add_argument(
        "--n-per-stratum", type=int, default=10, help="Max rows per stratum (default: 10)"
    )
    parser.add_argument(
        "--stratum-depth",
        type=int,
        default=5,
        help="Code prefix length defining a stratum (default: 2 = NAF division)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed (default: 42)")
    args = parser.parse_args()

    df = load_dataframe(args.input)
    sampled = stratified_sample(
        df,
        code_column=args.code_column,
        n_per_stratum=args.n_per_stratum,
        stratum_depth=args.stratum_depth,
        seed=args.seed,
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    sampled.write_parquet(args.output)
    logger.info(f"Evaluation set written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
