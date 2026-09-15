"""Ajoute à la volée de nouveaux exemples labellisés à un jeu d'évaluation existant.

Les jeux d'évaluation (`eval_set.parquet`, `eval_set_sample200.parquet`, ...) ne
portent pas d'embeddings précalculés (les classifieurs embedent à l'inférence),
donc ajouter une ligne ne demande qu'un append au parquet — pas de
réentraînement ni de recalcul de features.

Depuis build_eval_set.py::stratified_sample (cf. son docstring), un jeu
d'évaluation fraîchement construit porte aussi `eval_stratum` (préfixe de code
normalisé, --stratum-depth) et `ipw_weight` (poids inverse-probabilité), utilisés
par run_eval.py pour l'exactitude pondérée et les intervalles de confiance par
bootstrap stratifié. Une ligne ajoutée à la main n'a pas été tirée d'une strate,
donc elle reçoit `ipw_weight=1.0` (elle ne compte que pour elle-même, aucune
correction à appliquer) et un `eval_stratum` dérivé de son code de la même façon
que build_eval_set.py (même profondeur par défaut), pour tomber dans la même
strate que les lignes tirées autour d'elle. `add_rows` ne garde que les colonnes
réellement présentes dans le fichier cible, donc ceci reste compatible avec les
jeux d'évaluation plus anciens qui n'ont que `sample_weight` (sans
`eval_stratum`/`ipw_weight`), comme l'actuel data/eval/eval_set.parquet.

Usage (un seul exemple) :
    uv run -m src.evaluation.add_eval_examples \\
        --eval-set data/eval/eval_set_sample200.parquet \\
        --libelle "Vente de fleurs en ligne" --code 4776Z

Usage (plusieurs exemples, depuis un CSV/JSON avec au moins les colonnes
`libelle` et `apet2025`) :
    uv run -m src.evaluation.add_eval_examples \\
        --eval-set data/eval/eval_set_sample200.parquet \\
        --input new_examples.csv

Le chemin --eval-set peut être local ou S3 (s3://...), lu et réécrit via
src.utils.storage comme le reste du pipeline d'évaluation.
"""

import argparse
import logging

import polars as pl

from src.evaluation.metrics import normalize_code
from src.evaluation.row_id import row_id_for
from src.utils.logging import configure_logging
from src.utils.storage import open_path

configure_logging()
logger = logging.getLogger(__name__)

SCHEMA_COLUMNS = ["apet2025", "libelle", "CJ", "NAT", "TYP", "SRF", "CRT", "sample_weight"]
DEFAULTS = {"CJ": None, "NAT": None, "TYP": None, "SRF": None, "CRT": None, "sample_weight": 1.0}
# Metadata build_eval_set.py has added since 4c0b118 (cf. module docstring); computed for
# every new row regardless of what `existing` has, since add_rows() below keeps only
# whichever of these columns are actually present in the target file.
DEFAULT_STRATUM_DEPTH = 5
DEFAULT_IPW_WEIGHT = 1.0


def load_eval_set(path: str) -> pl.DataFrame:
    with open_path(path, "rb") as f:
        return pl.read_parquet(f)


def eval_stratum_for(code: str | None, stratum_depth: int) -> str | None:
    norm = normalize_code(code)
    return norm[:stratum_depth] if norm else None


def load_new_rows(input_path: str, stratum_depth: int) -> pl.DataFrame:
    with open_path(input_path, "rb") as f:
        if input_path.endswith(".json"):
            df = pl.read_json(f)
        elif input_path.endswith(".parquet"):
            df = pl.read_parquet(f)
        else:
            df = pl.read_csv(f)

    missing_required = {"libelle", "apet2025"} - set(df.columns)
    if missing_required:
        raise ValueError(f"Input file is missing required column(s): {missing_required}")

    for col, default in DEFAULTS.items():
        if col not in df.columns:
            df = df.with_columns(pl.lit(default).alias(col))

    if "ipw_weight" not in df.columns:
        df = df.with_columns(pl.lit(DEFAULT_IPW_WEIGHT).alias("ipw_weight"))
    if "eval_stratum" not in df.columns:
        df = df.with_columns(
            pl.col("apet2025")
            .map_elements(lambda c: eval_stratum_for(c, stratum_depth), return_dtype=pl.Utf8)
            .alias("eval_stratum")
        )

    return df.select(SCHEMA_COLUMNS + ["eval_stratum", "ipw_weight"])


def build_single_row(
    libelle: str,
    code: str,
    cj: str | None,
    nat: str | None,
    typ: str | None,
    srf: float | None,
    crt: str | None,
    sample_weight: float,
    stratum_depth: int,
) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "apet2025": [code],
            "libelle": [libelle],
            "CJ": [cj],
            "NAT": [nat],
            "TYP": [typ],
            "SRF": [srf],
            "CRT": [crt],
            "sample_weight": [sample_weight],
            "eval_stratum": [eval_stratum_for(code, stratum_depth)],
            "ipw_weight": [DEFAULT_IPW_WEIGHT],
        },
        schema={
            "apet2025": pl.Utf8,
            "libelle": pl.Utf8,
            "CJ": pl.Utf8,
            "NAT": pl.Utf8,
            "TYP": pl.Utf8,
            "SRF": pl.Float64,
            "CRT": pl.Utf8,
            "sample_weight": pl.Float64,
            "eval_stratum": pl.Utf8,
            "ipw_weight": pl.Float64,
        },
    )


def add_rows(existing: pl.DataFrame, new_rows: pl.DataFrame) -> pl.DataFrame:
    """Concatène en écartant les doublons (même `row_id_for(libelle, code normalisé)`)."""
    new_rows = new_rows.with_columns(
        pl.col("apet2025").map_elements(normalize_code, return_dtype=pl.Utf8).alias("apet2025")
    )
    dropped = new_rows.filter(pl.col("apet2025").is_null())
    for row in dropped.iter_rows(named=True):
        logger.warning(f"Skipping row with empty/invalid code: {row['libelle']!r}")
    new_rows = new_rows.filter(pl.col("apet2025").is_not_null())

    existing_ids = {
        row_id_for(row["libelle"], row["apet2025"]) for row in existing.iter_rows(named=True)
    }
    new_ids = new_rows.with_columns(
        pl.struct(["libelle", "apet2025"])
        .map_elements(lambda s: row_id_for(s["libelle"], s["apet2025"]), return_dtype=pl.Utf8)
        .alias("_row_id")
    )
    duplicates = new_ids.filter(pl.col("_row_id").is_in(existing_ids))
    for row in duplicates.iter_rows(named=True):
        logger.warning(
            f"Skipping duplicate already in eval set: {row['libelle']!r} ({row['apet2025']})"
        )

    to_add = new_ids.filter(~pl.col("_row_id").is_in(existing_ids)).drop("_row_id")
    if to_add.is_empty():
        logger.info("No new rows to add (all duplicates or invalid).")
        return existing

    logger.info(
        f"Adding {len(to_add)} new row(s) to eval set ({len(existing)} -> "
        f"{len(existing) + len(to_add)})."
    )
    return pl.concat([existing, to_add.select(existing.columns)], how="vertical")


def main() -> int:
    parser = argparse.ArgumentParser(description="Add new labeled examples to an eval-set parquet")
    parser.add_argument(
        "--eval-set", required=True, help="Eval-set parquet to update (local path or S3 key)"
    )

    single = parser.add_argument_group("single example")
    single.add_argument("--libelle", help="Business activity description")
    single.add_argument("--code", help="Ground-truth NAF code (apet2025)")
    single.add_argument("--cj", default=None, help="Legal category (CJ), if known")
    single.add_argument("--nat", default=None, help="Nature (NAT), if known")
    single.add_argument("--typ", default=None, help="Type (TYP), if known")
    single.add_argument("--srf", type=float, default=None, help="Surface (SRF), if known")
    single.add_argument("--crt", default=None, help="CRT flag, if known")
    single.add_argument(
        "--sample-weight", type=float, default=1.0, help="sample_weight (default: 1.0)"
    )

    parser.add_argument(
        "--input", help="CSV/JSON/parquet file with new rows (columns: libelle, apet2025, ...)"
    )
    parser.add_argument(
        "--stratum-depth",
        type=int,
        default=DEFAULT_STRATUM_DEPTH,
        help="Code prefix length for eval_stratum, matching build_eval_set.py (default: 5)",
    )
    args = parser.parse_args()

    if bool(args.input) == bool(args.libelle or args.code):
        parser.error("Pass either --input, or --libelle and --code, but not both")
    if args.libelle and not args.code:
        parser.error("--code is required alongside --libelle")

    existing = load_eval_set(args.eval_set)

    if args.input:
        new_rows = load_new_rows(args.input, args.stratum_depth)
    else:
        new_rows = build_single_row(
            args.libelle,
            args.code,
            args.cj,
            args.nat,
            args.typ,
            args.srf,
            args.crt,
            args.sample_weight,
            args.stratum_depth,
        )

    updated = add_rows(existing, new_rows)
    with open_path(args.eval_set, "wb") as f:
        updated.write_parquet(f)
    logger.info(f"Eval set written to {args.eval_set}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
