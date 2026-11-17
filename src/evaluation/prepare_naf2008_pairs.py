"""Jeu d'évaluation NAF 2008 du MatchVerifier : code de production vs code humain.

Le test de la campagne d'annotation 2024 (`test_data_NAF2008.parquet`) porte le code
qu'un annotateur a retenu (`apet_manual`), mais pas celui que la chaîne de production
avait attribué. Celui-ci est dans les exports Label Studio de la même campagne
(`task.data.apet_finale`, un JSON par annotation), joints ici par `liasse_numero`.

On obtient l'exact analogue du protocole COICOP (cf. docs/eval/match_verifier_coicop.md) :
le code de production — `AUTO` (règles) ou `FASTTEXT` (modèle), cf. `mode_calcul_ape` —
joue le rôle de Sirus, l'annotateur celui de la vérité. Le parquet de sortie se donne
tel quel à src.evaluation.match_verifier_eval :

    NOMENCLATURE=naf2008 uv run -m src.evaluation.match_verifier_eval \
        --train-set data/eval/naf2008/pairs_test_NAF2008.parquet \
        --text-column libelle --code-column apet_finale \
        --expected-code-column apet_manual ...

Usage :
    uv run -m src.evaluation.prepare_naf2008_pairs \
        --output data/eval/naf2008/pairs_test_NAF2008.parquet
"""

import argparse
import json
import logging
from concurrent.futures import ThreadPoolExecutor

import polars as pl
from dotenv import load_dotenv

from src.evaluation.build_eval_set import load_dataframe
from src.utils.logging import configure_logging
from src.utils.storage import get_file_system

load_dotenv(override=True)
configure_logging()
logger = logging.getLogger(__name__)

CAMPAIGN = "projet-ape/label-studio/annotation-campaign-2024/NAF2008"
TEST_PATH = f"{CAMPAIGN}/preprocessed/test_data_NAF2008.parquet"
EXPORTS_PATH = f"{CAMPAIGN}/data-annotated"


def load_production_codes(exports_path: str) -> pl.DataFrame:
    """(liasse_numero, apet_finale) de chaque tâche annotée, terminée ou en cours."""
    fs = get_file_system()
    paths = [p for p in fs.find(exports_path) if not p.endswith(".keep")]
    logger.info(f"Reading {len(paths)} Label Studio exports under {exports_path}")

    def read(path: str) -> dict | None:
        try:
            data = json.loads(fs.cat(path))["task"]["data"]
        except (KeyError, TypeError, json.JSONDecodeError):
            return None
        return {"liasse_numero": data.get("liasse_numero"), "apet_finale": data.get("apet_finale")}

    with ThreadPoolExecutor(max_workers=32) as pool:
        rows = [r for r in pool.map(read, paths) if r and r["liasse_numero"]]
    codes = pl.DataFrame(rows).unique()
    conflicts = codes.group_by("liasse_numero").len().filter(pl.col("len") > 1)
    if len(conflicts):
        raise ValueError(f"{len(conflicts)} liasses ont plusieurs codes de production")
    return codes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--test-set", default=TEST_PATH)
    parser.add_argument("--exports", default=EXPORTS_PATH)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    test = load_dataframe(args.test_set)
    codes = load_production_codes(args.exports)
    pairs = (
        test.join(codes, on="liasse_numero", how="left")
        .rename({"text_description": "libelle"})
        .with_columns(pl.col("apet_manual").replace("", None))
    )
    missing = pairs.filter(pl.col("apet_finale").is_null())
    if len(missing):
        logger.warning(f"{len(missing)} lignes sans code de production, écartées")
    pairs = pairs.drop_nulls(["apet_finale", "apet_manual"])
    logger.info(
        f"{len(pairs)} lignes ; accord production / humain : "
        f"{(pairs['apet_finale'] == pairs['apet_manual']).mean():.1%}"
    )
    with get_file_system().open(args.output, "wb") if args.output.startswith(
        "projet-"
    ) else open(args.output, "wb") as f:
        pairs.write_parquet(f)
    logger.info(f"Écrit : {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
