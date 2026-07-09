"""Audit de la qualité des labels du jeu d'entraînement via un LLM vérificateur.

Tire un échantillon aléatoire du jeu d'entraînement et demande au MatchVerifier
s'il pense que le code associé à chaque libellé est correct. Ne compare pas à
une prédiction de modèle : c'est un contrôle qualité du label de référence
lui-même (cf. src.evaluation.run_eval pour l'évaluation d'un classifieur).

Nécessite à l'exécution : la base Neo4j et l'API LLM configurées dans
l'environnement (mêmes prérequis que src.main).

Usage :
    uv run -m src.evaluation.verify_train_labels \
        --train-set projet-ape/data/08112022_27102024/naf2025/split/df_train.parquet \
        --n-samples 200 \
        --output data/eval/train_verification
"""

import argparse
import asyncio
import json
import logging
import os

import polars as pl

from src.agents.closers.match_verifier import MatchVerificationInput, MatchVerifier
from src.config import neo4j_config
from src.evaluation.build_eval_set import load_dataframe
from src.neo4j_graph.graph import Graph
from src.utils.logging import configure_logging

configure_logging()
logger = logging.getLogger(__name__)

TRAIN_SET_PATH = "projet-ape/data/08112022_27102024/naf2025/split/df_train.parquet"


async def verify_rows(
    verifier: MatchVerifier, rows: list[dict], text_column: str, code_column: str
) -> list[dict]:
    """Verify each row's existing (text, code) label with MatchVerifier.

    No `proposed_explanation`/`proposed_confidence` is passed, so the verifier
    treats each code as a raw ground-truth label rather than a model's guess
    (cf. the "no explanation provided" prompt branch in MatchVerifier).
    """
    results = []
    for i, row in enumerate(rows, 1):
        text = row[text_column]
        code = row[code_column]
        logger.info(f"{i}/{len(rows)}: {text!r} -> {code}")

        verification = await verifier(MatchVerificationInput(activity=text, code=str(code)))
        results.append(
            {
                text_column: text,
                code_column: code,
                "llm_is_match": verification.is_match,
                "llm_confidence": verification.confidence,
                "llm_explanation": verification.explanation,
            }
        )
    return results


async def run(args) -> int:
    df = load_dataframe(args.train_set)
    sample = df.sample(n=min(args.n_samples, len(df)), seed=args.seed)
    logger.info(f"Verifying {len(sample)} train labels with MatchVerifier")

    verifier = MatchVerifier(Graph(neo4j_config))
    results = await verify_rows(verifier, sample.to_dicts(), args.text_column, args.code_column)

    details = pl.DataFrame(results)
    n = len(details)
    n_flagged = int((~details["llm_is_match"]).sum()) if n else 0
    summary = {
        "n": n,
        "n_flagged": n_flagged,
        "agreement_rate": (1 - n_flagged / n) if n else None,
        "mean_llm_confidence": float(details["llm_confidence"].mean()) if n else None,
    }

    os.makedirs(args.output, exist_ok=True)
    details_path = os.path.join(args.output, "train_label_verification.parquet")
    details.write_parquet(details_path)
    summary_path = os.path.join(args.output, "train_label_verification_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))

    flagged = details.filter(~pl.col("llm_is_match"))
    if len(flagged):
        print("\nFlagged rows (LLM thinks the label may be wrong):")
        for r in flagged.to_dicts():
            print(f"  {r[args.text_column]!r} -> {r[args.code_column]}: {r['llm_explanation']}")

    logger.info(f"Details written to {details_path}, summary to {summary_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify a sample of train-set labels with an LLM")
    parser.add_argument(
        "--train-set", default=TRAIN_SET_PATH, help="Train parquet (local path or S3 key)"
    )
    parser.add_argument("--text-column", default="libelle", help="Text column (default: libelle)")
    parser.add_argument(
        "--code-column", default="nace2025", help="Label column (default: nace2025)"
    )
    parser.add_argument(
        "--n-samples", type=int, default=200, help="Number of rows to sample (default: 200)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed (default: 42)")
    parser.add_argument("--output", default="data/eval/train_verification", help="Output directory")
    args = parser.parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
