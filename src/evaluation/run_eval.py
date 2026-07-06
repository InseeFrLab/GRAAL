"""Exécution d'une campagne d'évaluation d'un classifieur sur un jeu d'évaluation.

Charge un jeu d'évaluation (cf. build_eval_set.py), classe chaque libellé avec
la méthode choisie, calcule les métriques (exactitude feuille et par niveau,
taux d'échec) et écrit les prédictions détaillées + le rapport de métriques.

Nécessite à l'exécution : la base Neo4j et l'API LLM configurées dans
l'environnement (mêmes prérequis que src.main).

Usage :
    uv run -m src.evaluation.run_eval \
        --eval-set data/eval/eval_set.parquet \
        --method navigator \
        --output-dir data/eval/results
"""

import argparse
import asyncio
import json
import logging
import os

import polars as pl

from src.evaluation.metrics import evaluate
from src.main import classify_agentic_rag, classify_navigator
from src.utils.logging import configure_logging

configure_logging()
logger = logging.getLogger(__name__)

METHODS = {
    "navigator": classify_navigator,
    "agentic-rag": classify_agentic_rag,
}


async def run(args) -> int:
    df = pl.read_parquet(args.eval_set)
    labels = df[args.text_column].to_list()
    y_true = df[args.code_column].to_list()
    logger.info(f"Evaluating method '{args.method}' on {len(labels)} labels")

    method = METHODS[args.method]
    predictions = await method(labels)

    # Les classifieurs renvoient des MatchVerificationInput ; un échec (pas de
    # code final atteint) est représenté par une prédiction sans code.
    y_pred = [getattr(pred, "code", None) for pred in predictions]
    explanations = [getattr(pred, "proposed_explanation", None) for pred in predictions]
    confidences = [getattr(pred, "proposed_confidence", None) for pred in predictions]

    report = evaluate(y_true, y_pred)
    logger.info(f"Metrics: {report}")

    os.makedirs(args.output_dir, exist_ok=True)

    details = df.with_columns(
        pl.Series("prediction", y_pred),
        pl.Series("explanation", explanations),
        pl.Series("confidence", confidences),
    )
    details_path = os.path.join(args.output_dir, f"predictions_{args.method}.parquet")
    details.write_parquet(details_path)

    report_path = os.path.join(args.output_dir, f"metrics_{args.method}.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(json.dumps(report, indent=2, ensure_ascii=False))
    logger.info(f"Predictions written to {details_path}, metrics to {report_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run an evaluation campaign")
    parser.add_argument("--eval-set", required=True, help="Evaluation parquet (cf. build_eval_set)")
    parser.add_argument("--method", choices=sorted(METHODS), default="navigator")
    parser.add_argument("--text-column", default="libelle", help="Text column (default: libelle)")
    parser.add_argument("--code-column", default="nace2025", help="Label column (default: nace2025)")
    parser.add_argument("--output-dir", default="data/eval/results", help="Output directory")
    args = parser.parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
