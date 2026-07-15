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

from src.agents.closers.match_verifier import MatchVerificationInput
from src.evaluation.bootstrap import bootstrap_ci
from src.evaluation.config import PATH_EVAL_OUTPUT
from src.evaluation.metrics import accuracy_at_depth, evaluate
from src.main import (
    classify_agentic_rag,
    classify_navigator,
    classify_summary,
    classify_supervised,
)
from src.utils import storage
from src.utils.logging import configure_logging

configure_logging()
logger = logging.getLogger(__name__)

METHODS = {
    "navigator": classify_navigator,
    "agentic-rag": classify_agentic_rag,
    "summary": classify_summary,
    # Reference baseline: production model, for the Navigator/Agentic-RAG/supervised
    # comparison called for in the cadrage's note de conception (§3.3-B).
    "supervised": classify_supervised,
}


async def run(args) -> int:
    with storage.open_path(args.eval_set, "rb") as f:
        df = pl.read_parquet(f)
    df = df.sample(fraction=args.fraction)
    labels = df[args.text_column].to_list()
    y_true = df[args.code_column].to_list()
    logger.info(f"Evaluating method '{args.method}' on {len(labels)} labels")

    if "ipw_weight" in df.columns:
        weights = df["ipw_weight"].to_list()
    else:
        logger.warning(
            "No 'ipw_weight' column in eval set: weighted accuracy skipped "
            "(eval set predates IPW weighting, rebuild it with build_eval_set.py)."
        )
        weights = None

    storage.makedirs(args.output_dir)
    method = METHODS[args.method]

    # Predictions are checkpointed to disk as they're produced (one JSONL line per
    # label, flushed immediately) instead of only written after the whole batch
    # succeeds: classify_navigator/classify_agentic_rag/classify_supervised already
    # process labels one at a time internally, so calling them one label at a time
    # here is equivalent to the previous bulk call, not just a wrapper around it — and
    # it means a crash partway through (e.g. an unhandled error reaching the LLM
    # endpoint) loses only what's unflushed, not every prediction gathered so far.
    # Note: when --output-dir is an s3:// path, this durability guarantee is weaker —
    # s3fs buffers writes and only uploads on close, so a crash still loses everything
    # written since the checkpoint file was opened.
    checkpoint_path = os.path.join(args.output_dir, f"predictions_{args.method}.checkpoint.jsonl")
    predictions = []
    with storage.open_path(checkpoint_path, "w", encoding="utf-8") as ckpt:
        for label in labels:
            try:
                pred = await method(label)
            except Exception as e:
                # Not every classifier has BaseClassifier's own try/except+fallback
                # (e.g. SummaryAgenticClassifier is a single free-running Runner.run
                # with no Python-owned guard rail — cf. its docstring), so an error
                # reaching the LLM endpoint (e.g. openai.APITimeoutError) can still
                # propagate here. Record it as a failed prediction instead of losing
                # the rest of the batch: code="" reads as a missing prediction in
                # evaluate() (cf. normalize_code), the same convention classifiers'
                # own fallbacks use for "no final code reached".
                logger.exception(f"Classification failed for {label!r}, recording as a failure")
                pred = MatchVerificationInput(
                    activity=label, code="", proposed_explanation=str(e), proposed_confidence=0.0
                )
            predictions.append(pred)
            ckpt.write(pred.model_dump_json() + "\n")
            ckpt.flush()

    # Les classifieurs renvoient des MatchVerificationInput ; un échec (pas de
    # code final atteint) est représenté par une prédiction sans code. Une
    # confiance nulle avec un code renseigné signale un repli de finalisation
    # (`_fallback_output`), pas un échec au sens strict : cf. low_confidence_rate.
    y_pred = [getattr(pred, "code", None) for pred in predictions]
    explanations = [getattr(pred, "proposed_explanation", None) for pred in predictions]
    confidences = [getattr(pred, "proposed_confidence", None) for pred in predictions]

    report = evaluate(y_true, y_pred, weights=weights, confidences=confidences)

    if args.bootstrap > 0:
        if "eval_stratum" in df.columns:
            strata = df["eval_stratum"].to_list()
        else:
            logger.warning(
                "No 'eval_stratum' column in eval set: bootstrap falls back to a single "
                "stratum (unstratified), understating the effect of the sampling design."
            )
            strata = [0] * len(y_true)

        def accuracy_metric(t, p, w):
            return accuracy_at_depth(t, p, weights=w)

        _, ci = bootstrap_ci(
            y_true, y_pred, strata, accuracy_metric, n_resamples=args.bootstrap, seed=args.seed
        )
        report["leaf_accuracy_ci"] = list(ci)

        if weights is not None:
            _, weighted_ci = bootstrap_ci(
                y_true,
                y_pred,
                strata,
                accuracy_metric,
                n_resamples=args.bootstrap,
                seed=args.seed,
                weights=weights,
            )
            report["leaf_accuracy_weighted_ci"] = list(weighted_ci)

    logger.info(f"Metrics: {report}")

    details = df.with_columns(
        pl.Series("prediction", y_pred),
        pl.Series("explanation", explanations),
        pl.Series("confidence", confidences),
    )
    details_path = os.path.join(args.output_dir, f"predictions_{args.method}.parquet")
    with storage.open_path(details_path, "wb") as f:
        details.write_parquet(f)

    report_path = os.path.join(args.output_dir, f"metrics_{args.method}.json")
    with storage.open_path(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    storage.remove(checkpoint_path)

    print(json.dumps(report, indent=2, ensure_ascii=False))
    logger.info(f"Predictions written to {details_path}, metrics to {report_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run an evaluation campaign")
    parser.add_argument(
        "--eval-set", default=PATH_EVAL_OUTPUT, help="Evaluation parquet (cf. build_eval_set)"
    )
    parser.add_argument(
        "--fraction", default="1.0", help="Dataset's fraction to evaluate (default: 1.0)"
    )
    parser.add_argument("--method", choices=sorted(METHODS), default="navigator")
    parser.add_argument("--text-column", default="libelle", help="Text column (default: libelle)")
    parser.add_argument(
        "--code-column", default="apet2025", help="Label column (default: apet2025)"
    )
    parser.add_argument("--output-dir", default="data/eval/results", help="Output directory")
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=0,
        help="Number of bootstrap resamples for accuracy CIs (0 = off, default)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Bootstrap seed (default: 42)")
    args = parser.parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
