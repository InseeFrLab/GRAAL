"""Audit de la qualité des labels du jeu d'entraînement via un LLM vérificateur.

Tire un échantillon aléatoire du jeu d'entraînement et demande au MatchVerifier
s'il pense que le code associé à chaque libellé est correct. Ne compare pas à
une prédiction de modèle : c'est un contrôle qualité du label de référence
lui-même (cf. src.evaluation.run_eval pour l'évaluation d'un classifieur).

Pour chaque libellé où le vérificateur signale un désaccord (llm_is_match =
False), demande dans la foulée au SummaryAgenticClassifier (cf.
src.agents.Text2Code.classifiers.summary_classifier) sa propre proposition de
code — un second avis indépendant, calculé une bonne fois pour toutes ici et
stocké dans les colonnes summary_code/summary_confidence/summary_explanation du
parquet produit, plutôt que recalculé à la demande par un outil de revue. Coûte
un appel agentique de plus, mais seulement sur les libellés effectivement en
désaccord.

Chaque appel (MatchVerifier et SummaryAgenticClassifier) est retenté une fois en
cas d'échec (timeout LLM notamment) avant d'abandonner : une ligne où
MatchVerifier échoue deux fois est passée (log + exclue du résultat, plutôt que
de faire planter tout le run) ; une ligne où seul SummaryAgenticClassifier échoue
garde son verdict mais les colonnes summary_* restent vides. Chaque ligne traitée
est aussi journalisée au fil de l'eau dans <output>/train_label_verification.checkpoint.jsonl,
pour ne perdre que le travail non flush si le run est interrompu.

Nécessite à l'exécution : la base Neo4j et l'API LLM configurées dans
l'environnement (mêmes prérequis que src.main), ainsi que le résumé NACE
(uv run -m src.neo4j_graph.build_nace_summary) pour la proposition du
SummaryAgenticClassifier — sans lui, l'audit continue mais les colonnes
summary_* restent vides (passer --summary-path '' pour l'ignorer explicitement).

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
from src.agents.Text2Code.classifiers.summary_classifier import (
    DEFAULT_SUMMARY_PATH,
    SummaryAgenticClassifier,
)
from src.config import neo4j_config
from src.evaluation.build_eval_set import load_dataframe
from src.neo4j_graph.graph import Graph
from src.utils import storage
from src.utils.logging import configure_logging
from src.utils.retry import call_with_retries

configure_logging()
logger = logging.getLogger(__name__)

TRAIN_SET_PATH = "projet-ape/data/08112022_27102024/naf2025/split/df_train.parquet"

# SummaryAgenticClassifier is a multi-turn agentic loop (several sequential LLM calls
# via tool calling): a single timed-out turn kills the whole call, not just that one
# HTTP request, and the OpenAI client's own retry (base_agent.py) only covers a single
# request. Worth one extra full attempt before giving up on a libellé.
_SUMMARY_CLASSIFIER_MAX_ATTEMPTS = 2
# MatchVerifier is a single-shot call, so far less exposed to the multi-turn failure
# mode above, but a transient timeout can still hit it — worth the same one retry
# rather than losing the row entirely.
_VERIFIER_MAX_ATTEMPTS = 2


async def verify_rows(
    verifier: MatchVerifier,
    rows: list[dict],
    text_column: str,
    code_column: str,
    summary_classifier: SummaryAgenticClassifier | None = None,
    checkpoint_path: str | None = None,
) -> list[dict]:
    """Verify each row's existing (text, code) label with MatchVerifier.

    No `proposed_explanation`/`proposed_confidence` is passed, so the verifier
    treats each code as a raw ground-truth label rather than a model's guess
    (cf. the "no explanation provided" prompt branch in MatchVerifier). Retries once
    on failure (cf. _VERIFIER_MAX_ATTEMPTS); a row where both attempts fail is
    skipped entirely (logged, not included in the returned results) rather than
    crashing the whole batch.

    When `summary_classifier` is given, also asks it for a second-opinion code on
    every row flagged as a mismatch (is_match=False), and stores the result in the
    summary_code/summary_confidence/summary_explanation columns. Also retries once
    on failure (cf. _SUMMARY_CLASSIFIER_MAX_ATTEMPTS); left null otherwise, including
    when both attempts fail.

    When `checkpoint_path` is given, each row's entry is appended there as soon as
    it's computed (flushed immediately, one JSON line), so a crash partway through
    (e.g. an unhandled error reaching the LLM endpoint) loses only what's unflushed,
    not the whole run — same rationale as run_eval.py's checkpointing.
    """
    checkpoint = None
    if checkpoint_path is not None:
        storage.makedirs(os.path.dirname(checkpoint_path) or ".")
        checkpoint = storage.open_path(checkpoint_path, "w", encoding="utf-8")

    try:
        results = []
        for i, row in enumerate(rows, 1):
            text = row[text_column]
            code = row[code_column]
            logger.info(f"{i}/{len(rows)}: {text!r} -> {code}")

            verification = await call_with_retries(
                lambda: verifier(MatchVerificationInput(activity=text, code=str(code))),
                _VERIFIER_MAX_ATTEMPTS,
                f"MatchVerifier for {text!r}",
            )
            if verification is None:
                logger.warning(f"Skipping {text!r}: MatchVerifier never succeeded")
                continue

            entry = {
                text_column: text,
                code_column: code,
                "llm_is_match": verification.is_match,
                "llm_confidence": verification.confidence,
                "llm_explanation": verification.explanation,
                "summary_code": None,
                "summary_confidence": None,
                "summary_explanation": None,
            }

            if not verification.is_match and summary_classifier is not None:
                proposal = await call_with_retries(
                    lambda: summary_classifier(text),
                    _SUMMARY_CLASSIFIER_MAX_ATTEMPTS,
                    f"SummaryAgenticClassifier for {text!r}",
                )
                if proposal is not None:
                    entry["summary_code"] = proposal.code
                    entry["summary_confidence"] = proposal.proposed_confidence
                    entry["summary_explanation"] = proposal.proposed_explanation

            results.append(entry)
            if checkpoint is not None:
                checkpoint.write(json.dumps(entry, ensure_ascii=False) + "\n")
                checkpoint.flush()
        return results
    finally:
        if checkpoint is not None:
            checkpoint.close()


async def run(args) -> int:
    df = load_dataframe(args.train_set)
    sample = df.sample(n=min(args.n_samples, len(df)), seed=args.seed)
    logger.info(f"Verifying {len(sample)} train labels with MatchVerifier")

    graph = Graph(neo4j_config)
    verifier = MatchVerifier(graph)

    summary_classifier = None
    if args.summary_path:
        try:
            summary_classifier = SummaryAgenticClassifier(graph, summary_path=args.summary_path)
        except Exception:
            logger.exception(
                "Could not build SummaryAgenticClassifier, mismatch rows won't get a "
                "summary_code/summary_confidence/summary_explanation proposal"
            )

    checkpoint_path = os.path.join(args.output, "train_label_verification.checkpoint.jsonl")
    results = await verify_rows(
        verifier,
        sample.to_dicts(),
        args.text_column,
        args.code_column,
        summary_classifier=summary_classifier,
        checkpoint_path=checkpoint_path,
    )

    details = pl.DataFrame(results)
    n = len(details)
    n_skipped = len(sample) - n
    n_flagged = int((~details["llm_is_match"]).sum()) if n else 0
    summary = {
        "n": n,
        "n_skipped": n_skipped,
        "n_flagged": n_flagged,
        "agreement_rate": (1 - n_flagged / n) if n else None,
        "mean_llm_confidence": float(details["llm_confidence"].mean()) if n else None,
    }

    storage.makedirs(args.output)
    details_path = os.path.join(args.output, "train_label_verification.parquet")
    with storage.open_path(details_path, "wb") as f:
        details.write_parquet(f)
    summary_path = os.path.join(args.output, "train_label_verification_summary.json")
    with storage.open_path(summary_path, "w", encoding="utf-8") as f:
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
    parser.add_argument(
        "--summary-path",
        default=DEFAULT_SUMMARY_PATH,
        help="NACE summary file for the SummaryAgenticClassifier second opinion computed on "
        "flagged (llm_is_match=False) rows (cf. src.neo4j_graph.build_nace_summary); pass "
        "'' to skip it",
    )
    args = parser.parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
