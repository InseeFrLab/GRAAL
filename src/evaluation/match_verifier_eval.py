"""Audit de la qualité des labels du jeu d'entraînement via le MatchVerifier.

Tire un échantillon aléatoire du jeu d'entraînement et demande au MatchVerifier
s'il pense que le code associé à chaque libellé est correct. Ne compare pas à
une prédiction de modèle : c'est un contrôle qualité du label de référence
lui-même (cf. src.evaluation.run_eval pour l'évaluation d'un classifieur), et
surtout la matière première de l'évaluation du MatchVerifier lui-même — le
parquet produit ici est relu par src.evaluation.apps.match_verifier_eval_app,
où des annotateurs humains jugent chaque verdict.

Le parquet de sortie tient en six colonnes, volontairement : libelle,
current_code, match_verifier_verdict, match_verifier_explanation,
match_verifier_confidence, match_verifier_duration_s (le temps d'inférence de
l'appel, cf. verify_rows). C'est ce dont l'app de revue a besoin, rien de plus
(les versions précédentes de ce script demandaient en plus au
SummaryAgenticClassifier un second avis sur les lignes en désaccord ; ce second
avis relève de l'évaluation des classifieurs — cf.
src.evaluation.evaluate_eval_set_multi_method — pas de celle du vérificateur).

Chaque appel est retenté une fois en cas d'échec (timeout LLM notamment) avant
d'abandonner : une ligne où MatchVerifier échoue deux fois est passée (log +
exclue du résultat, plutôt que de faire planter tout le run). Les lignes tournent
concurremment, `--concurrency` bornant le nombre d'appels LLM en vol ; chaque
ligne terminée est journalisée au fil de l'eau dans
<output>/match_verifier_eval.checkpoint.jsonl, pour ne perdre que le travail non
flush si le run est interrompu.

Nécessite à l'exécution : la base Neo4j et l'API LLM configurées dans
l'environnement (mêmes prérequis que src.main).

Usage :
    uv run -m src.evaluation.match_verifier_eval \
        --train-set projet-ape/data/08112022_27102024/naf2025/split/df_train.parquet \
        --n-samples 500 \
        --output s3://projet-ape/graal/data/eval/match_verifier_eval
"""

import argparse
import asyncio
import json
import logging
import os
import time
from datetime import datetime

import polars as pl
from langfuse import propagate_attributes

from src.agents.closers.match_verifier import MatchVerificationInput, MatchVerifier
from src.config import neo4j_config
from src.evaluation.build_eval_set import load_dataframe
from src.evaluation.row_id import row_id_for
from src.neo4j_graph.graph import Graph
from src.utils import storage
from src.utils.logging import configure_logging
from src.utils.retry import call_with_retries

configure_logging()
logger = logging.getLogger(__name__)

TRAIN_SET_PATH = "projet-ape/data/08112022_27102024/naf2025/split/df_train.parquet"
DEFAULT_OUTPUT = "s3://projet-ape/graal/data/eval/match_verifier_eval"

# Output column names, fixed: match_verifier_eval_app reads exactly these.
TEXT_COLUMN_OUT = "libelle"
CODE_COLUMN_OUT = "current_code"
VERDICT_COLUMN = "match_verifier_verdict"
EXPLANATION_COLUMN = "match_verifier_explanation"
CONFIDENCE_COLUMN = "match_verifier_confidence"
DURATION_COLUMN = "match_verifier_duration_s"

# MatchVerifier is a single-shot call, so far less exposed to the multi-turn failure
# mode that hits the agentic classifiers, but a transient timeout can still hit it —
# worth one retry rather than losing the row entirely.
_VERIFIER_MAX_ATTEMPTS = 2


async def verify_rows(
    verifier: MatchVerifier,
    rows: list[dict],
    text_column: str,
    code_column: str,
    concurrency: int = 5,
    checkpoint_path: str | None = None,
    session_id: str | None = None,
) -> list[dict]:
    """Verify each row's existing (text, code) label with MatchVerifier.

    No `proposed_explanation`/`proposed_confidence` is passed, so the verifier
    treats each code as a raw ground-truth label rather than a model's guess
    (cf. the "no explanation provided" prompt branch in MatchVerifier). Retries once
    on failure (cf. _VERIFIER_MAX_ATTEMPTS); a row where both attempts fail is
    skipped entirely (logged, not included in the returned results) rather than
    crashing the whole batch.

    Rows run concurrently, with at most `concurrency` verifier calls in flight at
    once — MatchVerifier holds no per-call mutable state and its one shared Graph is
    a Neo4j driver, safe for concurrent use across sessions (same rationale as
    evaluate_eval_set_multi_method.py).

    When `checkpoint_path` is given, each row's entry is appended there as soon as
    it's computed (flushed immediately, one JSON line), so a crash partway through
    (e.g. an unhandled error reaching the LLM endpoint) loses only what's unflushed,
    not the whole run — same rationale as run_eval.py's checkpointing. Note: on an
    s3:// path that guarantee is void, s3fs only uploading on close (hence
    --checkpoint-dir).

    When `session_id` is given, each call's Langfuse trace is grouped under it and
    tagged with the row_id it belongs to — the same hash the review app uses, so a
    trace, a parquet row and a human judgment for one activity all correlate by that
    single value (same wrapper rationale as run_eval.py).
    """
    checkpoint = None
    if checkpoint_path is not None:
        storage.makedirs(os.path.dirname(checkpoint_path) or ".")
        checkpoint = storage.open_path(checkpoint_path, "w", encoding="utf-8")

    semaphore = asyncio.Semaphore(concurrency)
    total = len(rows)
    results: list[dict | None] = [None] * total

    async def handle(i: int, row: dict) -> None:
        text = row[text_column]
        code = row[code_column]

        async def call():
            with propagate_attributes(
                session_id=session_id,
                metadata={"row_id": row_id_for(text, str(code))},
                tags=["eval", "match_verifier_eval"],
            ):
                return await verifier(MatchVerificationInput(activity=text, code=str(code)))

        async with semaphore:
            # Timed inside the semaphore, so the column measures the call itself and not
            # how long the row queued behind --concurrency others. It covers everything
            # `verifier(...)` does — the Neo4j notice lookup in build_prompt, then the
            # LLM turns — and, on a retried row, the failed attempt too: the cost of
            # getting this verdict, which is what a per-verdict latency figure is for
            # (`attempt_count` is what tells retried rows apart).
            start = time.perf_counter()
            verification = await call_with_retries(
                call, _VERIFIER_MAX_ATTEMPTS, f"MatchVerifier {i + 1}/{total} for {text!r}"
            )
            duration = time.perf_counter() - start
        if verification is None:
            logger.warning(f"Skipping {text!r}: MatchVerifier never succeeded")
            return

        entry = {
            TEXT_COLUMN_OUT: text,
            CODE_COLUMN_OUT: str(code),
            VERDICT_COLUMN: verification.is_match,
            EXPLANATION_COLUMN: verification.explanation,
            CONFIDENCE_COLUMN: verification.confidence,
            DURATION_COLUMN: duration,
        }
        results[i] = entry
        logger.info(
            f"{i + 1}/{total} ({duration:5.1f}s): {text!r} -> {code} : "
            f"{'match' if verification.is_match else 'no match'}"
        )
        if checkpoint is not None:
            # write()+flush() never straddles an `await`, so concurrent rows can't
            # interleave a torn line even without an explicit lock.
            checkpoint.write(json.dumps(entry, ensure_ascii=False) + "\n")
            checkpoint.flush()

    try:
        await asyncio.gather(*(handle(i, row) for i, row in enumerate(rows)))
    finally:
        if checkpoint is not None:
            checkpoint.close()
    return [entry for entry in results if entry is not None]


async def run(args) -> int:
    df = load_dataframe(args.train_set)
    sample = df.sample(n=min(args.n_samples, len(df)), seed=args.seed)
    logger.info(f"Verifying {len(sample)} train labels with MatchVerifier")

    graph = Graph(neo4j_config)
    verifier = MatchVerifier(graph)

    session_id = f"match_verifier_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    checkpoint_dir = args.checkpoint_dir or args.output
    checkpoint_path = os.path.join(checkpoint_dir, "match_verifier_eval.checkpoint.jsonl")
    results = await verify_rows(
        verifier,
        sample.to_dicts(),
        args.text_column,
        args.code_column,
        concurrency=args.concurrency,
        checkpoint_path=checkpoint_path,
        session_id=session_id,
    )

    details = pl.DataFrame(
        results,
        schema=[
            (TEXT_COLUMN_OUT, pl.Utf8),
            (CODE_COLUMN_OUT, pl.Utf8),
            (VERDICT_COLUMN, pl.Boolean),
            (EXPLANATION_COLUMN, pl.Utf8),
            (CONFIDENCE_COLUMN, pl.Float64),
            (DURATION_COLUMN, pl.Float64),
        ],
    )
    n = len(details)
    n_skipped = len(sample) - n
    n_flagged = int((~details[VERDICT_COLUMN]).sum()) if n else 0
    summary = {
        "n": n,
        "n_skipped": n_skipped,
        "n_flagged": n_flagged,
        "agreement_rate": (1 - n_flagged / n) if n else None,
        "mean_confidence": float(details[CONFIDENCE_COLUMN].mean()) if n else None,
        "mean_duration_s": float(details[DURATION_COLUMN].mean()) if n else None,
        "total_duration_s": float(details[DURATION_COLUMN].sum()) if n else None,
        "session_id": session_id,
        "seed": args.seed,
    }

    storage.makedirs(args.output)
    details_path = os.path.join(args.output, "match_verifier_eval.parquet")
    with storage.open_path(details_path, "wb") as f:
        details.write_parquet(f)
    summary_path = os.path.join(args.output, "match_verifier_eval_summary.json")
    with storage.open_path(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))

    flagged = details.filter(~pl.col(VERDICT_COLUMN))
    if len(flagged):
        print("\nFlagged rows (MatchVerifier thinks the label may be wrong):")
        for r in flagged.to_dicts():
            print(f"  {r[TEXT_COLUMN_OUT]!r} -> {r[CODE_COLUMN_OUT]}: {r[EXPLANATION_COLUMN]}")

    logger.info(f"Details written to {details_path}, summary to {summary_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify a sample of train-set labels with MatchVerifier"
    )
    parser.add_argument(
        "--train-set", default=TRAIN_SET_PATH, help="Train parquet (local path or S3 key)"
    )
    parser.add_argument(
        "--text-column", default="libelle", help="Input text column (default: libelle)"
    )
    parser.add_argument(
        "--code-column", default="nace2025", help="Input label column (default: nace2025)"
    )
    parser.add_argument(
        "--n-samples", type=int, default=500, help="Number of rows to sample (default: 500)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed (default: 42)")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Max MatchVerifier calls in flight at once (default: 5)",
    )
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT, help=f"Output directory (default: {DEFAULT_OUTPUT})"
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Directory for the per-row checkpoint (default: --output). Worth pointing at a "
        "local directory when --output is on S3: s3fs only uploads on close, so a checkpoint "
        "written there survives nothing that the final parquet wouldn't have survived anyway",
    )
    args = parser.parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
