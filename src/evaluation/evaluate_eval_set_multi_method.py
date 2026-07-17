"""Multi-méthode + arbitrage sur un échantillon du jeu d'évaluation, chronométré.

Applique à un échantillon du jeu d'évaluation (cf. build_eval_set.py) le même
principe d'audit que src.evaluation.verify_train_labels applique au jeu
d'entraînement — juger la vérité terrain, reclassifier, arbitrer — mais en
mode batch concurrent plutôt que séquentiel, sur les 4 méthodes de
classification à la fois plutôt qu'une seule, et avec le temps de chaque appel
(Navigator, Agentic-RAG, Summary, Supervisé, MatchVerifier, CodeChooser)
enregistré pour estimer le coût en temps de chaque méthode.

Sur un même échantillon (taille et graine fixées pour la reproductibilité) :
  1. MatchVerifier juge si le label de vérité terrain (apet2025) semble correct
     (le jeu d'évaluation n'est pas plus digne de confiance a priori que le
     jeu d'entraînement — cf. verify_train_labels.py)
  2. Les 4 méthodes de classification (Navigator, Agentic-RAG, Summary,
     modèle supervisé de production) proposent chacune un code
  3. MatchVerifier juge, de la même façon qu'à l'étape 1, chacune des 4
     prédictions (lignes sans code sautées : rien à vérifier) — le jugement
     humain porte ensuite sur les 5 candidats (vérité terrain + 4 méthodes) à
     égalité, chacun avec son propre verdict MatchVerifier
  4. CodeChooser arbitre entre la vérité terrain et les 4 prédictions, sauf si
     elles s'accordent déjà toutes sur un seul code

Chaque ligne traitée est journalisée au fil de l'eau dans
<output-dir>/eval_multi_method.checkpoint.jsonl (flush immédiat), pour ne
perdre que le travail non flush si le run est interrompu — même rationale que
run_eval.py/verify_train_labels.py.

Nécessite à l'exécution : la base Neo4j et l'API LLM configurées dans
l'environnement (mêmes prérequis que src.main), ainsi que le résumé NACE
(uv run -m src.neo4j_graph.build_nace_summary) pour SummaryAgenticClassifier.

Usage :
    uv run -m src.evaluation.evaluate_eval_set_multi_method \
        --n-samples 60 --seed 123 --output-dir data/eval/multi_method
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

from src.agents.closers.code_chooser import CodeChooser
from src.agents.closers.match_verifier import MatchVerificationInput, MatchVerifier
from src.config import neo4j_config
from src.evaluation.config import PATH_EVAL_OUTPUT
from src.evaluation.metrics import fallback_rate, normalize_code, retry_rate
from src.evaluation.row_id import row_id_for
from src.main import classify_agentic_rag, classify_navigator, classify_summary, classify_supervised
from src.neo4j_graph.graph import Graph
from src.utils import storage
from src.utils.logging import configure_logging
from src.utils.retry import call_with_retries

configure_logging()
logger = logging.getLogger(__name__)

CLASSIFY_METHODS = {
    "navigator": classify_navigator,
    "agentic_rag": classify_agentic_rag,
    "summary": classify_summary,
    "supervised": classify_supervised,
}


async def timed_call(name: str, thunk, idx: int, total: int):
    """Await the zero-arg async `thunk()`, returning `(result, duration_seconds)`.

    Retried once on failure (cf. call_with_retries — same harmonized policy as
    run_eval.py and verify_train_labels.py); `result` is None if every attempt
    failed, rather than raising, so one bad label/row doesn't abort the whole run.
    """
    start = time.perf_counter()
    result = await call_with_retries(thunk, what=f"[{name}] {idx}/{total}")
    duration = time.perf_counter() - start
    logger.info(f"[{name}] {idx}/{total} ({duration:5.1f}s)")
    return result, duration


async def run_batched(name: str, thunks: list, concurrency: int):
    """Run zero-arg async `thunks` with at most `concurrency` in flight at once.

    Each classifier call builds its own Navigator/classifier instance (cf.
    src.main), so concurrent calls don't share mutable state; MatchVerifier
    and CodeChooser share one Graph, but Neo4j drivers are safe for concurrent
    use across sessions. Order of results matches the order of `thunks`.
    """
    semaphore = asyncio.Semaphore(concurrency)
    total = len(thunks)

    async def guarded(idx: int, thunk):
        async with semaphore:
            return await timed_call(name, thunk, idx, total)

    results = await asyncio.gather(*(guarded(i, t) for i, t in enumerate(thunks, 1)))
    preds = [r for r, _ in results]
    durations = [d for _, d in results]
    return preds, durations


def traced(thunk, *, session_id: str, row_id: str, tag: str):
    """Wrap a zero-arg async thunk so its Langfuse trace is tagged with `row_id` (to
    correlate this specific call — ground truth verify, a classifier, its own
    verify, or CodeChooser — with the eval row it belongs to) and grouped under
    `session_id` (marking it as part of this campaign, filterable apart from ad hoc
    CLI usage) — same rationale as run_eval.py's identical wrapper.
    """

    async def _call():
        with propagate_attributes(
            session_id=session_id,
            metadata={"row_id": row_id},
            tags=["eval", "multi_method_eval", tag],
        ):
            return await thunk()

    return _call


async def run(args) -> int:
    with storage.open_path(args.eval_set, "rb") as f:
        df = pl.read_parquet(f)
    sample = df.sample(n=args.n_samples, seed=args.seed)
    labels = sample[args.text_column].to_list()
    truth = sample[args.code_column].to_list()
    logger.info(f"Sampled {len(labels)} rows from {args.eval_set} (seed={args.seed})")

    # Same rationale as run_eval.py: group this campaign's Langfuse traces under one
    # session_id, and tag each call with the row_id it belongs to (row_id_for is the
    # same hash the review app uses, so a trace, a parquet row, and a human judgment
    # for the same activity all correlate by this one value).
    session_id = f"eval_multi_method_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    row_ids = [row_id_for(labels[i], truth[i]) for i in range(len(labels))]

    graph = Graph(neo4j_config)
    verifier = MatchVerifier(graph)

    # Step 1: is the ground truth itself trustworthy? (cf. verify_train_labels.py)
    verify_thunks = [
        traced(
            (lambda vi=MatchVerificationInput(activity=a, code=str(c)): verifier(vi)),
            session_id=session_id,
            row_id=row_ids[i],
            tag="match_verifier",
        )
        for i, (a, c) in enumerate(zip(labels, truth))
    ]
    verifications, verify_durations = await run_batched(
        "match_verifier", verify_thunks, args.concurrency
    )

    # Step 2: each classification method proposes a code. Concurrency is bounded
    # within a method (each call builds its own Navigator/classifier instance,
    # cf. src.main, so concurrent calls don't share mutable state); methods run
    # one after another so log output and failures stay easy to attribute.
    all_preds: dict[str, list] = {}
    all_durations: dict[str, list] = {}
    for name, fn in CLASSIFY_METHODS.items():
        thunks = [
            traced(
                (lambda label=label: fn(label)),
                session_id=session_id,
                row_id=row_ids[i],
                tag=name,
            )
            for i, label in enumerate(labels)
        ]
        preds, durations = await run_batched(name, thunks, args.concurrency)
        all_preds[name] = preds
        all_durations[name] = durations

    # Step 3: verify each method's own prediction the same way as the ground truth
    # (step 1) — skipping rows where the method produced no code, since there's
    # nothing to verify. Predictions already carry (code, proposed_explanation,
    # proposed_confidence) as a MatchVerificationInput, so they're passed to
    # `verifier` unchanged.
    all_verifications: dict[str, list] = {}
    all_verify_durations: dict[str, list] = {}
    for name in CLASSIFY_METHODS:
        preds = all_preds[name]
        verifiable_at = [i for i, p in enumerate(preds) if getattr(p, "code", None)]
        thunks = [
            traced(
                (lambda pred=preds[i]: verifier(pred)),
                session_id=session_id,
                row_id=row_ids[i],
                tag=f"{name}_verify",
            )
            for i in verifiable_at
        ]
        verified, verify_durs = await run_batched(f"{name}_verify", thunks, args.concurrency)
        verifications_by_row = dict(zip(verifiable_at, verified))
        durations_by_row = dict(zip(verifiable_at, verify_durs))
        all_verifications[name] = [verifications_by_row.get(i) for i in range(len(preds))]
        all_verify_durations[name] = [durations_by_row.get(i) for i in range(len(preds))]

    # Step 4: arbitrate with CodeChooser among {ground truth} ∪ {method predictions},
    # deduped by normalized code — skipped when they already all agree. Candidate
    # dedup is pure/cheap, so it's done upfront for every row before batching only
    # the rows that actually need arbitration.
    candidates_per_row = []
    for i in range(len(labels)):
        raw_candidates = [truth[i]] + [
            getattr(all_preds[name][i], "code", None) for name in CLASSIFY_METHODS
        ]
        unique = {}
        for code in raw_candidates:
            norm = normalize_code(code)
            if norm and norm not in unique:
                unique[norm] = code
        candidates_per_row.append(list(unique.values()))

    arbitrate_at = [i for i, codes in enumerate(candidates_per_row) if len(codes) >= 2]
    chooser_thunks = [
        traced(
            (
                lambda activity=labels[i], codes=candidates_per_row[i]: CodeChooser(
                    graph, num_choices=len(codes)
                )(activity=activity, codes=codes)
            ),
            session_id=session_id,
            row_id=row_ids[i],
            tag="code_chooser",
        )
        for i in arbitrate_at
    ]
    choices, sparse_chooser_durations = await run_batched(
        "code_chooser", chooser_thunks, args.concurrency
    )
    choice_by_row: dict[int, tuple] = dict(
        zip(arbitrate_at, zip(choices, sparse_chooser_durations))
    )

    storage.makedirs(args.output_dir)
    checkpoint_path = os.path.join(args.output_dir, "eval_multi_method.checkpoint.jsonl")
    rows = []
    with storage.open_path(checkpoint_path, "w", encoding="utf-8") as ckpt:
        for i, activity in enumerate(labels):
            codes = candidates_per_row[i]
            choice, chooser_duration = choice_by_row.get(i, (None, None))

            verification = verifications[i]
            row = {
                args.text_column: activity,
                args.code_column: truth[i],
                "ground_truth_is_match": getattr(verification, "is_match", None),
                "ground_truth_verify_confidence": getattr(verification, "confidence", None),
                "ground_truth_verify_explanation": getattr(verification, "explanation", None),
                "ground_truth_verify_duration_s": verify_durations[i],
                "ground_truth_verify_tool_calls": getattr(verification, "tool_call_count", None),
                "ground_truth_verify_attempt_count": getattr(verification, "attempt_count", None),
                "n_unique_candidates": len(codes),
                "chosen_code": getattr(choice, "chosen_code", None),
                "chooser_confidence": getattr(choice, "confidence", None),
                "chooser_explanation": getattr(choice, "explanation", None),
                "chooser_duration_s": chooser_duration,
                "chooser_tool_calls": getattr(choice, "tool_call_count", None),
                "chooser_attempt_count": getattr(choice, "attempt_count", None),
            }
            for name in CLASSIFY_METHODS:
                row[f"{name}_code"] = getattr(all_preds[name][i], "code", None)
                row[f"{name}_duration_s"] = all_durations[name][i]
                row[f"{name}_tool_call_count"] = getattr(
                    all_preds[name][i], "tool_call_count", None
                )
                row[f"{name}_attempt_count"] = getattr(all_preds[name][i], "attempt_count", None)
                method_verification = all_verifications[name][i]
                row[f"{name}_is_match"] = getattr(method_verification, "is_match", None)
                row[f"{name}_verify_confidence"] = getattr(method_verification, "confidence", None)
                row[f"{name}_verify_explanation"] = getattr(
                    method_verification, "explanation", None
                )
                row[f"{name}_verify_duration_s"] = all_verify_durations[name][i]
                row[f"{name}_verify_tool_calls"] = getattr(
                    method_verification, "tool_call_count", None
                )
                row[f"{name}_verify_attempt_count"] = getattr(
                    method_verification, "attempt_count", None
                )

            rows.append(row)
            ckpt.write(json.dumps(row, ensure_ascii=False) + "\n")
            ckpt.flush()
    chooser_durations = sparse_chooser_durations

    details = pl.DataFrame(rows)
    details_path = os.path.join(args.output_dir, "eval_multi_method.parquet")
    with storage.open_path(details_path, "wb") as f:
        details.write_parquet(f)

    def stats(durations: list[float]) -> dict | None:
        if not durations:
            return None
        total = sum(durations)
        return {"n": len(durations), "total_s": total, "mean_s": total / len(durations)}

    def tool_call_stats(counts: list) -> dict | None:
        """Sanity check, not an accuracy metric: how much tool use actually happened
        (e.g. confirming Navigator/Agentic-RAG/Summary explore the graph rather than
        finalizing on zero tool calls). `None` counts (methods that don't report it,
        or rows skipped upstream) are excluded rather than counted as zero.
        """
        known = [c for c in counts if c is not None]
        if not known:
            return None
        return {"n": len(known), "total": sum(known), "mean": sum(known) / len(known)}

    def chooser_agrees(row: dict) -> bool:
        if not row["chosen_code"]:
            return False
        return normalize_code(row["chosen_code"]) == normalize_code(row[args.code_column])

    n_agree_truth = sum(1 for r in rows if chooser_agrees(r))
    n_ground_truth_confirmed = sum(1 for r in rows if r["ground_truth_is_match"])
    summary = {
        "n_rows": len(rows),
        "seed": args.seed,
        "timings": {
            "match_verifier": stats(verify_durations),
            **{name: stats(all_durations[name]) for name in CLASSIFY_METHODS},
            **{
                f"{name}_verify": stats([d for d in all_verify_durations[name] if d is not None])
                for name in CLASSIFY_METHODS
            },
            "code_chooser": stats(chooser_durations),
        },
        "non_final_fallback_rate": {
            name: fallback_rate([getattr(p, "proposed_explanation", None) for p in all_preds[name]])
            for name in CLASSIFY_METHODS
        },
        "tool_calls": {
            "match_verifier": tool_call_stats(
                [getattr(v, "tool_call_count", None) for v in verifications]
            ),
            **{
                name: tool_call_stats(
                    [getattr(p, "tool_call_count", None) for p in all_preds[name]]
                )
                for name in CLASSIFY_METHODS
            },
            **{
                f"{name}_verify": tool_call_stats(
                    [getattr(v, "tool_call_count", None) for v in all_verifications[name]]
                )
                for name in CLASSIFY_METHODS
            },
            "code_chooser": tool_call_stats([getattr(c, "tool_call_count", None) for c in choices]),
        },
        "retry_rate": {
            "match_verifier": retry_rate(
                [getattr(v, "attempt_count", None) for v in verifications]
            ),
            **{
                name: retry_rate([getattr(p, "attempt_count", None) for p in all_preds[name]])
                for name in CLASSIFY_METHODS
            },
            **{
                f"{name}_verify": retry_rate(
                    [getattr(v, "attempt_count", None) for v in all_verifications[name]]
                )
                for name in CLASSIFY_METHODS
            },
            "code_chooser": retry_rate([getattr(c, "attempt_count", None) for c in choices]),
        },
        "n_ground_truth_confirmed": n_ground_truth_confirmed,
        "n_arbitrated": sum(1 for r in rows if r["n_unique_candidates"] >= 2),
        "n_chooser_agrees_with_ground_truth": n_agree_truth,
    }
    summary_path = os.path.join(args.output_dir, "eval_multi_method_summary.json")
    with storage.open_path(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    storage.remove(checkpoint_path)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    logger.info(f"Details written to {details_path}, summary to {summary_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run all 4 classification methods + MatchVerifier + CodeChooser on a "
        "sample of the eval set, with per-call timing"
    )
    parser.add_argument(
        "--eval-set", default=PATH_EVAL_OUTPUT, help="Evaluation parquet (cf. build_eval_set)"
    )
    parser.add_argument("--n-samples", type=int, default=60, help="Sample size (default: 60)")
    parser.add_argument("--seed", type=int, default=123, help="Sampling seed (default: 123)")
    parser.add_argument("--text-column", default="libelle", help="Text column (default: libelle)")
    parser.add_argument(
        "--code-column", default="apet2025", help="Ground-truth column (default: apet2025)"
    )
    parser.add_argument("--output-dir", default="data/eval/multi_method", help="Output directory")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=6,
        help="Max in-flight calls per stage (default: 6)",
    )
    args = parser.parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
