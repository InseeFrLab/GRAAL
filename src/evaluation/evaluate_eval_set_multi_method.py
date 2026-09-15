"""Multi-méthode + arbitrage sur un échantillon du jeu d'évaluation, chronométré.

Applique à un échantillon du jeu d'évaluation (cf. build_eval_set.py) le même
principe d'audit que src.evaluation.verify_train_labels applique au jeu
d'entraînement — juger la vérité terrain, reclassifier, arbitrer — mais sur
les 4 méthodes de classification à la fois plutôt qu'une seule, et avec le
temps de chaque appel (Navigator, Agentic-RAG, Summary, Supervisé,
MatchVerifier, CodeChooser) enregistré pour estimer le coût en temps de
chaque méthode. Chaque ligne exécute son propre pipeline complet (vérifier la
vérité terrain -> classifier -> vérifier chaque prédiction -> arbitrer) de
bout en bout ; les lignes tournent concurremment les unes des autres, mais
--concurrency borne le nombre total d'appels LLM en vol à un instant donné,
tous types et toutes lignes confondus (pas un plafond par étape).

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

Chaque ligne, dès que son propre pipeline termine, est journalisée au fil de
l'eau dans <output-dir>/eval_multi_method.checkpoint.jsonl (flush immédiat) :
un run interrompu ne perd que les lignes encore en vol, pas tout le batch —
même rationale que run_eval.py/verify_train_labels.py.

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


async def process_row(
    *,
    idx: int,
    total: int,
    activity: str,
    code,
    row_id: str,
    text_column: str,
    code_column: str,
    session_id: str,
    verifier: MatchVerifier,
    graph: Graph,
    semaphore: asyncio.Semaphore,
) -> tuple[dict, dict]:
    """Run one row's whole pipeline (verify ground truth -> classify -> verify each
    prediction -> arbitrate) end to end, so the caller can checkpoint it the moment
    it finishes instead of waiting for every row to clear the same stage — unlike
    the old stage-batched design, an interrupted run here only loses rows still in
    flight, not the whole batch (cf. module docstring).

    `semaphore` throttles individual LLM calls across the whole run — any kind, any
    row — to at most `--concurrency` in flight at once: the same knob as before,
    just no longer scoped to a single stage, since a stage boundary is no longer
    meaningful once rows progress independently. Each classifier call builds its
    own Navigator/classifier instance (cf. src.main), so concurrent calls don't
    share mutable state; MatchVerifier and CodeChooser share one Graph, but Neo4j
    drivers are safe for concurrent use across sessions.

    Returns (row, meta): `row` is the persisted record (identical shape to the
    previous stage-batched version); `meta` carries each classifier's
    proposed_explanation, needed only for non_final_fallback_rate in the summary
    and not otherwise persisted.
    """

    async def guarded(name: str, thunk):
        async with semaphore:
            return await timed_call(name, thunk, idx, total)

    gt_verification, gt_duration = await guarded(
        "match_verifier",
        traced(
            (lambda vi=MatchVerificationInput(activity=activity, code=str(code)): verifier(vi)),
            session_id=session_id,
            row_id=row_id,
            tag="match_verifier",
        ),
    )

    classify_results = await asyncio.gather(
        *(
            guarded(
                name,
                traced(
                    (lambda fn=fn: fn(activity)), session_id=session_id, row_id=row_id, tag=name
                ),
            )
            for name, fn in CLASSIFY_METHODS.items()
        )
    )
    preds = dict(zip(CLASSIFY_METHODS, (r for r, _ in classify_results)))
    durations = dict(zip(CLASSIFY_METHODS, (d for _, d in classify_results)))

    # Verify each method's own prediction the same way as the ground truth above —
    # skipping methods that produced no code, since there's nothing to verify.
    # Predictions already carry (code, proposed_explanation, proposed_confidence) as
    # a MatchVerificationInput, so they're passed to `verifier` unchanged.
    verify_names = [name for name in CLASSIFY_METHODS if getattr(preds[name], "code", None)]
    verify_results = await asyncio.gather(
        *(
            guarded(
                f"{name}_verify",
                traced(
                    (lambda pred=preds[name]: verifier(pred)),
                    session_id=session_id,
                    row_id=row_id,
                    tag=f"{name}_verify",
                ),
            )
            for name in verify_names
        )
    )
    verifications = dict(zip(verify_names, (r for r, _ in verify_results)))
    verify_durations = dict(zip(verify_names, (d for _, d in verify_results)))

    # Arbitrate with CodeChooser among {ground truth} ∪ {method predictions}, deduped
    # by normalized code — skipped when they already all agree.
    raw_candidates = [code] + [getattr(preds[name], "code", None) for name in CLASSIFY_METHODS]
    unique: dict[str, str] = {}
    for c in raw_candidates:
        norm = normalize_code(c)
        if norm and norm not in unique:
            unique[norm] = c
    codes = list(unique.values())

    choice, chooser_duration = None, None
    if len(codes) >= 2:
        choice, chooser_duration = await guarded(
            "code_chooser",
            traced(
                (
                    lambda: CodeChooser(graph, num_choices=len(codes))(
                        activity=activity, codes=codes
                    )
                ),
                session_id=session_id,
                row_id=row_id,
                tag="code_chooser",
            ),
        )

    row = {
        text_column: activity,
        code_column: code,
        "ground_truth_is_match": getattr(gt_verification, "is_match", None),
        "ground_truth_verify_confidence": getattr(gt_verification, "confidence", None),
        "ground_truth_verify_explanation": getattr(gt_verification, "explanation", None),
        "ground_truth_verify_duration_s": gt_duration,
        "ground_truth_verify_tool_calls": getattr(gt_verification, "tool_call_count", None),
        "ground_truth_verify_attempt_count": getattr(gt_verification, "attempt_count", None),
        "n_unique_candidates": len(codes),
        "chosen_code": getattr(choice, "chosen_code", None),
        "chooser_confidence": getattr(choice, "confidence", None),
        "chooser_explanation": getattr(choice, "explanation", None),
        "chooser_duration_s": chooser_duration,
        "chooser_tool_calls": getattr(choice, "tool_call_count", None),
        "chooser_attempt_count": getattr(choice, "attempt_count", None),
    }
    meta = {}
    for name in CLASSIFY_METHODS:
        pred = preds[name]
        row[f"{name}_code"] = getattr(pred, "code", None)
        row[f"{name}_duration_s"] = durations[name]
        row[f"{name}_tool_call_count"] = getattr(pred, "tool_call_count", None)
        row[f"{name}_attempt_count"] = getattr(pred, "attempt_count", None)
        method_verification = verifications.get(name)
        row[f"{name}_is_match"] = getattr(method_verification, "is_match", None)
        row[f"{name}_verify_confidence"] = getattr(method_verification, "confidence", None)
        row[f"{name}_verify_explanation"] = getattr(method_verification, "explanation", None)
        row[f"{name}_verify_duration_s"] = verify_durations.get(name)
        row[f"{name}_verify_tool_calls"] = getattr(method_verification, "tool_call_count", None)
        row[f"{name}_verify_attempt_count"] = getattr(method_verification, "attempt_count", None)
        meta[name] = getattr(pred, "proposed_explanation", None)

    return row, meta


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
    semaphore = asyncio.Semaphore(args.concurrency)
    total = len(labels)

    storage.makedirs(args.output_dir)
    checkpoint_path = os.path.join(args.output_dir, "eval_multi_method.checkpoint.jsonl")
    rows: list[dict | None] = [None] * total
    metas: list[dict | None] = [None] * total

    # Each row is checkpointed (flushed) the moment its own pipeline finishes, not
    # after every row reaches the same stage — write()+flush() here never straddles
    # an `await`, so concurrent rows finishing at different times can't interleave
    # a torn line even without an explicit lock.
    with storage.open_path(checkpoint_path, "w", encoding="utf-8") as ckpt:

        async def handle(i: int) -> None:
            row, meta = await process_row(
                idx=i + 1,
                total=total,
                activity=labels[i],
                code=truth[i],
                row_id=row_ids[i],
                text_column=args.text_column,
                code_column=args.code_column,
                session_id=session_id,
                verifier=verifier,
                graph=graph,
                semaphore=semaphore,
            )
            rows[i] = row
            metas[i] = meta
            ckpt.write(json.dumps(row, ensure_ascii=False) + "\n")
            ckpt.flush()

        await asyncio.gather(*(handle(i) for i in range(total)))

    details = pl.DataFrame(rows)
    details_path = os.path.join(args.output_dir, "eval_multi_method.parquet")
    with storage.open_path(details_path, "wb") as f:
        details.write_parquet(f)

    def stats(durations: list) -> dict | None:
        known = [d for d in durations if d is not None]
        if not known:
            return None
        total_s = sum(known)
        return {"n": len(known), "total_s": total_s, "mean_s": total_s / len(known)}

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
            "match_verifier": stats([r["ground_truth_verify_duration_s"] for r in rows]),
            **{name: stats([r[f"{name}_duration_s"] for r in rows]) for name in CLASSIFY_METHODS},
            **{
                f"{name}_verify": stats([r[f"{name}_verify_duration_s"] for r in rows])
                for name in CLASSIFY_METHODS
            },
            "code_chooser": stats([r["chooser_duration_s"] for r in rows]),
        },
        "non_final_fallback_rate": {
            name: fallback_rate([m[name] for m in metas]) for name in CLASSIFY_METHODS
        },
        "tool_calls": {
            "match_verifier": tool_call_stats([r["ground_truth_verify_tool_calls"] for r in rows]),
            **{
                name: tool_call_stats([r[f"{name}_tool_call_count"] for r in rows])
                for name in CLASSIFY_METHODS
            },
            **{
                f"{name}_verify": tool_call_stats([r[f"{name}_verify_tool_calls"] for r in rows])
                for name in CLASSIFY_METHODS
            },
            "code_chooser": tool_call_stats([r["chooser_tool_calls"] for r in rows]),
        },
        "retry_rate": {
            "match_verifier": retry_rate([r["ground_truth_verify_attempt_count"] for r in rows]),
            **{
                name: retry_rate([r[f"{name}_attempt_count"] for r in rows])
                for name in CLASSIFY_METHODS
            },
            **{
                f"{name}_verify": retry_rate([r[f"{name}_verify_attempt_count"] for r in rows])
                for name in CLASSIFY_METHODS
            },
            "code_chooser": retry_rate([r["chooser_attempt_count"] for r in rows]),
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
        help="Max in-flight LLM calls at once, across the whole run (default: 6)",
    )
    args = parser.parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
