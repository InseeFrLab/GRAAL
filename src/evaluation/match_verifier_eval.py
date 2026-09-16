"""Audit de la qualité des labels du jeu d'entraînement via le MatchVerifier.

Tire un échantillon aléatoire du jeu d'entraînement et demande au MatchVerifier
s'il pense que le code associé à chaque libellé est correct. Ne compare pas à
une prédiction de modèle : c'est un contrôle qualité du label de référence
lui-même (cf. src.evaluation.run_eval pour l'évaluation d'un classifieur), et
surtout la matière première de l'évaluation du MatchVerifier lui-même — le
parquet produit ici est relu par src.evaluation.apps.match_verifier_eval_app,
où des annotateurs humains jugent chaque verdict.

Le parquet de sortie tient en dix colonnes, volontairement : libelle,
current_code, puis tout ce que le MatchVerifier produit sur la paire —
match_verifier_verdict (is_match), match_verifier_match_score,
match_verifier_alternative_code, match_verifier_alternative_score,
match_verifier_explanation, match_verifier_is_match_score — plus
match_verifier_p_match (la probabilité du verdict lue dans les logprobs, cf.
MatchVerifier) et match_verifier_duration_s (le temps d'inférence de l'appel, cf.
verify_rows).

Les quatre colonnes autour du verdict sont ce qui donne à l'annotateur de quoi
juger autre chose qu'une impression : le vérificateur note le code en place,
nomme le meilleur code concurrent qu'il ait trouvé — à chaque ligne, y compris
quand il valide (cf. MatchVerifier) — et note celui-là aussi. Un verdict devient
alors relisible : on voit contre quoi le code a été comparé, et de combien il a
gagné. `p_match`, lui, n'a de sens qu'une fois les annotations humaines
disponibles : c'est en le seuillant a posteriori qu'on choisit un point de
fonctionnement pour le vérificateur au lieu de subir celui de son prompt, rien de
plus (les versions précédentes de ce script demandaient en plus au
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

Ce que le script mesure, c'est le prompt du MatchVerifier, et ce prompt vit
dans le dépôt : les résultats sont donc écrits sous <output>/<commit>/<modèle>/
(le tag git de HEAD au moment du run et GENERATION_MODEL, cf.
src.utils.run_provenance) plutôt qu'à plat, et
le script refuse de démarrer si src/agents/closers/match_verifier.py a des
modifications non commitées — sans quoi le nom du dossier désignerait un commit
qui ne contient pas le prompt effectivement évalué. `--allow-dirty` lève le
refus au prix d'un dossier suffixé `-dirty`, pour itérer sur un prompt avant de
le commiter sans polluer les résultats de référence. Le modèle n'a pas de flag :
il est lu dans l'environnement que lisent les agents eux-mêmes, donc pour
comparer deux modèles, `GENERATION_MODEL=... uv run -m ...`.

Nécessite à l'exécution : la base Neo4j et l'API LLM configurées dans
l'environnement (mêmes prérequis que src.main).

Usage :
    uv run -m src.evaluation.match_verifier_eval \
        --n-samples 500 \
        --output s3://projet-ape/graal/data/eval/match_verifier_eval

Le millésime du jeu d'entraînement décide du nom de la colonne de label : `apet2025`
pour celui de TRAIN_SET_PATH (défaut), `nace2025` pour le millésime
08112022_27102024, d'où `--code-column`.
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
from src.utils.run_provenance import (
    DirtyWorkingTreeError,
    assert_clean,
    revision_sha,
    revision_tag,
    run_subpath,
)

configure_logging()
logger = logging.getLogger(__name__)

TRAIN_SET_PATH = "projet-ape/data/25062026/naf2025/split/df_train.parquet"
DEFAULT_OUTPUT = "s3://projet-ape/graal/data/eval/match_verifier_eval"

# Output file names, fixed: match_verifier_eval_app resolves exactly these under
# <output>/<commit>/<model>/.
DETAILS_FILENAME = "match_verifier_eval.parquet"
SUMMARY_FILENAME = "match_verifier_eval_summary.json"

# Files whose content *is* what this eval measures: a run made while these carry
# uncommitted edits cannot honestly be filed under the current commit (cf.
# src.utils.run_provenance). Extend this list if the prompt grows new inputs — the
# notice that makes up most of the prompt is composed by Graph.get_notice, the
# nomenclature summary that opens it by build_nace_summary.build_summary_text, and
# the toolset and turn budget the verifier runs under are decided in BaseAgent, so
# all four files define what a run measures.
PROMPT_FILES = [
    "src/agents/closers/match_verifier.py",
    "src/agents/base_agent.py",
    "src/neo4j_graph/graph.py",
    "src/neo4j_graph/build_nace_summary.py",
]

# Output column names, fixed: match_verifier_eval_app reads exactly these.
TEXT_COLUMN_OUT = "libelle"
CODE_COLUMN_OUT = "current_code"
VERDICT_COLUMN = "match_verifier_verdict"
MATCH_SCORE_COLUMN = "match_verifier_match_score"
ALTERNATIVE_CODE_COLUMN = "match_verifier_alternative_code"
ALTERNATIVE_SCORE_COLUMN = "match_verifier_alternative_score"
EXPLANATION_COLUMN = "match_verifier_explanation"
IS_MATCH_SCORE_COLUMN = "match_verifier_is_match_score"
P_MATCH_COLUMN = "match_verifier_p_match"
DURATION_COLUMN = "match_verifier_duration_s"

# Écart (alternative_score - match_score) à partir duquel on considère que le
# vérificateur a vraiment trouvé mieux. Reporté dans le résumé parce que c'est le
# chiffre qui dit si la recherche d'alternative sert à quelque chose : une
# alternative systématiquement notée loin derrière le code en place signifierait que
# le modèle la remplit pour la forme.
ALTERNATIVE_BEATS_MARGIN = 20

# Seuils auxquels le taux de rejet est reporté dans le résumé : `is_match` est un
# point de fonctionnement parmi d'autres, et p_match permet de les parcourir sans
# relancer le run. Un rejet au seuil t, c'est p_match < t.
P_MATCH_THRESHOLDS = (0.1, 0.25, 0.5, 0.75, 0.9)

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

    No `proposed_explanation`/`proposed_confidence` is passed: there is no model
    rationale to show, and the verifier's prompt says nothing about where a code comes
    from anyway — a training label and a classifier's guess are judged by the same
    words, on the pair alone (cf. MatchVerifier.build_prompt). Retries once
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
            MATCH_SCORE_COLUMN: verification.match_score,
            ALTERNATIVE_CODE_COLUMN: verification.alternative_code,
            ALTERNATIVE_SCORE_COLUMN: verification.alternative_score,
            EXPLANATION_COLUMN: verification.explanation,
            IS_MATCH_SCORE_COLUMN: verification.is_match_score,
            P_MATCH_COLUMN: verification.p_match,
            DURATION_COLUMN: duration,
        }
        results[i] = entry
        logger.info(
            f"{i + 1}/{total} ({duration:5.1f}s): {text!r} -> {code} "
            f"({verification.match_score}%) vs {verification.alternative_code} "
            f"({verification.alternative_score}%) : "
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
    # Before anything expensive: refuse a run whose results couldn't be attributed
    # to a commit, rather than discovering it after 500 LLM calls.
    assert_clean(PROMPT_FILES, allow_dirty=args.allow_dirty)
    commit = revision_tag(allow_dirty=args.allow_dirty)
    subpath = run_subpath(allow_dirty=args.allow_dirty)
    output_dir = os.path.join(args.output, subpath)
    logger.info(f"Pinning this run to {subpath}; writing to {output_dir}")

    df = load_dataframe(args.train_set)
    sample = df.sample(n=min(args.n_samples, len(df)), seed=args.seed)
    logger.info(f"Verifying {len(sample)} train labels with MatchVerifier")

    graph = Graph(neo4j_config)
    verifier = MatchVerifier(graph)

    session_id = f"match_verifier_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    # The checkpoint gets the same subpath, so a local --checkpoint-dir shared
    # across runs doesn't mix rows from two prompt revisions (or two models) into
    # one file.
    checkpoint_dir = (
        os.path.join(args.checkpoint_dir, subpath) if args.checkpoint_dir else output_dir
    )
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
            (MATCH_SCORE_COLUMN, pl.Int64),
            (ALTERNATIVE_CODE_COLUMN, pl.Utf8),
            (ALTERNATIVE_SCORE_COLUMN, pl.Int64),
            (EXPLANATION_COLUMN, pl.Utf8),
            (IS_MATCH_SCORE_COLUMN, pl.Int64),
            (P_MATCH_COLUMN, pl.Float64),
            (DURATION_COLUMN, pl.Float64),
        ],
    )
    n = len(details)
    n_skipped = len(sample) - n
    n_flagged = int((~details[VERDICT_COLUMN]).sum()) if n else 0
    p_match = details[P_MATCH_COLUMN].drop_nulls()
    summary = {
        "n": n,
        "n_skipped": n_skipped,
        "n_flagged": n_flagged,
        "agreement_rate": (1 - n_flagged / n) if n else None,
        # Ce que le taux de rejet deviendrait si le verdict était pris en seuillant
        # p_match plutôt qu'en lisant is_match : de quoi voir, avant même d'annoter,
        # si le prompt tranche net (taux stables d'un seuil à l'autre) ou si beaucoup
        # de lignes sont des quasi-égalités.
        "flag_rate_by_p_match_threshold": {
            str(t): float((p_match < t).mean()) for t in P_MATCH_THRESHOLDS
        }
        if len(p_match)
        else None,
        "mean_p_match": float(p_match.mean()) if len(p_match) else None,
        "n_missing_p_match": n - len(p_match),
        "mean_is_match_score": float(details[IS_MATCH_SCORE_COLUMN].mean()) if n else None,
        # Les trois chiffres qui disent si la recherche d'alternative fait son travail.
        # `alternative_differs_rate` doit rester très proche de 1 : le prompt exige une
        # alternative *différente* du code jugé à chaque ligne, donc un taux qui s'en
        # écarte est un manquement à la consigne, pas une propriété des données.
        # `alternative_beats_rate` est la part de lignes où le concurrent trouvé est
        # nettement mieux noté que le code en place, et devrait de près suivre le taux de
        # rejet ; un écart entre les deux signale un verdict qui ne suit pas ses propres
        # scores.
        "mean_match_score": float(details[MATCH_SCORE_COLUMN].mean()) if n else None,
        "mean_alternative_score": float(details[ALTERNATIVE_SCORE_COLUMN].mean()) if n else None,
        "alternative_differs_rate": float(
            (
                details[ALTERNATIVE_CODE_COLUMN].str.replace_all(r"\.", "")
                != details[CODE_COLUMN_OUT].str.replace_all(r"\.", "")
            ).mean()
        )
        if n
        else None,
        "alternative_beats_rate": float(
            (
                (details[ALTERNATIVE_SCORE_COLUMN] - details[MATCH_SCORE_COLUMN])
                >= ALTERNATIVE_BEATS_MARGIN
            ).mean()
        )
        if n
        else None,
        "mean_duration_s": float(details[DURATION_COLUMN].mean()) if n else None,
        "total_duration_s": float(details[DURATION_COLUMN].sum()) if n else None,
        "session_id": session_id,
        "seed": args.seed,
        "commit": commit,
        "commit_sha": revision_sha(),
        "model": os.environ["GENERATION_MODEL"],
    }

    storage.makedirs(output_dir)
    details_path = os.path.join(output_dir, DETAILS_FILENAME)
    with storage.open_path(details_path, "wb") as f:
        details.write_parquet(f)
    summary_path = os.path.join(output_dir, SUMMARY_FILENAME)
    with storage.open_path(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))

    flagged = details.filter(~pl.col(VERDICT_COLUMN))
    if len(flagged):
        print("\nFlagged rows (MatchVerifier thinks the label may be wrong):")
        for r in flagged.to_dicts():
            print(
                f"  {r[TEXT_COLUMN_OUT]!r} -> {r[CODE_COLUMN_OUT]} "
                f"({r[MATCH_SCORE_COLUMN]}%) vs {r[ALTERNATIVE_CODE_COLUMN]} "
                f"({r[ALTERNATIVE_SCORE_COLUMN]}%): {r[EXPLANATION_COLUMN]}"
            )

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
        "--code-column",
        default="apet2025",
        help="Input label column (default: apet2025, the label column of --train-set's "
        "default; the 08112022_27102024 vintage names the same column nace2025)",
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
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Root output directory (default: {DEFAULT_OUTPUT}). Results go to "
        "<output>/<commit>/<model>/, so runs of successive prompt revisions — or of two "
        "models on the same prompt — never overwrite each other",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Root directory for the per-row checkpoint, also suffixed with /<commit>/<model> "
        "(default: --output). Worth pointing at a local directory when --output is on S3: "
        "s3fs only uploads on close, so a checkpoint written there survives nothing that "
        "the final parquet wouldn't have survived anyway",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Run even though the prompt files have uncommitted changes "
        f"({', '.join(PROMPT_FILES)}), writing to a scratch <commit>-dirty/<model> directory. "
        "For iterating on a prompt before committing it; the results are not "
        "reproducible from any commit",
    )
    args = parser.parse_args()
    try:
        return asyncio.run(run(args))
    except DirtyWorkingTreeError as exc:
        # A refusal by design, not a crash: print the (actionable) message alone
        # rather than a traceback that buries it.
        logger.error(str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
