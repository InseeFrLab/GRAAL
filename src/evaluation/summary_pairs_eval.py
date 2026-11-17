"""Classe avec SummaryAgenticClassifier les libellés d'un run de match_verifier_eval.

Ajoute un second classifieur aux paires déjà jugées par le MatchVerifier : pour chaque
libellé distinct du parquet d'un run (`match_verifier_eval.parquet`), le code que
retient SummaryAgenticClassifier, à comparer au code humain et au code jugé (Sirus
pour la COICOP). Le Summary sert de point de comparaison au vérificateur : un
désaccord Summary / code jugé est lui aussi un signal d'erreur, dont on peut mesurer
le rappel et la précision comme ceux d'un rejet du vérificateur.

Le classifieur ne voit que le libellé, comme le vérificateur. Les libellés tournent
concurremment (`--concurrency`), chaque appel est retenté une fois, et chaque
libellé classé est écrit au fil de l'eau dans un checkpoint JSONL. Contrairement à
match_verifier_eval, le checkpoint est **repris** au lancement : un run interrompu
ne reclasse que les libellés qui manquent, ce qui compte pour plusieurs milliers
d'appels multi-tours.

Nécessite : la base Neo4j et l'API LLM de l'environnement, le profil `NOMENCLATURE`
du graphe, et le résumé de la nomenclature généré depuis ce graphe :

    NOMENCLATURE=coicop2018 uv run -m src.neo4j_graph.build_nace_summary \
        --output data/coicop_summary.txt
    NOMENCLATURE=coicop2018 uv run -m src.evaluation.summary_pairs_eval \
        --run-dir data/eval/match_verifier_eval_coicop/<commit>/<modèle> \
        --summary-path data/coicop_summary.txt

Sortie : `summary_classifier_predictions.parquet` dans `--run-dir`, une ligne par
libellé (colonnes `libelle`, `summary_code`, `summary_confidence`,
`summary_explanation`, `summary_tool_calls`, `summary_fallback`,
`summary_duration_s`).
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

from src.agents.Text2Code.classifiers.summary_classifier import SummaryAgenticClassifier
from src.config import neo4j_config
from src.evaluation.row_id import row_id_for
from src.neo4j_graph.graph import Graph
from src.nomenclature import NOMENCLATURE
from src.utils.logging import configure_logging
from src.utils.retry import call_with_retries

configure_logging()
logger = logging.getLogger(__name__)

DETAILS_FILENAME = "match_verifier_eval.parquet"
OUTPUT_FILENAME = "summary_classifier_predictions.parquet"
CHECKPOINT_FILENAME = "summary_classifier_predictions.checkpoint.jsonl"
FALLBACK_MARKER = "[non-final fallback]"

SCHEMA = [
    ("libelle", pl.Utf8),
    ("summary_code", pl.Utf8),
    ("summary_confidence", pl.Float64),
    ("summary_explanation", pl.Utf8),
    ("summary_tool_calls", pl.Int64),
    ("summary_fallback", pl.Boolean),
    ("summary_duration_s", pl.Float64),
]

_MAX_ATTEMPTS = 2


def load_checkpoint(path: str) -> dict[str, dict]:
    """Les libellés déjà classés, par libellé ; une ligne tronquée (run tué en pleine
    écriture) est ignorée et sera reclassée."""
    done = {}
    if not os.path.exists(path):
        return done
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            done[entry["libelle"]] = entry
    return done


async def classify_labels(
    classifier: SummaryAgenticClassifier,
    labels: list[str],
    checkpoint_path: str,
    concurrency: int,
    session_id: str,
) -> None:
    semaphore = asyncio.Semaphore(concurrency)
    total = len(labels)

    with open(checkpoint_path, "a", encoding="utf-8") as checkpoint:

        async def handle(i: int, label: str) -> None:
            async def call():
                with propagate_attributes(
                    session_id=session_id,
                    metadata={"row_id": row_id_for(label, "")},
                    tags=["eval", "summary_pairs_eval"],
                ):
                    return await classifier(label)

            async with semaphore:
                start = time.perf_counter()
                result = await call_with_retries(
                    call, _MAX_ATTEMPTS, f"Summary {i + 1}/{total} for {label!r}"
                )
                duration = time.perf_counter() - start
            if result is None:
                logger.warning(f"Skipping {label!r}: SummaryAgenticClassifier never succeeded")
                return
            entry = {
                "libelle": label,
                "summary_code": result.code,
                "summary_confidence": result.proposed_confidence,
                "summary_explanation": result.proposed_explanation,
                "summary_tool_calls": result.tool_call_count,
                "summary_fallback": FALLBACK_MARKER in (result.proposed_explanation or ""),
                "summary_duration_s": duration,
            }
            logger.info(f"{i + 1}/{total} ({duration:5.1f}s): {label!r} -> {result.code}")
            # write()+flush() never straddles an `await`: no torn lines between rows.
            checkpoint.write(json.dumps(entry, ensure_ascii=False) + "\n")
            checkpoint.flush()

        await asyncio.gather(*(handle(i, label) for i, label in enumerate(labels)))


async def run(args) -> int:
    details = pl.read_parquet(os.path.join(args.run_dir, DETAILS_FILENAME))
    labels = details["libelle"].unique(maintain_order=True).to_list()
    if args.limit:
        labels = labels[: args.limit]
    checkpoint_path = os.path.join(args.run_dir, CHECKPOINT_FILENAME)
    done = load_checkpoint(checkpoint_path)
    todo = [label for label in labels if label not in done]
    logger.info(
        f"{len(labels)} labels, {len(done)} already in checkpoint, {len(todo)} to classify "
        f"({NOMENCLATURE.key}, summary {args.summary_path})"
    )

    classifier = SummaryAgenticClassifier(Graph(neo4j_config), summary_path=args.summary_path)
    session_id = f"summary_pairs_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    await classify_labels(classifier, todo, checkpoint_path, args.concurrency, session_id)

    done = load_checkpoint(checkpoint_path)
    predictions = pl.DataFrame(
        [done[label] for label in labels if label in done],
        schema=SCHEMA,
    )
    output_path = os.path.join(args.run_dir, OUTPUT_FILENAME)
    predictions.write_parquet(output_path)
    logger.info(
        f"{len(predictions)}/{len(labels)} labels classified "
        f"({len(labels) - len(predictions)} failed), written to {output_path}"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Classify the labels of a match_verifier_eval run with SummaryAgenticClassifier"
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help=f"Run directory holding {DETAILS_FILENAME} (<output>/<commit>/<model>)",
    )
    parser.add_argument(
        "--summary-path",
        required=True,
        help="Nomenclature summary built from the same graph (cf. build_nace_summary)",
    )
    parser.add_argument(
        "--concurrency", type=int, default=5, help="Max classifier calls in flight (default: 5)"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Only the first N distinct labels (for a trial)"
    )
    return asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
