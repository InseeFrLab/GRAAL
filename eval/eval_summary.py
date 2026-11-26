"""Évaluation de `SummaryAgenticClassifier` sur le golden set NAF 2025.

Classe un échantillon du golden set, compare au label de référence `apet2025`, et
trace pour chaque libellé la séquence d'appels d'outils de l'agent (nom + arguments)
— de quoi vérifier qu'il explore réellement le graphe au lieu de conclure depuis le
seul résumé.

Golden set : s3://projet-ape/data/golden_set_naf2025.parquet, documenté dans
s3://projet-ape/data/golden_set_naf2025_doc.txt. Par défaut, l'échantillon est tiré
dans `in_representative_subset` (distribution des codes proche de la production) ;
`--all` tire dans le fichier complet, où l'agriculture est très sur-représentée.

Prérequis : `.env` avec Neo4j, l'API LLM et les accès S3 (mêmes variables que src.main).

Ablation du vérificateur (`--verify`) : chaque prédiction est ensuite jugée par
`MatchVerifier`, et deux bras sont comparés —
    classifier          le code du classifieur, tel quel ;
    classifier+verifier le code du classifieur, remplacé par `alternative_code` du
                        vérificateur quand celui-ci le rejette (is_match = false) et
                        que l'alternative est un code terminal existant.
S'y ajoutent les métriques du vérificateur vu comme filtre (taux d'acceptation,
exactitude parmi les acceptés, rappel/précision de la détection d'erreurs).

Usage (depuis la racine du dépôt) :
    uv run python -m eval.eval_summary --n 10                 # debug
    uv run python -m eval.eval_summary --n 200 --concurrency 8 --verify

Sorties dans --out (défaut eval/results/summary_<horodatage>/) :
    predictions.jsonl  une ligne par libellé, écrite au fil de l'eau
    metrics.json       exactitude par niveau + statistiques d'usage des outils
"""

import argparse
import asyncio
import contextvars
import json
import logging
import os
import time
from collections import Counter
from datetime import datetime

import polars as pl
from dotenv import load_dotenv

load_dotenv(override=True)
# Les identifiants S3 du .env sont une paire key/secret statique : un AWS_SESSION_TOKEN
# résiduel (vide hérité du pod, ou temporaire d'une autre session) fait rejeter la
# signature par MinIO.
os.environ.pop("AWS_SESSION_TOKEN", None)

from agents import Runner  # noqa: E402
from agents.exceptions import AgentsException  # noqa: E402
from src.agents.closers.match_verifier import MatchVerifier  # noqa: E402
from src.agents.Text2Code.classifiers.summary_classifier import (  # noqa: E402
    DEFAULT_SUMMARY_PATH,
    SummaryAgenticClassifier,
)
from src.config import neo4j_config  # noqa: E402
from src.neo4j_graph.build_nace_summary import build_summary_text  # noqa: E402
from src.neo4j_graph.graph import Graph  # noqa: E402
from src.utils import storage  # noqa: E402
from src.utils.logging import configure_logging  # noqa: E402

configure_logging()
logger = logging.getLogger("eval.eval_summary")

GOLDEN_SET = "s3://projet-ape/data/golden_set_naf2025.parquet"
FALLBACK_MARKER = "[non-final fallback]"
DEPTHS = {"division": 2, "groupe": 3, "classe": 4, "sous-classe": None}

# --- Capture des appels d'outils -------------------------------------------------
# Chaque Runner.run (exploration, finalisation) ajoute ses items à la liste du
# libellé en cours. Un ContextVar plutôt qu'une liste globale : chaque tâche asyncio a
# sa propre copie du contexte, ce qui reste correct avec --concurrency > 1.
# Chaque Runner.run consigne aussi, dans `_runs`, sa durée et sa consommation de
# tokens : de quoi voir quelle phase (exploration, finalisation, vérificateur) coûte.
_trace: contextvars.ContextVar[list | None] = contextvars.ContextVar("trace", default=None)
_runs: contextvars.ContextVar[list | None] = contextvars.ContextVar("runs", default=None)
_original_run = Runner.run


def _run_record(agent, seconds: float, usage) -> dict:
    return {
        "agent": getattr(agent, "name", None),
        "tool_choice": getattr(getattr(agent, "model_settings", None), "tool_choice", None),
        "seconds": round(seconds, 1),
        "requests": getattr(usage, "requests", None),
        "input_tokens": getattr(usage, "input_tokens", None),
        "output_tokens": getattr(usage, "output_tokens", None),
    }


async def _tracing_run(agent, *args, **kwargs):
    sink, runs = _trace.get(), _runs.get()
    start = time.perf_counter()
    try:
        result = await _original_run(agent, *args, **kwargs)
    except AgentsException as e:
        if sink is not None and e.run_data is not None:
            sink.extend(e.run_data.new_items)
        if runs is not None:
            usage = e.run_data.context_wrapper.usage if e.run_data is not None else None
            runs.append(
                _run_record(agent, time.perf_counter() - start, usage) | {"error": type(e).__name__}
            )
        raise
    if sink is not None:
        sink.extend(result.new_items)
    if runs is not None:
        runs.append(_run_record(agent, time.perf_counter() - start, result.context_wrapper.usage))
    return result


Runner.run = _tracing_run


def tool_calls(items) -> list[dict]:
    calls = [
        {
            "name": getattr(i.raw_item, "name", None),
            "arguments": getattr(i.raw_item, "arguments", None),
        }
        for i in items
        if i.type == "tool_call_item"
    ]
    outputs = [str(getattr(i, "output", "")) for i in items if i.type == "tool_call_output_item"]
    for call, out in zip(calls, outputs):
        call["unknown_code"] = "not found" in out[:200]
    return calls


def normalize(code) -> str | None:
    if code is None:
        return None
    return str(code).replace(".", "").replace(" ", "").upper() or None


def is_correct(pred, gold, depth: int | None) -> bool:
    p, g = normalize(pred), normalize(gold)
    if p is None or g is None:
        return False
    return p == g if depth is None else p[:depth] == g[:depth]


def ensure_summary(path: str, rebuild: bool) -> str:
    if os.path.exists(path) and not rebuild:
        return path
    logger.info(f"Construction du résumé de la nomenclature dans {path}")
    text = build_summary_text(Graph(neo4j_config).get_summary_tree(5), 5)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


async def verify_one(verifier, graph, clf_output, row: dict) -> None:
    """Juge la prédiction et remplit les champs `verifier_*` et `pred_verified`."""
    sink: list = []
    _trace.set(sink)
    _runs.set(row.setdefault("verifier_runs", []))
    start = time.perf_counter()
    try:
        v = await verifier(clf_output)
    except Exception as e:  # noqa: BLE001
        logger.exception(f"Échec du vérificateur sur {row['text']!r}")
        row.update(verifier_error=f"{type(e).__name__}: {e}", pred_verified=row["pred"])
        return
    finally:
        row["verifier_tools"] = tool_calls(sink)
        row["verifier_duration_s"] = round(time.perf_counter() - start, 1)
    alt = v.alternative_code
    alt_is_leaf = bool(alt) and bool(graph.get_code_information(alt).get("is_final"))
    row.update(
        verifier_is_match=v.is_match,
        verifier_match_score=v.match_score,
        verifier_alternative=alt,
        verifier_alternative_score=v.alternative_score,
        verifier_p_match=v.p_match,
        verifier_explanation=v.explanation,
        pred_verified=row["pred"] if v.is_match or not alt_is_leaf else alt,
    )


async def classify_one(clf, text: str, gold: str, verifier=None, meta=None) -> dict:
    sink: list = []
    _trace.set(sink)
    out = None
    row = {"text": text, "gold": gold, **(meta or {}), "runs": []}
    _runs.set(row["runs"])
    start = time.perf_counter()
    try:
        out = await clf(text)
        row.update(
            pred=out.code,
            confidence=out.proposed_confidence,
            explanation=out.proposed_explanation,
            fallback=out.proposed_explanation.startswith(FALLBACK_MARKER),
        )
    except Exception as e:  # noqa: BLE001 — une erreur LLM ne doit pas arrêter le lot
        logger.exception(f"Échec sur {text!r}")
        row.update(pred=None, fallback=False, error=f"{type(e).__name__}: {e}")
    row["duration_s"] = round(time.perf_counter() - start, 1)
    row["tools"] = tool_calls(sink)
    row["n_tool_calls"] = len(row["tools"])
    row["correct"] = is_correct(row["pred"], gold, None)
    if verifier is not None and out is not None:
        await verify_one(verifier, clf.graph, out, row)
        row["correct_verified"] = is_correct(row.get("pred_verified"), gold, None)
    return row


def accuracy(rows: list[dict], pred_key: str) -> dict:
    n = len(rows)
    return {
        name: round(sum(is_correct(r.get(pred_key), r["gold"], d) for r in rows) / n, 3)
        if n
        else None
        for name, d in DEPTHS.items()
    }


def verifier_metrics(rows: list[dict]) -> dict:
    judged = [r for r in rows if "verifier_is_match" in r]
    if not judged:
        return {}
    accepted = [r for r in judged if r["verifier_is_match"]]
    rejected = [r for r in judged if not r["verifier_is_match"]]
    wrong = [r for r in judged if not r["correct"]]
    caught = [r for r in rejected if not r["correct"]]
    replaced = [r for r in rejected if r["pred_verified"] != r["pred"]]
    return {
        "n_judged": len(judged),
        "verifier_errors": sum(1 for r in rows if r.get("verifier_error")),
        "acceptance_rate": round(len(accepted) / len(judged), 3),
        # Le vérificateur comme filtre : que vaut ce qu'il laisse passer ?
        "accuracy_on_accepted": accuracy(accepted, "pred") if accepted else None,
        "accuracy_on_rejected": accuracy(rejected, "pred") if rejected else None,
        # Détection d'erreurs du classifieur (positif = « mauvais code »).
        "error_recall": round(len(caught) / len(wrong), 3) if wrong else None,
        "error_precision": round(len(caught) / len(rejected), 3) if rejected else None,
        # Effet net du remplacement par l'alternative.
        "n_replaced": len(replaced),
        "replaced_fixed": sum(1 for r in replaced if r["correct_verified"] and not r["correct"]),
        "replaced_broke": sum(1 for r in replaced if r["correct"] and not r["correct_verified"]),
        "replaced_still_wrong": sum(
            1 for r in replaced if not r["correct"] and not r["correct_verified"]
        ),
    }


def _quantiles(values: list[float]) -> dict:
    if not values:
        return {}
    v = sorted(values)
    pick = lambda q: v[min(len(v) - 1, int(q * len(v)))]  # noqa: E731
    return {
        "mean": round(sum(v) / len(v), 1),
        "p50": pick(0.5),
        "p90": pick(0.9),
        "max": v[-1],
    }


def timing_metrics(rows: list[dict]) -> dict:
    """Temps par libellé (s) et coût moyen par phase — exploration (tool_choice libre),
    finalisation (tool_choice="none"), vérificateur."""
    phases: dict[str, list[dict]] = {}
    for r in rows:
        for run in r["runs"]:
            key = "finalize" if run["tool_choice"] == "none" else "exploration"
            phases.setdefault(key, []).append(run)
        for run in r.get("verifier_runs", []):
            phases.setdefault("verifier", []).append(run)

    def mean(runs, field):
        vals = [x[field] for x in runs if x.get(field) is not None]
        return round(sum(vals) / len(vals), 1) if vals else None

    out = {
        "classifier_s": _quantiles([r["duration_s"] for r in rows]),
        "per_phase_mean": {
            name: {
                f: mean(runs, f) for f in ("seconds", "requests", "input_tokens", "output_tokens")
            }
            for name, runs in phases.items()
        },
    }
    if any("verifier_duration_s" in r for r in rows):
        out["verifier_s"] = _quantiles(
            [r["verifier_duration_s"] for r in rows if "verifier_duration_s" in r]
        )
        out["total_s"] = _quantiles(
            [r["duration_s"] + r.get("verifier_duration_s", 0) for r in rows]
        )
    return out


def label_metrics(rows: list[dict], pred_key: str) -> dict:
    """Exactitude à la sous-classe, vue sous l'angle de la fiabilité du label.

    - any_annotation : le code prédit figure parmi les annotations de la liasse (une
      seule pour la plupart, jusqu'à 4 pour la triple annotation) ;
    - unanimous : restreinte aux liasses dont toutes les annotations concordent ;
    - by_source : par source du golden set — les « multivoques » (A) sont par
      construction les cas ambigus, les « univoques » (B) des traductions d'annotations
      NAF 2008 (cf. golden_set_naf2025_doc.txt).
    """
    if not rows or "annotations" not in rows[0]:
        return {}

    def acc(sub, ok):
        return {"accuracy": round(sum(map(ok, sub)) / len(sub), 3), "n": len(sub)} if sub else None

    exact = lambda r: is_correct(r.get(pred_key), r["gold"], None)  # noqa: E731
    any_annot = lambda r: any(is_correct(r.get(pred_key), a, None) for a in r["annotations"])  # noqa: E731
    sources = sorted({r["source"] for r in rows})
    return {
        "any_annotation": acc(rows, any_annot),
        "unanimous": acc([r for r in rows if r["unanimous"]], exact),
        "not_unanimous": acc([r for r in rows if not r["unanimous"]], exact),
        "by_source": {s: acc([r for r in rows if r["source"] == s], exact) for s in sources},
    }


def compute_metrics(rows: list[dict]) -> dict:
    n = len(rows)
    with_tools = [r for r in rows if r["n_tool_calls"] > 0]
    return {
        "n": n,
        "accuracy": accuracy(rows, "pred"),
        # Exactitude hors échecs techniques (timeout, etc.), pour ne pas confondre
        # une panne d'infra avec une erreur de classement.
        "accuracy_excluding_errors": accuracy([r for r in rows if not r.get("error")], "pred"),
        "label_view": label_metrics(rows, "pred"),
        "errors": sum(1 for r in rows if r.get("error")),
        "non_final_fallbacks": sum(1 for r in rows if r.get("fallback")),
        "zero_tool_calls": n - len(with_tools),
        "mean_tool_calls": round(sum(r["n_tool_calls"] for r in rows) / n, 2),
        "max_tool_calls": max((r["n_tool_calls"] for r in rows), default=0),
        "tool_calls_on_unknown_code": sum(
            1 for r in rows for c in r["tools"] if c.get("unknown_code")
        ),
        "tool_usage": dict(Counter(c["name"] for r in rows for c in r["tools"])),
        "timing": timing_metrics(rows),
    }


def ablation_metrics(rows: list[dict]) -> dict:
    """Mêmes libellés, avec vs sans vérificateur (les échecs du vérificateur gardent le
    code du classifieur, cf. verify_one)."""
    return {
        "accuracy": {
            "classifier": accuracy(rows, "pred"),
            "classifier+verifier": accuracy(rows, "pred_verified"),
        },
        "label_view": {
            "classifier": label_metrics(rows, "pred"),
            "classifier+verifier": label_metrics(rows, "pred_verified"),
        },
        "verifier": verifier_metrics(rows),
    }


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--golden", default=GOLDEN_SET)
    parser.add_argument("--n", type=int, default=10, help="taille de l'échantillon (0 = tout)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--all", action="store_true", help="ne pas restreindre à in_representative_subset"
    )
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--summary", default=DEFAULT_SUMMARY_PATH)
    parser.add_argument(
        "--rebuild-summary", action="store_true", help="regénérer le résumé depuis Neo4j"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="ablation : juger chaque prédiction avec MatchVerifier",
    )
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    out_dir = args.out or os.path.join(
        "eval", "results", f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    os.makedirs(out_dir, exist_ok=True)

    with storage.open_path(args.golden, "rb") as f:
        df = pl.read_parquet(f)
    if not args.all:
        df = df.filter(pl.col("in_representative_subset"))
    if args.n and args.n < df.height:
        df = df.sample(n=args.n, seed=args.seed)
    texts, golds = df["libelle"].to_list(), df["apet2025"].to_list()
    metas = [
        {
            # « XXXXX » = jugé inclassable par un annotateur : pas un code à comparer.
            "annotations": [a for a in m["apet2025_annotations"] if a != "XXXXX"],
            "unanimous": m["unanimous"],
            "source": m["source"],
        }
        for m in df.select("apet2025_annotations", "unanimous", "source").to_dicts()
    ]
    logger.info(f"{len(texts)} libellés à classer, sorties dans {out_dir}")

    graph = Graph(neo4j_config)
    clf = SummaryAgenticClassifier(
        graph, summary_path=ensure_summary(args.summary, args.rebuild_summary)
    )
    verifier = MatchVerifier(graph) if args.verify else None

    semaphore = asyncio.Semaphore(args.concurrency)
    rows: list[dict] = []
    with open(os.path.join(out_dir, "predictions.jsonl"), "w", encoding="utf-8") as ckpt:

        async def worker(text, gold, meta):
            async with semaphore:
                row = await classify_one(clf, text, gold, verifier, meta)
            rows.append(row)
            ckpt.write(json.dumps(row, ensure_ascii=False) + "\n")
            ckpt.flush()
            calls = ", ".join(f"{c['name']}({c['arguments']})" for c in row["tools"]) or "-"
            verdict = ""
            if "verifier_is_match" in row:
                verdict = (
                    f" | verifier: {'accepte' if row['verifier_is_match'] else 'rejette'}"
                    f" -> {row['pred_verified']} {'OK' if row['correct_verified'] else 'KO'}"
                )
            secs = f"{row['duration_s']}s"
            if "verifier_duration_s" in row:
                secs += f"+{row['verifier_duration_s']}s"
            logger.info(
                f"[{len(rows)}/{len(texts)}] ({secs}) {'OK' if row['correct'] else 'KO'} "
                f"gold={gold} pred={row['pred']}{verdict} outils: {calls}"
            )

        await asyncio.gather(*(worker(t, g, m) for t, g, m in zip(texts, golds, metas)))

    metrics = compute_metrics(rows)
    if args.verify:
        metrics["ablation"] = ablation_metrics(rows)
    metrics.update(
        golden=args.golden,
        representative_subset=not args.all,
        seed=args.seed,
        model=os.environ.get("GENERATION_MODEL"),
    )
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
