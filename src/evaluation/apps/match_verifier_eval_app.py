"""Standalone web app for human review of match_verifier_eval.py's output.

For each row of the MatchVerifier eval parquet (cf. src.evaluation.match_verifier_eval
— columns libelle, current_code, match_verifier_verdict, match_verifier_explanation,
match_verifier_confidence), shows the activity text, the code currently attached to it
with its official notice, and MatchVerifier's own verdict on that pair. The reviewer
answers one question — is MatchVerifier's verdict on that pair correct — plus a free
field for the code they would have assigned. An earlier version also asked, separately,
whether the code itself was correct (which let /metrics recompute precision/recall
without trusting the verdict); that question was retired as not worth its annotation
cost, but `human_code_correct` stays in the log and schema so the judgments already
collected under it remain readable.

Companion of multi_method_review_app.py, narrowed to one candidate per row: that app
reviews 5 candidates (ground truth + 4 classifiers), CodeChooser's arbitration and
every source's verdict at once, on the eval set; this one reviews MatchVerifier alone
against training labels, which is what match_verifier_eval.py produces.

Reviewers each work through a slice: a shared pool every reviewer sees (for inter-rater
agreement) plus a slice unique to them (cf. --reviewers/--shared-n/--unique-n/--seed).
The split is deterministic from the input file and those arguments, so reviewing is
fully asynchronous — anyone can pick up their own progress at any time, as long as the
input file and those four arguments don't change mid-review.

Judgments are logged append-only to JSONL (keyed by (reviewer, row_id), so re-running
is idempotent and safe to interrupt) and, after each submission, materialized as a
parquet next to it — one row per (reviewer, judged row), carrying the reviewer's name
in the `reviewer` column alongside the original parquet's five columns and the human
answers. The /metrics page derives, from reviewed rows only: how often the human called
the verdict right, inter-rater agreement on the shared pool, and — over the judgments
made back when the code question was still asked — MatchVerifier's accuracy, precision
and recall against it.

Une revue porte sur un run, et un run porte sur un prompt et un modèle :
`match_verifier_eval.py` écrit ses résultats sous <output>/<commit>/<modèle>/, et
cette app lit donc <input>/<commit>/<modèle>/match_verifier_eval.parquet et
journalise les jugements sous <output-dir>/<commit>/<modèle>/. `--commit` et
`--model` valent par défaut le tag git de HEAD et GENERATION_MODEL ; si aucun run
n'existe pour ce couple, l'app refuse de démarrer en listant les runs disponibles,
plutôt que de mélanger dans un même JSONL des jugements portant sur deux révisions
du prompt ou deux modèles (ce qui fausserait silencieusement /metrics).

Nécessite Neo4j configuré dans l'environnement pour afficher la notice du code ;
sans connexion, la revue reste possible mais sans notice.

Usage :
    uv run -m src.evaluation.apps.match_verifier_eval_app \
        --input s3://projet-ape/graal/data/eval/match_verifier_eval \
        --commit ec8bf27 --model qwen3-6-35b-moe \
        --reviewers meilame,theo,nathan \
        --port 5052
"""

import argparse
import json
import logging
import os
import random
import threading
from datetime import datetime, timezone

import polars as pl
from flask import Flask, redirect, render_template_string, request, url_for

from src.config import neo4j_config
from src.evaluation.row_id import row_id_for
from src.neo4j_graph.graph import Graph
from src.utils import storage
from src.utils.logging import configure_logging
from src.utils.run_provenance import model_slug, revision_tag

configure_logging()
logger = logging.getLogger(__name__)

DEFAULT_INPUT = "s3://projet-ape/graal/data/eval/match_verifier_eval"
DEFAULT_OUTPUT_DIR = "data/eval/human_review"
REVIEW_FILENAME = "match_verifier_review.jsonl"

# Input file and columns, as written by match_verifier_eval.py (duplicated rather
# than imported: that module pulls in the whole agent stack at import time).
DETAILS_FILENAME = "match_verifier_eval.parquet"
TEXT_COLUMN = "libelle"
CODE_COLUMN = "current_code"
VERDICT_COLUMN = "match_verifier_verdict"
EXPLANATION_COLUMN = "match_verifier_explanation"
CONFIDENCE_COLUMN = "match_verifier_confidence"


def available_runs(input_path: str) -> list[str]:
    """`<commit>/<model>` of every run found under the `input_path` root.

    Two levels deep, so the listing shown when a run is missing names something
    that can be pasted straight back as --commit/--model. Only directories holding
    the eval parquet count — a half-written run isn't one to offer.
    """
    runs = []
    for commit in storage.list_dir(input_path):
        for model in storage.list_dir(os.path.join(input_path, commit)):
            if storage.path_exists(os.path.join(input_path, commit, model, DETAILS_FILENAME)):
                runs.append(f"{commit}/{model}")
    return runs


def resolve_input(input_path: str, commit: str, model: str) -> str:
    """Path of the eval parquet for `commit`/`model` under the `input_path` root.

    A path already ending in .parquet is taken as-is — the escape hatch for a run
    that predates this layout, or one copied somewhere by hand. Otherwise the
    run's subdirectory is required to exist: silently falling back to "the latest
    run" would reshuffle the reviewer split (which is derived from the input rows)
    under in-progress review work, and pool judgments on two different prompts, or
    two different models, into one set of metrics.
    """
    if input_path.endswith(".parquet"):
        return input_path
    path = os.path.join(input_path, commit, model, DETAILS_FILENAME)
    if not storage.path_exists(path):
        runs = available_runs(input_path)
        raise SystemExit(
            f"No eval run for {commit}/{model}: {path} does not exist.\n"
            f"Runs available under {input_path}: {', '.join(runs) or '(none)'}.\n"
            "Pass --commit/--model from that list to review an earlier run, or produce "
            "one for the current commit and model with "
            "`uv run -m src.evaluation.match_verifier_eval`."
        )
    return path


def resolve_output(output_dir: str, commit: str, model: str) -> str:
    """JSONL log path for `commit`/`model` under the `output_dir` root.

    Namespaced the same way as the input: judgments of a verdict produced by one
    prompt revision and model say nothing about the next, and /metrics pools every
    judgment in the file it reads.
    """
    return os.path.join(output_dir, commit, model, REVIEW_FILENAME)


def load_rows(input_path: str) -> list[dict]:
    with storage.open_path(input_path, "rb") as f:
        df = pl.read_parquet(f)
    missing = {TEXT_COLUMN, CODE_COLUMN, VERDICT_COLUMN} - set(df.columns)
    if missing:
        raise ValueError(
            f"{input_path} is missing column(s) {sorted(missing)} — expected the output of "
            "src.evaluation.match_verifier_eval"
        )
    rows = []
    for r in df.to_dicts():
        code = str(r[CODE_COLUMN])
        rows.append(
            {
                "row_id": row_id_for(r[TEXT_COLUMN], code),
                "libelle": r[TEXT_COLUMN],
                "current_code": code,
                "verdict": r[VERDICT_COLUMN],
                "explanation": r.get(EXPLANATION_COLUMN),
                "confidence": r.get(CONFIDENCE_COLUMN),
            }
        )
    return rows


def build_assignment(
    row_ids: list[str], reviewers: list[str], shared_n: int, unique_n: int, seed: int
) -> tuple[dict[str, list[str]], set[str]]:
    """Deterministic reviewer split: a shared pool every reviewer sees, plus a slice
    unique to each — same input + same (reviewers, shared_n, unique_n, seed) always
    reproduces the same split, which is what makes asynchronous review safe (cf.
    module docstring).
    """
    needed = shared_n + unique_n * len(reviewers)
    if len(row_ids) < needed:
        raise ValueError(
            f"Input has {len(row_ids)} rows, but {needed} are needed for "
            f"{len(reviewers)} reviewers (--shared-n {shared_n} + --unique-n {unique_n} each)."
        )
    order = list(row_ids)
    random.Random(seed).shuffle(order)
    shared_pool = order[:shared_n]
    rest = order[shared_n:]

    assignment = {}
    for i, reviewer in enumerate(reviewers):
        unique_slice = rest[i * unique_n : (i + 1) * unique_n]
        working_set = shared_pool + unique_slice
        random.Random(f"{seed}:{reviewer}").shuffle(working_set)
        assignment[reviewer] = working_set
    return assignment, set(shared_pool)


def load_judgments(output_path: str) -> dict[str, dict[str, dict]]:
    """{reviewer: {row_id: judgment_entry}} — last line wins per (reviewer, row)."""
    judgments: dict[str, dict[str, dict]] = {}
    if not storage.path_exists(output_path):
        return judgments
    with storage.open_path(output_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            judgments.setdefault(entry["reviewer"], {})[entry["row_id"]] = entry
    return judgments


# Les annotateurs travaillent en parallèle sur une même instance (le déploiement
# vit dans codif-ape-cd), et Flask sert les requêtes sur des threads. Or ajouter une ligne à
# un JSONL sur S3 est un read-modify-write — s3fs recharge l'objet puis le réécrit
# entier en dessous de 5 Mo — donc deux soumissions simultanées perdraient l'une des
# deux. Ce verrou sérialise l'append et la réécriture du parquet qui le suit ; il ne
# vaut que dans un processus, d'où le `replicas: 1` du Deployment.
_write_lock = threading.Lock()


def append_judgment(output_path: str, entry: dict) -> None:
    storage.makedirs(os.path.dirname(output_path) or ".")
    with storage.open_path(output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def parquet_path_for(output_path: str) -> str:
    """The parquet materialized next to the JSONL log, same basename."""
    base = output_path[: -len(".jsonl")] if output_path.endswith(".jsonl") else output_path
    return f"{base}.parquet"


REVIEW_SCHEMA = [
    ("row_id", pl.Utf8),
    ("reviewer", pl.Utf8),
    (TEXT_COLUMN, pl.Utf8),
    (CODE_COLUMN, pl.Utf8),
    (VERDICT_COLUMN, pl.Boolean),
    (EXPLANATION_COLUMN, pl.Utf8),
    (CONFIDENCE_COLUMN, pl.Float64),
    ("human_code_correct", pl.Boolean),
    ("human_verdict_correct", pl.Boolean),
    ("human_suggested_code", pl.Utf8),
    ("reviewed_at", pl.Utf8),
]


def write_review_parquet(output_path: str, all_judgments: dict[str, dict[str, dict]]) -> str:
    """Rewrite the whole review parquet from the JSONL log (one row per (reviewer,
    judged row), reviewer name in the `reviewer` column).

    Rewritten in full on every submission rather than appended to: the JSONL is the
    append-only source of truth (last line wins per (reviewer, row), so a re-judged
    row must replace its earlier one, not duplicate it), and at review-campaign
    scale — a few hundred rows times a handful of reviewers — a full rewrite is
    cheaper than reconciling.
    """
    records = [
        {name: entry.get(name) for name, _ in REVIEW_SCHEMA}
        for by_row in all_judgments.values()
        for entry in by_row.values()
    ]
    df = pl.DataFrame(records, schema=REVIEW_SCHEMA)
    path = parquet_path_for(output_path)
    storage.makedirs(os.path.dirname(path) or ".")
    with storage.open_path(path, "wb") as f:
        df.write_parquet(f)
    return path


def compute_metrics(
    rows: list[dict], all_judgments: dict[str, dict[str, dict]], shared_ids: set[str]
) -> dict:
    """Metrics derived from every (reviewer, row) judgment pooled together.

    `verdict_agreement` is how often the human explicitly said "that verdict is right",
    and is the live metric: it is the only question the form still asks. The confusion
    matrix and label_accuracy instead read `human_code_correct`, the retired question,
    so they only cover judgments recorded while it was still asked — every rate here is
    therefore computed over its own denominator (`verdict_n`, `label_n`, `verifier.n`)
    rather than over the judgment count, which would otherwise drift as new judgments
    answer only one of the two.
    """
    known_ids = {r["row_id"] for r in rows}

    n_judgments = 0
    code_total = label_correct = 0
    verdict_total = verdict_agree = 0
    tp = fp = tn = fn = 0

    for by_row in all_judgments.values():
        for row_id, judgment in by_row.items():
            if row_id not in known_ids:
                continue
            n_judgments += 1
            code_correct = judgment.get("human_code_correct")
            verdict = judgment.get(VERDICT_COLUMN)
            if code_correct is not None:
                code_total += 1
                if code_correct:
                    label_correct += 1

            verdict_correct = judgment.get("human_verdict_correct")
            if verdict_correct is not None:
                verdict_total += 1
                if verdict_correct:
                    verdict_agree += 1

            if verdict is not None and code_correct is not None:
                if verdict and code_correct:
                    tp += 1
                elif verdict and not code_correct:
                    fp += 1
                elif not verdict and code_correct:
                    fn += 1
                else:
                    tn += 1

    def rate(numerator: int, denominator: int) -> float | None:
        return numerator / denominator if denominator else None

    n_confusion = tp + fp + tn + fn
    return {
        "n_total": len(rows),
        "n_judgments": n_judgments,
        "label_accuracy": rate(label_correct, code_total),
        "label_n": code_total,
        "verdict_agreement": rate(verdict_agree, verdict_total),
        "verdict_n": verdict_total,
        "verifier": {
            "n": n_confusion,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "accuracy": rate(tp + tn, n_confusion),
            "precision": rate(tp, tp + fp),
            "recall": rate(tp, tp + fn),
        },
        "agreement": compute_inter_rater_agreement(all_judgments, shared_ids),
    }


def compute_inter_rater_agreement(
    all_judgments: dict[str, dict[str, dict]], shared_ids: set[str]
) -> dict:
    """Percent agreement between every pair of reviewers on the shared pool, on each
    question ("verdict correct?", and the retired "code correct?") separately.

    A question counts for a pair only when both reviewers actually answered it —
    without that guard two unanswered questions would compare equal and report
    perfect agreement on something nobody was asked.
    """
    reviewers = sorted(all_judgments)
    pairs = []
    code_match = code_total = 0
    verdict_match = verdict_total = 0

    for i in range(len(reviewers)):
        for j in range(i + 1, len(reviewers)):
            a, b = reviewers[i], reviewers[j]
            pair_code_match = pair_code_total = 0
            pair_verdict_match = pair_verdict_total = 0
            for row_id in shared_ids:
                ja = all_judgments[a].get(row_id)
                jb = all_judgments[b].get(row_id)
                if ja is None or jb is None:
                    continue
                ca, cb = ja.get("human_code_correct"), jb.get("human_code_correct")
                if ca is not None and cb is not None:
                    pair_code_total += 1
                    code_total += 1
                    if ca == cb:
                        pair_code_match += 1
                        code_match += 1
                va, vb = ja.get("human_verdict_correct"), jb.get("human_verdict_correct")
                if va is not None and vb is not None:
                    pair_verdict_total += 1
                    verdict_total += 1
                    if va == vb:
                        pair_verdict_match += 1
                        verdict_match += 1
            if pair_code_total or pair_verdict_total:
                pairs.append(
                    {
                        "reviewers": (a, b),
                        "n_code": pair_code_total,
                        "code_agreement": (
                            pair_code_match / pair_code_total if pair_code_total else None
                        ),
                        "n_verdict": pair_verdict_total,
                        "verdict_agreement": (
                            pair_verdict_match / pair_verdict_total if pair_verdict_total else None
                        ),
                    }
                )

    return {
        "pairs": pairs,
        "overall_code_agreement": code_match / code_total if code_total else None,
        "overall_verdict_agreement": verdict_match / verdict_total if verdict_total else None,
        "n_code": code_total,
        "n_verdict": verdict_total,
    }


NAV_VERDICT_GROUPS = [
    (False, "MatchVerifier : pas de correspondance"),
    (True, "MatchVerifier : correspondance"),
    (None, "MatchVerifier : sans verdict"),
]


def nav_entry_label(position: int, libelle: str, judgment: dict | None) -> str:
    """One line of the jump-to dropdown: rank in the reviewer's order, review status,
    and the beginning of the activity text (truncated — an <option> can't wrap)."""
    if judgment is None:
        status = "à juger    "
    else:
        status = {True: "verdict OK ", False: "verdict KO ", None: "jugé       "}[
            judgment.get("human_verdict_correct")
        ]
    text = libelle if len(libelle) <= 70 else libelle[:69] + "\u2026"
    return f"{position:>3}. [{status}] {text}"


def build_nav_groups(
    order: list[str], rows_by_id: dict[str, dict], judgments: dict[str, dict]
) -> list[tuple[str, list[dict]]]:
    """The reviewer's assigned rows, grouped by MatchVerifier's verdict.

    Grouped by verdict rather than kept in review order because the point of the
    dropdown is to reach a *class* of rows directly — typically the ones the verifier
    rejected — while the position prefix keeps the review order readable inside each
    group. Each group's header carries how many of its rows are still unjudged.
    """
    by_verdict: dict[bool | None, list[dict]] = {}
    for i, row_id in enumerate(order):
        row = rows_by_id[row_id]
        verdict = None if row["verdict"] is None else bool(row["verdict"])
        judgment = judgments.get(row_id)
        by_verdict.setdefault(verdict, []).append(
            {
                "row_id": row_id,
                "label": nav_entry_label(i + 1, row["libelle"], judgment),
                "judged": judgment is not None,
            }
        )

    groups = []
    for verdict, title in NAV_VERDICT_GROUPS:
        entries = by_verdict.get(verdict)
        if not entries:
            continue
        todo = sum(1 for e in entries if not e["judged"])
        plural = "s" if len(entries) > 1 else ""
        groups.append((f"{title} — {len(entries)} ligne{plural}, {todo} à juger", entries))
    return groups


_code_notice_cache: dict[str, dict | None] = {}
_graph: Graph | None | bool = None


def get_graph() -> Graph | None:
    global _graph
    if _graph is None:
        try:
            _graph = Graph(neo4j_config)
        except Exception:
            logger.exception("Could not connect to Neo4j, code notices will be unavailable")
            _graph = False
    return _graph or None


def get_code_notice(code: str) -> dict | None:
    """Nom du code et notice officielle complète, ou None si Neo4j est indisponible.

    `text` vient de `Graph.get_notice` et non du seul champ `description` de
    `get_code_information` : ce dernier n'est que le nom suivi de la section
    « comprend », sans les exclusions — or « ne comprend pas » est précisément ce
    qui tranche les cas limites qu'on demande au relecteur d'arbitrer, et sans quoi
    il juge sur moins d'information que le MatchVerifier dont il relit le verdict.
    `get_notice` recompose toutes les sections en dédupliquant, soit exactement la
    notice que les agents mettent dans leur prompt.
    """
    if code in _code_notice_cache:
        return _code_notice_cache[code]
    graph = get_graph()
    notice = None
    if graph is not None:
        try:
            info = graph.get_code_information(code)
            if info and not info.get("error"):
                notice = {"name": info.get("name"), "text": graph.get_notice(code)}
        except Exception:
            logger.exception(f"Failed to fetch code information for {code}")
    _code_notice_cache[code] = notice
    return notice


STYLE = """
<style>
  :root { color-scheme: light dark; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    max-width: 860px; margin: 2rem auto; padding: 0 1.25rem;
    line-height: 1.5; color: #1a1a1a; background: #fff;
  }
  @media (prefers-color-scheme: dark) {
    body { color: #e8e8e8; background: #1b1b1b; }
    .card, .candidate { background: #262626 !important; border-color: #3a3a3a !important; }
    .muted { color: #999 !important; }
    a { color: #7db8ff; }
  }
  nav { display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.5rem; }
  nav a { text-decoration: none; font-weight: 600; }
  .progress { font-size: 0.9rem; color: #666; }
  .card {
    border: 1px solid #ddd; border-radius: 10px; padding: 1.25rem 1.5rem;
    margin-bottom: 1rem; background: #fafafa;
  }
  .label { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.04em;
    color: #888; margin-bottom: 0.3rem; }
  .activite { font-size: 1.3rem; font-weight: 600; margin: 0.2rem 0 1rem; }
  .code-badge { display: inline-block; font-family: monospace; font-size: 1.05rem;
    background: #eef2ff; padding: 0.15rem 0.5rem; border-radius: 6px; font-weight: 700; }
  @media (prefers-color-scheme: dark) { .code-badge { background: #2a3350; color: #cfd9ff; } }
  .notice { white-space: pre-line; font-size: 0.9rem; margin-top: 0.5rem; }
  .muted { color: #777; font-size: 0.9rem; margin-top: 0.4rem; }
  .verdict-badge { display: inline-block; padding: 0.1rem 0.6rem; border-radius: 999px;
    font-size: 0.78rem; font-weight: 700; }
  .verdict-yes { background: #d7f5df; color: #166534; }
  .verdict-no { background: #fde2e2; color: #991b1b; }
  @media (prefers-color-scheme: dark) {
    .verdict-yes { background: #14351f; color: #7be0a0; }
    .verdict-no { background: #3a1717; color: #ffb4b4; }
  }
  .candidate {
    border: 1px solid #ddd; border-radius: 10px; padding: 1rem 1.25rem;
    margin-bottom: 0.75rem; background: #fafafa;
  }
  .judgments { display: flex; flex-wrap: wrap; gap: 1.5rem; margin-top: 0.9rem;
    padding-top: 0.75rem; border-top: 1px dashed #ddd; }
  @media (prefers-color-scheme: dark) { .judgments { border-color: #3a3a3a; } }
  .jg-label { font-size: 0.85rem; font-weight: 600; margin-right: 0.6rem; }
  .judgment-group label { margin-right: 0.7rem; font-size: 0.9rem; cursor: pointer; }
  .other-row { display: flex; gap: 0.5rem; margin: 1rem 0; }
  .other-row input[type=text] {
    flex: 1; padding: 0.5rem 0.7rem; border-radius: 8px; border: 1px solid #ccc; font-family: monospace;
  }
  button.submit-all {
    font-size: 1rem; font-weight: 700; padding: 0.7rem 1.4rem; border-radius: 8px;
    border: none; cursor: pointer; background: #16a34a; color: white; margin-top: 0.5rem;
  }
  button.submit-all:hover { opacity: 0.88; }
  .existing-banner { padding: 0.5rem 0.9rem; border-radius: 8px; margin-bottom: 1rem;
    font-weight: 600; background: #d7f5df; color: #166534; }
  @media (prefers-color-scheme: dark) { .existing-banner { background: #14351f; color: #7be0a0; } }
  .nav-links { display: flex; justify-content: space-between; margin-top: 1rem; }
  .nav-links a.disabled { pointer-events: none; color: #bbb; }
  .picker a { display: block; padding: 0.8rem 1rem; margin-bottom: 0.6rem; border-radius: 8px;
    border: 1px solid #ddd; text-decoration: none; font-weight: 600; }
  table { border-collapse: collapse; width: 100%; margin: 1rem 0; }
  th, td { text-align: left; padding: 0.4rem 0.6rem; border-bottom: 1px solid #e5e5e5; font-size: 0.9rem; }
  @media (prefers-color-scheme: dark) { th, td { border-color: #3a3a3a; } }
  .stat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 0.75rem; margin: 1.25rem 0; }
  .stat { border: 1px solid #ddd; border-radius: 10px; padding: 0.8rem; text-align: center; background: #fafafa; }
  .stat .n { font-size: 1.4rem; font-weight: 700; display: block; }
  .stat .l { font-size: 0.72rem; color: #777; text-transform: uppercase; letter-spacing: 0.03em; }
  .row-picker { margin: -0.7rem 0 1.25rem; }
  .row-picker select {
    width: 100%; padding: 0.5rem 0.6rem; border-radius: 8px; border: 1px solid #ddd;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.82rem;
    background: #fafafa; color: inherit;
  }
  @media (prefers-color-scheme: dark) {
    .row-picker select { background: #262626; border-color: #3a3a3a; color: #e8e8e8; }
  }
</style>
"""

NAV = """
<nav>
  <a href="{{ url_for('index', reviewer=reviewer) }}">&larr; Revue ({{ reviewer }})</a>
  <span class="progress">{{ n_reviewed }} / {{ n_total }} jugés &middot; run {{ run }}</span>
  <a href="{{ url_for('metrics') }}">Métriques &rarr;</a>
</nav>
"""

PICKER_TEMPLATE = (
    STYLE
    + """
<h2>Qui êtes-vous ?</h2>
<p class="progress">Revue du run <code>{{ run }}</code></p>
<div class="picker">
  {% for r in reviewers %}
  <a href="{{ url_for('index', reviewer=r) }}">{{ r }}</a>
  {% endfor %}
</div>
"""
)

REVIEW_TEMPLATE = (
    STYLE
    + NAV
    + """
<div class="row-picker">
  <select onchange="if (this.value) location.href = this.value;">
    {% for title, entries in nav_groups %}
    <optgroup label="{{ title }}">
      {% for e in entries %}
      <option value="{{ url_for('review', row_id=e.row_id, reviewer=reviewer) }}"
        {{ "selected" if e.row_id == row.row_id }}>{{ e.label }}</option>
      {% endfor %}
    </optgroup>
    {% endfor %}
  </select>
</div>

{% if existing %}
  <div class="existing-banner">Déjà jugé par {{ reviewer }} le {{ existing.reviewed_at }}</div>
{% endif %}

<div class="card">
  <div class="label">Activité ({{ idx }} / {{ total }})</div>
  <div class="activite">{{ row.libelle }}</div>
</div>

<form method="post" action="{{ url_for('judge', row_id=row.row_id, reviewer=reviewer) }}">

<div class="candidate">
  <div>
    <span class="code-badge">{{ row.current_code }}</span>
    {% if notice %} &mdash; {{ notice.name }}{% endif %}
    {% if notice and notice.text %}<div class="notice">{{ notice.text }}</div>{% endif %}
  </div>
  <div class="muted">
    MatchVerifier :
    <span class="verdict-badge {{ 'verdict-yes' if row.verdict else 'verdict-no' }}">
      {{ "correspondance" if row.verdict else "pas de correspondance" }}
    </span>
    {% if row.confidence is not none %}
      (confiance {{ "%.0f"|format(row.confidence * 100) }}%)
    {% endif %}
    <div class="notice">{{ row.explanation }}</div>
  </div>
  <div class="judgments">
    <div class="judgment-group">
      <span class="jg-label">Le verdict MatchVerifier est-il correct ?</span>
      <label><input type="radio" name="verdict_correct" value="yes" required
        {{ "checked" if existing and existing.human_verdict_correct }}> Oui</label>
      <label><input type="radio" name="verdict_correct" value="no"
        {{ "checked" if existing and existing.human_verdict_correct == false }}> Non</label>
    </div>
  </div>
</div>

<div class="label" style="margin-bottom: 0.3rem;">Si le code est faux, quel serait le bon ?</div>
<div class="other-row">
  <input type="text" name="suggested_code" placeholder="Code correct (optionnel)"
    value="{{ existing.human_suggested_code if existing and existing.human_suggested_code else '' }}">
</div>

<button type="submit" class="submit-all">Valider cette activité</button>
</form>

<div class="nav-links">
  <a href="{{ url_for('review', row_id=prev_id, reviewer=reviewer) if prev_id else '#' }}"
     class="{{ '' if prev_id else 'disabled' }}">&larr; Précédent</a>
  <a href="{{ url_for('review', row_id=next_id, reviewer=reviewer) if next_id else '#' }}"
     class="{{ '' if next_id else 'disabled' }}">Passer &rarr;</a>
</div>
"""
)

METRICS_TEMPLATE = (
    STYLE
    + """
<nav>
  <a href="{{ url_for('index') }}">&larr; Revue</a>
</nav>

<h2>Métriques MatchVerifier <span class="progress">(run {{ run }})</span></h2>

<div class="stat-grid">
  <div class="stat"><span class="n">{{ m.n_judgments }}/{{ m.n_assigned_total }}</span><span class="l">Jugements</span></div>
  <div class="stat"><span class="n">{{ "%.0f%%"|format(m.label_accuracy * 100) if m.label_accuracy is not none else "—" }}</span><span class="l">Labels corrects ({{ m.label_n }})</span></div>
  <div class="stat"><span class="n">{{ "%.0f%%"|format(m.verdict_agreement * 100) if m.verdict_agreement is not none else "—" }}</span><span class="l">Verdicts jugés corrects ({{ m.verdict_n }})</span></div>
</div>

<h3>Verdict (is_match) vs jugement humain du code</h3>
<div class="stat-grid">
  <div class="stat"><span class="n">{{ "%.0f%%"|format(m.verifier.accuracy * 100) if m.verifier.accuracy is not none else "—" }}</span><span class="l">Exactitude ({{ m.verifier.n }})</span></div>
  <div class="stat"><span class="n">{{ "%.0f%%"|format(m.verifier.precision * 100) if m.verifier.precision is not none else "—" }}</span><span class="l">Précision</span></div>
  <div class="stat"><span class="n">{{ "%.0f%%"|format(m.verifier.recall * 100) if m.verifier.recall is not none else "—" }}</span><span class="l">Rappel</span></div>
</div>
<table>
  <tr><th></th><th>Code jugé correct</th><th>Code jugé incorrect</th></tr>
  <tr><td>Verdict « correspondance »</td><td>{{ m.verifier.tp }}</td><td>{{ m.verifier.fp }}</td></tr>
  <tr><td>Verdict « pas de correspondance »</td><td>{{ m.verifier.fn }}</td><td>{{ m.verifier.tn }}</td></tr>
</table>
<p class="muted">
  « Verdicts jugés corrects » : part des verdicts que l'annotateur a explicitement
  validés — c'est la seule question posée aujourd'hui. Exactitude/précision/rappel
  reposent sur l'ancienne question « le code est-il correct ? », retirée du formulaire :
  elles ne portent donc que sur les {{ m.label_n }} jugements qui y avaient répondu et
  ne bougeront plus.
</p>

<h3>Accord inter-annotateurs (pool partagé)</h3>
<div class="stat-grid">
  <div class="stat"><span class="n">{{ "%.0f%%"|format(m.agreement.overall_code_agreement * 100) if m.agreement.overall_code_agreement is not none else "—" }}</span><span class="l">Accord « code correct » ({{ m.agreement.n_code }})</span></div>
  <div class="stat"><span class="n">{{ "%.0f%%"|format(m.agreement.overall_verdict_agreement * 100) if m.agreement.overall_verdict_agreement is not none else "—" }}</span><span class="l">Accord « verdict correct » ({{ m.agreement.n_verdict }})</span></div>
</div>
{% if m.agreement.pairs %}
<table>
  <tr><th>Paire</th><th>N verdict</th><th>Accord « verdict correct »</th><th>N code</th><th>Accord « code correct »</th></tr>
  {% for p in m.agreement.pairs %}
  <tr>
    <td>{{ p.reviewers[0] }} / {{ p.reviewers[1] }}</td>
    <td>{{ p.n_verdict }}</td>
    <td>{{ "%.0f%%"|format(p.verdict_agreement * 100) if p.verdict_agreement is not none else "—" }}</td>
    <td>{{ p.n_code }}</td>
    <td>{{ "%.0f%%"|format(p.code_agreement * 100) if p.code_agreement is not none else "—" }}</td>
  </tr>
  {% endfor %}
</table>
{% endif %}
"""
)


class PrefixMiddleware:
    """Same Onyxia reverse-proxy handling as multi_method_review_app (cf. VSCODE_PROXY_URI)."""

    def __init__(self, app, prefix: str):
        self.app = app
        self.prefix = prefix

    def __call__(self, environ, start_response):
        if self.prefix:
            path = environ.get("PATH_INFO", "")
            if path.startswith(self.prefix):
                environ["PATH_INFO"] = path[len(self.prefix) :] or "/"
            environ["SCRIPT_NAME"] = self.prefix
        return self.app(environ, start_response)


def create_app(
    input_path: str,
    output_path: str,
    reviewers: list[str],
    run: str,
    shared_n: int,
    unique_n: int,
    seed: int,
    url_prefix: str = "",
) -> Flask:
    app = Flask(__name__)
    if url_prefix:
        app.wsgi_app = PrefixMiddleware(app.wsgi_app, url_prefix)

    rows = load_rows(input_path)
    rows_by_id = {r["row_id"]: r for r in rows}
    row_ids = [r["row_id"] for r in rows]
    assignment, shared_ids = build_assignment(row_ids, reviewers, shared_n, unique_n, seed)
    logger.info(
        f"Loaded {len(rows)} rows for run {run}; {shared_n} shared + {unique_n} unique "
        f"per reviewer ({', '.join(reviewers)})"
    )

    def order_for(reviewer: str) -> list[str]:
        return assignment[reviewer]

    def first_unreviewed(reviewer: str) -> str | None:
        done = load_judgments(output_path).get(reviewer, {})
        for row_id in order_for(reviewer):
            if row_id not in done:
                return row_id
        return None

    @app.route("/")
    def index():
        reviewer = request.args.get("reviewer")
        if reviewer not in reviewers:
            return render_template_string(PICKER_TEMPLATE, reviewers=reviewers, run=run)
        next_id = first_unreviewed(reviewer)
        if next_id is None:
            return redirect(url_for("metrics"))
        return redirect(url_for("review", row_id=next_id, reviewer=reviewer))

    @app.route("/review/<row_id>")
    def review(row_id):
        reviewer = request.args.get("reviewer")
        order = order_for(reviewer) if reviewer in reviewers else []
        if reviewer not in reviewers or row_id not in order:
            return redirect(url_for("index", reviewer=reviewer))
        row = rows_by_id[row_id]
        judgments = load_judgments(output_path).get(reviewer, {})
        idx = order.index(row_id)
        return render_template_string(
            REVIEW_TEMPLATE,
            row=row,
            reviewer=reviewer,
            run=run,
            notice=get_code_notice(row["current_code"]),
            idx=idx + 1,
            total=len(order),
            n_reviewed=len(judgments),
            n_total=len(order),
            prev_id=order[idx - 1] if idx > 0 else None,
            next_id=order[idx + 1] if idx + 1 < len(order) else None,
            existing=judgments.get(row_id),
            nav_groups=build_nav_groups(order, rows_by_id, judgments),
        )

    @app.route("/judge/<row_id>", methods=["POST"])
    def judge(row_id):
        reviewer = request.args.get("reviewer")
        if reviewer not in reviewers or row_id not in rows_by_id:
            return redirect(url_for("index", reviewer=reviewer))
        row = rows_by_id[row_id]

        def radio(name: str) -> bool | None:
            raw = request.form.get(name)
            return (raw == "yes") if raw in ("yes", "no") else None

        suggested_code = request.form.get("suggested_code", "").strip()
        with _write_lock:
            append_judgment(
                output_path,
                {
                    "row_id": row_id,
                    "reviewer": reviewer,
                    TEXT_COLUMN: row["libelle"],
                    CODE_COLUMN: row["current_code"],
                    VERDICT_COLUMN: row["verdict"],
                    EXPLANATION_COLUMN: row["explanation"],
                    CONFIDENCE_COLUMN: row["confidence"],
                    "human_code_correct": radio("code_correct"),
                    "human_verdict_correct": radio("verdict_correct"),
                    "human_suggested_code": suggested_code or None,
                    "reviewed_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            write_review_parquet(output_path, load_judgments(output_path))

        order = order_for(reviewer)
        idx = order.index(row_id)
        if idx + 1 < len(order):
            return redirect(url_for("review", row_id=order[idx + 1], reviewer=reviewer))
        return redirect(url_for("index", reviewer=reviewer))

    @app.route("/metrics")
    def metrics():
        all_judgments = load_judgments(output_path)
        m = compute_metrics(rows, all_judgments, shared_ids)
        m["n_assigned_total"] = sum(len(order_for(r)) for r in reviewers)
        return render_template_string(METRICS_TEMPLATE, m=m, run=run)

    return app


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Human review app for match_verifier_eval.py's output"
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help="Root directory written by src.evaluation.match_verifier_eval; the run "
        f"actually reviewed is <input>/<commit>/<model>/{DETAILS_FILENAME}. A path "
        "ending in .parquet is used as-is",
    )
    parser.add_argument(
        "--commit",
        default=None,
        help="Which eval run to review, by the commit tag its output directory is "
        "named after (default: the current HEAD's tag)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Which eval run to review, by the model its output directory is named "
        "after (default: GENERATION_MODEL, i.e. the model this checkout would use)",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Root directory for the judgments; they are logged to "
        f"<output-dir>/<commit>/<model>/{REVIEW_FILENAME}, with the review parquet "
        "(carrying the `reviewer` column) written next to it under the same basename",
    )
    parser.add_argument(
        "--reviewers",
        required=True,
        help="Comma-separated reviewer names (e.g. meilame,theo,nathan). Keep this, "
        "--shared-n, --unique-n and --input unchanged for the whole review period: "
        "the row split is deterministically derived from all four, so changing any "
        "of them reshuffles it out from under in-progress work.",
    )
    parser.add_argument(
        "--shared-n", type=int, default=50, help="Rows every reviewer sees (default: 50)"
    )
    parser.add_argument(
        "--unique-n", type=int, default=50, help="Extra rows unique to each reviewer (default: 50)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Row-split seed (default: 42)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5052)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--url-prefix",
        default=None,
        help="Path prefix to prepend to generated links (default: auto-detect Onyxia's "
        "/proxy/<port> reverse proxy from VSCODE_PROXY_URI; pass '' to disable)",
    )
    args = parser.parse_args()

    reviewers = [r.strip() for r in args.reviewers.split(",") if r.strip()]
    if not reviewers:
        parser.error("--reviewers needs at least one name")

    commit = args.commit or revision_tag()
    model = model_slug(args.model)
    run = f"{commit}/{model}"
    input_path = resolve_input(args.input, commit, model)
    output_path = resolve_output(args.output_dir, commit, model)
    logger.info(f"Reviewing run {run}: {input_path} -> {output_path}")

    if args.url_prefix is not None:
        url_prefix = args.url_prefix
    elif os.environ.get("VSCODE_PROXY_URI"):
        url_prefix = f"/proxy/{args.port}"
    else:
        url_prefix = ""

    app = create_app(
        input_path,
        output_path,
        reviewers,
        run,
        args.shared_n,
        args.unique_n,
        args.seed,
        url_prefix=url_prefix,
    )
    if url_prefix:
        logger.info(f"Prefixing generated links with {url_prefix!r} (Onyxia reverse proxy)")
    app.run(host=args.host, port=args.port, debug=args.debug)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
