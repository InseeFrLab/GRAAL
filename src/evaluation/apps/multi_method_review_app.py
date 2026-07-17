"""Standalone web app for human review of evaluate_eval_set_multi_method.py's output.

For each row of the multi-method eval parquet (cf.
src.evaluation.evaluate_eval_set_multi_method), shows the activity text and every
distinct candidate code proposed for it (ground truth + Navigator + Agentic RAG +
Summary + supervised model, deduplicated and grouped by which method(s) proposed
each one, same collapse as before), each with its official notice, CodeChooser's
own arbitration (when it ran), and MatchVerifier's verdict on that specific
candidate (taken from whichever method proposed it first, since predictions that
agree on the same code get near-identical verdicts). Independently for each
candidate, a human judges: is this code actually correct, and is MatchVerifier's
verdict on it right? This is what scores CodeChooser's arbitration, each method's
raw proposal, and MatchVerifier itself against something other than a training
label of unknown quality.

Three reviewers each work through a ~100-row slice: a shared pool every reviewer
sees (for inter-rater agreement) plus a slice unique to them (cf. --reviewers/
--shared-n/--unique-n/--seed). The split is deterministic from the input file and
those arguments, so reviewing is fully asynchronous — anyone can pick up their own
progress at any time, as long as the input file and those four arguments don't
change mid-review.

Judgments are logged append-only to JSONL (keyed by (reviewer_id, row_id), so
re-running is idempotent and safe to interrupt). The /metrics page derives, from
reviewed rows only: each method's accuracy against the human-judged candidates,
whether CodeChooser's own pick matches a human-judged-correct candidate, whether
MatchVerifier's verdicts hold up (both via the human's direct yes/no on each
verdict, and via precision/recall against the human's correctness judgments), and
inter-rater agreement on the shared pool.

Supersedes human_review_app.py (retired): that app only ever reviewed
MatchVerifier's verdict on a single training label; this one covers all 5
candidates (ground truth + 4 classifiers) on the eval set, MatchVerifier's verdict
on each of them, and CodeChooser's arbitration, in one pass.

Nécessite Neo4j configuré dans l'environnement pour afficher les notices des
codes candidats ; sans connexion, la revue reste possible mais sans notice.

Usage :
    uv run -m src.evaluation.apps.multi_method_review_app \
        --input data/eval/multi_method/eval_multi_method.parquet \
        --output data/eval/human_review/multi_method_review.jsonl \
        --reviewers alice,bob,carol \
        --port 5051
"""

import argparse
import json
import logging
import os
import random
from datetime import datetime, timezone

import polars as pl
from flask import Flask, redirect, render_template_string, request, url_for

from src.config import neo4j_config
from src.evaluation.metrics import normalize_code
from src.evaluation.row_id import row_id_for
from src.neo4j_graph.graph import Graph
from src.utils import storage
from src.utils.logging import configure_logging

configure_logging()
logger = logging.getLogger(__name__)

DEFAULT_INPUT = "data/eval/multi_method/eval_multi_method.parquet"
DEFAULT_OUTPUT = "data/eval/human_review/multi_method_review.jsonl"
TEXT_COLUMN = "libelle"

METHOD_LABELS = {
    "navigator": "Navigator",
    "agentic_rag": "Agentic RAG",
    "summary": "Summary",
    "supervised": "Supervisé",
}
LABEL_TO_METHOD = {label: name for name, label in METHOD_LABELS.items()}
GROUND_TRUTH_LABEL = "Vérité terrain"

# Per source label, the (is_match, confidence, explanation) column names to pull a
# candidate's MatchVerifier verdict from — only the first source to propose a given
# normalized code contributes its verdict to that candidate (cf. build_candidates).
VERIFY_COLUMNS = {
    GROUND_TRUTH_LABEL: (
        "ground_truth_is_match",
        "ground_truth_verify_confidence",
        "ground_truth_verify_explanation",
    ),
    **{
        label: (f"{name}_is_match", f"{name}_verify_confidence", f"{name}_verify_explanation")
        for name, label in METHOD_LABELS.items()
    },
}


def build_candidates(row: dict, code_column: str) -> list[dict]:
    """Dedupe (ground truth + each method's code) by normalized value.

    Returns a list of {code, norm, sources, verifier_is_match, verifier_confidence,
    verifier_explanation} ordered by first appearance (ground truth first, then
    methods in METHOD_LABELS order), `sources` being the list of labels that
    proposed this normalized code. The verifier fields come from whichever source
    proposed the code first — predictions that agree on the same code are expected
    to get near-identical MatchVerifier verdicts, so a human reviews one verdict per
    candidate rather than one per source.
    """
    raw = [(GROUND_TRUTH_LABEL, row[code_column])]
    raw += [(label, row.get(f"{name}_code")) for name, label in METHOD_LABELS.items()]

    by_norm: dict[str, dict] = {}
    order: list[str] = []
    for source, code in raw:
        norm = normalize_code(code)
        if norm is None:
            continue
        if norm not in by_norm:
            is_match_col, conf_col, expl_col = VERIFY_COLUMNS[source]
            by_norm[norm] = {
                "code": code,
                "norm": norm,
                "sources": [],
                "verifier_is_match": row.get(is_match_col),
                "verifier_confidence": row.get(conf_col),
                "verifier_explanation": row.get(expl_col),
            }
            order.append(norm)
        by_norm[norm]["sources"].append(source)
    return [by_norm[norm] for norm in order]


def load_rows(input_path: str, code_column: str) -> list[dict]:
    with storage.open_path(input_path, "rb") as f:
        df = pl.read_parquet(f)
    rows = []
    for r in df.to_dicts():
        row_id = row_id_for(r[TEXT_COLUMN], r[code_column])
        rows.append(
            {
                "row_id": row_id,
                "libelle": r[TEXT_COLUMN],
                "ground_truth_code": r[code_column],
                "chosen_code": r.get("chosen_code"),
                "chooser_confidence": r.get("chooser_confidence"),
                "chooser_explanation": r.get("chooser_explanation"),
                "n_unique_candidates": r.get("n_unique_candidates"),
                "method_codes": {name: r.get(f"{name}_code") for name in METHOD_LABELS},
                "candidates": build_candidates(r, code_column),
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
    """{reviewer_id: {row_id: judgment_entry}} — last line wins per (reviewer, row)."""
    judgments: dict[str, dict[str, dict]] = {}
    if not storage.path_exists(output_path):
        return judgments
    with storage.open_path(output_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            judgments.setdefault(entry["reviewer_id"], {})[entry["row_id"]] = entry
    return judgments


def append_judgment(output_path: str, entry: dict) -> None:
    storage.makedirs(os.path.dirname(output_path) or ".")
    with storage.open_path(output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def compute_metrics(
    rows: list[dict], all_judgments: dict[str, dict[str, dict]], shared_ids: set[str]
) -> dict:
    """Metrics derived from every (reviewer, row) judgment pooled together.

    A method/ground-truth/chooser's accuracy denominator is judgments where it
    actually produced a code (a None/failed prediction is excluded rather than
    counted wrong). A candidate's MatchVerifier verdict is attributed to every
    source method sharing its normalized code, consistent with the review UI
    collapsing them into one judged card.
    """
    rows_by_id = {r["row_id"]: r for r in rows}

    n_judgments = 0
    ground_truth_total = ground_truth_correct = 0
    chooser_total = chooser_correct = 0
    method_totals = {name: 0 for name in METHOD_LABELS}
    method_correct = {name: 0 for name in METHOD_LABELS}
    method_verify_totals = {name: 0 for name in METHOD_LABELS}
    method_verify_agree = {name: 0 for name in METHOD_LABELS}
    verify_tp = verify_fp = verify_tn = verify_fn = 0

    for by_row in all_judgments.values():
        for row_id, judgment in by_row.items():
            row = rows_by_id.get(row_id)
            if row is None:
                continue
            n_judgments += 1
            correct_norms = {
                normalize_code(c["code"]) for c in judgment["candidates"] if c.get("correct")
            }

            gt_norm = normalize_code(row["ground_truth_code"])
            ground_truth_total += 1
            if gt_norm in correct_norms:
                ground_truth_correct += 1

            chosen_norm = normalize_code(row["chosen_code"])
            if chosen_norm is not None:
                chooser_total += 1
                if chosen_norm in correct_norms:
                    chooser_correct += 1

            for name in METHOD_LABELS:
                code = row["method_codes"].get(name)
                if code is None:
                    continue
                method_totals[name] += 1
                if normalize_code(code) in correct_norms:
                    method_correct[name] += 1

            for c in judgment["candidates"]:
                norm = normalize_code(c["code"])
                is_correct = norm in correct_norms
                verifier_is_match = c.get("verifier_is_match")
                verdict_correct = c.get("verifier_verdict_correct")

                if verifier_is_match is not None:
                    if verifier_is_match and is_correct:
                        verify_tp += 1
                    elif verifier_is_match and not is_correct:
                        verify_fp += 1
                    elif not verifier_is_match and is_correct:
                        verify_fn += 1
                    else:
                        verify_tn += 1

                if verdict_correct is not None:
                    for source in c.get("sources", []):
                        name = LABEL_TO_METHOD.get(source)
                        if name is None:
                            continue
                        method_verify_totals[name] += 1
                        if verdict_correct:
                            method_verify_agree[name] += 1

    def rate(numerator: int, denominator: int) -> float | None:
        return numerator / denominator if denominator else None

    verify_n = verify_tp + verify_fp + verify_tn + verify_fn
    verifier_overall = {
        "n": verify_n,
        "precision": rate(verify_tp, verify_tp + verify_fp),
        "recall": rate(verify_tp, verify_tp + verify_fn),
        "accuracy": rate(verify_tp + verify_tn, verify_n),
    }

    agreement = compute_inter_rater_agreement(all_judgments, shared_ids)

    return {
        "n_total": len(rows),
        "n_judgments": n_judgments,
        "ground_truth_accuracy": rate(ground_truth_correct, ground_truth_total),
        "ground_truth_n": ground_truth_total,
        "chooser_accuracy": rate(chooser_correct, chooser_total),
        "chooser_n": chooser_total,
        "methods": {
            name: {
                "label": label,
                "accuracy": rate(method_correct[name], method_totals[name]),
                "n": method_totals[name],
                "verifier_agreement": rate(method_verify_agree[name], method_verify_totals[name]),
                "verifier_n": method_verify_totals[name],
            }
            for name, label in METHOD_LABELS.items()
        },
        "verifier_overall": verifier_overall,
        "agreement": agreement,
    }


def compute_inter_rater_agreement(
    all_judgments: dict[str, dict[str, dict]], shared_ids: set[str]
) -> dict:
    """Percent agreement between every pair of reviewers on the shared pool, matched
    by normalized candidate code, for both the "correct?" and "verifier verdict
    correct?" judgments.
    """
    reviewers = sorted(all_judgments)
    pairs = []
    correct_match = correct_total = 0
    verify_match = verify_total = 0

    for i in range(len(reviewers)):
        for j in range(i + 1, len(reviewers)):
            a, b = reviewers[i], reviewers[j]
            pair_match = pair_total = 0
            for row_id in shared_ids:
                ja = all_judgments[a].get(row_id)
                jb = all_judgments[b].get(row_id)
                if ja is None or jb is None:
                    continue
                by_norm_a = {normalize_code(c["code"]): c for c in ja["candidates"]}
                by_norm_b = {normalize_code(c["code"]): c for c in jb["candidates"]}
                for norm, ca in by_norm_a.items():
                    cb = by_norm_b.get(norm)
                    if cb is None:
                        continue
                    pair_total += 1
                    correct_total += 1
                    if ca.get("correct") == cb.get("correct"):
                        pair_match += 1
                        correct_match += 1
                    if (
                        ca.get("verifier_verdict_correct") is not None
                        and cb.get("verifier_verdict_correct") is not None
                    ):
                        verify_total += 1
                        if ca["verifier_verdict_correct"] == cb["verifier_verdict_correct"]:
                            verify_match += 1
            if pair_total:
                pairs.append(
                    {"reviewers": (a, b), "n": pair_total, "agreement": pair_match / pair_total}
                )

    return {
        "pairs": pairs,
        "overall_correct_agreement": correct_match / correct_total if correct_total else None,
        "overall_verify_agreement": verify_match / verify_total if verify_total else None,
        "n_correct": correct_total,
        "n_verify": verify_total,
    }


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
    if code in _code_notice_cache:
        return _code_notice_cache[code]
    graph = get_graph()
    notice = None
    if graph is not None:
        try:
            info = graph.get_code_information(code)
            if info and not info.get("error"):
                notice = info
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
  .notice { white-space: pre-line; font-size: 0.9rem; }
  .muted { color: #777; font-size: 0.9rem; margin-top: 0.4rem; }
  .sources { font-size: 0.8rem; color: #777; margin-top: 0.2rem; }
  .chooser-badge { display: inline-block; padding: 0.1rem 0.5rem; border-radius: 999px;
    background: #eef2ff; color: #3949ab; font-size: 0.75rem; font-weight: 700; margin-left: 0.4rem; }
  @media (prefers-color-scheme: dark) { .chooser-badge { background: #2a3350; color: #cfd9ff; } }
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
</style>
"""

NAV = """
<nav>
  <a href="{{ url_for('index', reviewer=reviewer) }}">&larr; Revue ({{ reviewer }})</a>
  <span class="progress">{{ n_reviewed }} / {{ n_total }} jugés</span>
  <a href="{{ url_for('metrics') }}">Métriques &rarr;</a>
</nav>
"""

PICKER_TEMPLATE = (
    STYLE
    + """
<h2>Qui êtes-vous ?</h2>
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
{% if existing %}
  <div class="existing-banner">Déjà jugé par {{ reviewer }} le {{ existing.reviewed_at }}</div>
{% endif %}

<div class="card">
  <div class="label">Activité ({{ idx }} / {{ total }})</div>
  <div class="activite">{{ row.libelle }}</div>
</div>

<form method="post" action="{{ url_for('judge', row_id=row.row_id, reviewer=reviewer) }}">

<div class="label" style="margin-bottom: 0.5rem;">
  Pour chaque candidat : le code est-il correct ? le verdict MatchVerifier est-il correct ?
  {% if row.n_unique_candidates %}<span class="muted">({{ row.n_unique_candidates }} candidat(s) distinct(s))</span>{% endif %}
</div>

{% for c in row.candidates %}
{% set existing_c = existing_by_norm.get(c.norm) if existing else none %}
<div class="candidate">
  <div class="candidate-info">
    <span class="code-badge">{{ c.code }}</span>
    {% if c.norm == chosen_norm %}<span class="chooser-badge">choix CodeChooser</span>{% endif %}
    {% set notice = notices.get(c.norm) %}
    {% if notice %} &mdash; {{ notice.name }}{% endif %}
    <div class="sources">Proposé par : {{ c.sources | join(", ") }}</div>
    {% if notice and notice.description %}<div class="notice">{{ notice.description }}</div>{% endif %}
    {% if c.verifier_is_match is not none %}
      <div class="muted">
        MatchVerifier : {{ "match" if c.verifier_is_match else "no match" }}
        {% if c.verifier_confidence is not none %}
          (confiance {{ "%.0f"|format(c.verifier_confidence * 100) }}%)
        {% endif %}
        — {{ c.verifier_explanation }}
      </div>
    {% endif %}
  </div>
  <div class="judgments">
    <div class="judgment-group">
      <span class="jg-label">Code correct ?</span>
      <label><input type="radio" name="correct__{{ c.norm }}" value="yes" required
        {{ "checked" if existing_c and existing_c.correct }}> Oui</label>
      <label><input type="radio" name="correct__{{ c.norm }}" value="no"
        {{ "checked" if existing_c and not existing_c.correct }}> Non</label>
    </div>
    {% if c.verifier_is_match is not none %}
    <div class="judgment-group">
      <span class="jg-label">Verdict MatchVerifier correct ?</span>
      <label><input type="radio" name="verifier__{{ c.norm }}" value="yes" required
        {{ "checked" if existing_c and existing_c.verifier_verdict_correct }}> Oui</label>
      <label><input type="radio" name="verifier__{{ c.norm }}" value="no"
        {{ "checked" if existing_c and existing_c.verifier_verdict_correct == false }}> Non</label>
    </div>
    {% endif %}
  </div>
</div>
{% endfor %}

<div class="label" style="margin-bottom: 0.3rem;">Aucun des candidats ci-dessus n'est correct ?</div>
<div class="other-row">
  <input type="text" name="other_code" placeholder="Code correct (optionnel)"
    value="{{ existing.other_code if existing else '' }}">
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

<h2>Métriques multi-méthode</h2>

<div class="stat-grid">
  <div class="stat"><span class="n">{{ m.n_judgments }}/{{ m.n_assigned_total }}</span><span class="l">Jugements</span></div>
  <div class="stat"><span class="n">{{ "%.0f%%"|format(m.ground_truth_accuracy * 100) if m.ground_truth_accuracy is not none else "—" }}</span><span class="l">Vérité terrain ({{ m.ground_truth_n }})</span></div>
  <div class="stat"><span class="n">{{ "%.0f%%"|format(m.chooser_accuracy * 100) if m.chooser_accuracy is not none else "—" }}</span><span class="l">CodeChooser ({{ m.chooser_n }})</span></div>
  {% for name, s in m.methods.items() %}
  <div class="stat"><span class="n">{{ "%.0f%%"|format(s.accuracy * 100) if s.accuracy is not none else "—" }}</span><span class="l">{{ s.label }} ({{ s.n }})</span></div>
  {% endfor %}
</div>

<h3>MatchVerifier</h3>
<div class="stat-grid">
  <div class="stat"><span class="n">{{ "%.0f%%"|format(m.verifier_overall.accuracy * 100) if m.verifier_overall.accuracy is not none else "—" }}</span><span class="l">Exactitude ({{ m.verifier_overall.n }})</span></div>
  <div class="stat"><span class="n">{{ "%.0f%%"|format(m.verifier_overall.precision * 100) if m.verifier_overall.precision is not none else "—" }}</span><span class="l">Précision</span></div>
  <div class="stat"><span class="n">{{ "%.0f%%"|format(m.verifier_overall.recall * 100) if m.verifier_overall.recall is not none else "—" }}</span><span class="l">Rappel</span></div>
  {% for name, s in m.methods.items() %}
  <div class="stat"><span class="n">{{ "%.0f%%"|format(s.verifier_agreement * 100) if s.verifier_agreement is not none else "—" }}</span><span class="l">Verdict correct : {{ s.label }} ({{ s.verifier_n }})</span></div>
  {% endfor %}
</div>
<p class="muted">
  Exactitude/précision/rappel : verdict MatchVerifier (is_match) comparé au jugement
  humain de correction du code. "Verdict correct" par méthode : part des verdicts que
  l'humain a explicitement jugés corrects (un verdict partagé par plusieurs méthodes
  via un code identique compte pour chacune).
</p>

<h3>Accord inter-annotateurs (pool partagé)</h3>
<div class="stat-grid">
  <div class="stat"><span class="n">{{ "%.0f%%"|format(m.agreement.overall_correct_agreement * 100) if m.agreement.overall_correct_agreement is not none else "—" }}</span><span class="l">Accord "correct" ({{ m.agreement.n_correct }})</span></div>
  <div class="stat"><span class="n">{{ "%.0f%%"|format(m.agreement.overall_verify_agreement * 100) if m.agreement.overall_verify_agreement is not none else "—" }}</span><span class="l">Accord verdict ({{ m.agreement.n_verify }})</span></div>
</div>
{% if m.agreement.pairs %}
<table>
  <tr><th>Paire</th><th>N</th><th>Accord "correct"</th></tr>
  {% for p in m.agreement.pairs %}
  <tr>
    <td>{{ p.reviewers[0] }} / {{ p.reviewers[1] }}</td>
    <td>{{ p.n }}</td>
    <td>{{ "%.0f%%"|format(p.agreement * 100) }}</td>
  </tr>
  {% endfor %}
</table>
{% endif %}
"""
)


class PrefixMiddleware:
    """Same Onyxia reverse-proxy handling as before (cf. VSCODE_PROXY_URI)."""

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
    code_column: str,
    reviewers: list[str],
    shared_n: int,
    unique_n: int,
    seed: int,
    url_prefix: str = "",
) -> Flask:
    app = Flask(__name__)
    if url_prefix:
        app.wsgi_app = PrefixMiddleware(app.wsgi_app, url_prefix)

    rows = load_rows(input_path, code_column)
    rows_by_id = {r["row_id"]: r for r in rows}
    row_ids = [r["row_id"] for r in rows]
    assignment, shared_ids = build_assignment(row_ids, reviewers, shared_n, unique_n, seed)
    logger.info(
        f"Loaded {len(rows)} rows; {shared_n} shared + {unique_n} unique per reviewer "
        f"({', '.join(reviewers)})"
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
            return render_template_string(PICKER_TEMPLATE, reviewers=reviewers)
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
        chosen_norm = normalize_code(row["chosen_code"])
        notices = {c["norm"]: get_code_notice(c["code"]) for c in row["candidates"]}
        idx = order.index(row_id)
        existing = judgments.get(row_id)
        existing_by_norm = (
            {normalize_code(c["code"]): c for c in existing["candidates"]} if existing else {}
        )
        return render_template_string(
            REVIEW_TEMPLATE,
            row=row,
            reviewer=reviewer,
            chosen_norm=chosen_norm,
            notices=notices,
            idx=idx + 1,
            total=len(order),
            n_reviewed=len(judgments),
            n_total=len(order),
            prev_id=order[idx - 1] if idx > 0 else None,
            next_id=order[idx + 1] if idx + 1 < len(order) else None,
            existing=existing,
            existing_by_norm=existing_by_norm,
        )

    @app.route("/judge/<row_id>", methods=["POST"])
    def judge(row_id):
        reviewer = request.args.get("reviewer")
        if reviewer not in reviewers or row_id not in rows_by_id:
            return redirect(url_for("index", reviewer=reviewer))
        row = rows_by_id[row_id]

        candidates_out = []
        for c in row["candidates"]:
            key = c["norm"]
            correct = request.form.get(f"correct__{key}") == "yes"
            verdict_correct = None
            if c.get("verifier_is_match") is not None:
                raw = request.form.get(f"verifier__{key}")
                if raw in ("yes", "no"):
                    verdict_correct = raw == "yes"
            candidates_out.append(
                {
                    "code": c["code"],
                    "sources": c["sources"],
                    "correct": correct,
                    "verifier_is_match": c.get("verifier_is_match"),
                    "verifier_confidence": c.get("verifier_confidence"),
                    "verifier_explanation": c.get("verifier_explanation"),
                    "verifier_verdict_correct": verdict_correct,
                }
            )

        other_code = request.form.get("other_code", "").strip()
        if other_code:
            candidates_out.append(
                {
                    "code": other_code,
                    "sources": ["Autre (revue)"],
                    "correct": True,
                    "verifier_is_match": None,
                    "verifier_confidence": None,
                    "verifier_explanation": None,
                    "verifier_verdict_correct": None,
                }
            )

        append_judgment(
            output_path,
            {
                "row_id": row_id,
                "reviewer_id": reviewer,
                "libelle": row["libelle"],
                "chosen_code": row["chosen_code"],
                "other_code": other_code or None,
                "candidates": candidates_out,
                "reviewed_at": datetime.now(timezone.utc).isoformat(),
            },
        )

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
        return render_template_string(METRICS_TEMPLATE, m=m)

    return app


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Human review app for evaluate_eval_set_multi_method.py's output"
    )
    parser.add_argument(
        "--input", default=DEFAULT_INPUT, help="Parquet produced by evaluate_eval_set_multi_method"
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="JSONL log of human judgments")
    parser.add_argument(
        "--code-column", default="apet2025", help="Ground-truth code column (default: apet2025)"
    )
    parser.add_argument(
        "--reviewers",
        required=True,
        help="Comma-separated reviewer names (e.g. alice,bob,carol). Keep this, "
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
    parser.add_argument("--port", type=int, default=5051)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--url-prefix",
        default=None,
        help="Path prefix to prepend to generated links (default: auto-detect Onyxia's "
        "/proxy/<port> reverse proxy from VSCODE_PROXY_URI; pass '' to disable)",
    )
    args = parser.parse_args()

    reviewers = [r.strip() for r in args.reviewers.split(",") if r.strip()]
    if len(reviewers) < 2:
        parser.error("--reviewers needs at least 2 names")

    if args.url_prefix is not None:
        url_prefix = args.url_prefix
    elif os.environ.get("VSCODE_PROXY_URI"):
        url_prefix = f"/proxy/{args.port}"
    else:
        url_prefix = ""

    app = create_app(
        args.input,
        args.output,
        args.code_column,
        reviewers,
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
