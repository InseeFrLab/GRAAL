"""Standalone web app for human review of evaluate_eval_set_multi_method.py's output.

For each row of the multi-method eval parquet (cf.
src.evaluation.evaluate_eval_set_multi_method), shows the activity text, every
candidate code proposed for it (ground truth + Navigator + Agentic RAG +
Summary + supervised model, deduplicated and grouped by which method(s)
proposed each one), the official notice for each candidate, and CodeChooser's
own arbitration (when it ran). A human picks which candidate is actually
correct (or enters a different code entirely) — this human-derived truth is
the only way to score CodeChooser's arbitration and each method's raw
proposal against something other than a training label of unknown quality.

Judgments are logged append-only to JSONL (row_id keyed, so re-running is
idempotent and safe to interrupt). The /metrics page derives, from reviewed
rows only: each method's accuracy against the human-judged correct code,
whether CodeChooser's own pick matches it, and whether the original ground
truth label was itself correct.

Separate from human_review_app.py (which reviews MatchVerifier's verdict on
training labels) because the input shape and the judgment being collected are
different: there, the question is "is this is_match verdict right?"; here,
it's "which of these N candidate codes is actually right?".

Nécessite Neo4j configuré dans l'environnement pour afficher les notices des
codes candidats (comme human_review_app.py) ; sans connexion, la revue reste
possible mais sans notice.

Usage :
    uv run -m src.evaluation.apps.multi_method_review_app \
        --input data/eval/multi_method/eval_multi_method.parquet \
        --output data/eval/human_review/multi_method_review.jsonl \
        --port 5051
"""

import argparse
import hashlib
import json
import logging
import os
from datetime import datetime, timezone

import polars as pl
from flask import Flask, redirect, render_template_string, request, url_for

from src.config import neo4j_config
from src.evaluation.metrics import normalize_code
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


def row_id_for(libelle: str, ground_truth_code: str) -> str:
    return hashlib.sha1(f"{libelle}|{ground_truth_code}".encode("utf-8")).hexdigest()[:12]


def build_candidates(row: dict, code_column: str) -> list[dict]:
    """Dedupe (ground truth + each method's code) by normalized value.

    Returns a list of {code, sources} ordered by first appearance (ground
    truth first, then methods in METHOD_LABELS order), `sources` being the
    list of labels ("Vérité terrain", "Navigator", ...) that proposed this
    normalized code.
    """
    raw = [("Vérité terrain", row[code_column])]
    raw += [(label, row.get(f"{name}_code")) for name, label in METHOD_LABELS.items()]

    by_norm: dict[str, dict] = {}
    order: list[str] = []
    for source, code in raw:
        norm = normalize_code(code)
        if norm is None:
            continue
        if norm not in by_norm:
            by_norm[norm] = {"code": code, "norm": norm, "sources": []}
            order.append(norm)
        by_norm[norm]["sources"].append(source)
    return [by_norm[norm] for norm in order]


def load_rows(input_path: str, code_column: str, n: int | None, seed: int) -> list[dict]:
    with storage.open_path(input_path, "rb") as f:
        df = pl.read_parquet(f)
    if n is not None and n < len(df):
        df = df.sample(n=n, seed=seed)
    rows = []
    for r in df.to_dicts():
        row_id = row_id_for(r[TEXT_COLUMN], r[code_column])
        rows.append(
            {
                "row_id": row_id,
                "libelle": r[TEXT_COLUMN],
                "ground_truth_code": r[code_column],
                "ground_truth_is_match": r.get("ground_truth_is_match"),
                "ground_truth_verify_confidence": r.get("ground_truth_verify_confidence"),
                "ground_truth_verify_explanation": r.get("ground_truth_verify_explanation"),
                "chosen_code": r.get("chosen_code"),
                "chooser_confidence": r.get("chooser_confidence"),
                "chooser_explanation": r.get("chooser_explanation"),
                "n_unique_candidates": r.get("n_unique_candidates"),
                "method_codes": {name: r.get(f"{name}_code") for name in METHOD_LABELS},
                "candidates": build_candidates(r, code_column),
            }
        )
    return rows


def load_judgments(output_path: str) -> dict[str, dict]:
    judgments = {}
    if not storage.path_exists(output_path):
        return judgments
    with storage.open_path(output_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            judgments[entry["row_id"]] = entry
    return judgments


def append_judgment(output_path: str, entry: dict) -> None:
    storage.makedirs(os.path.dirname(output_path) or ".")
    with storage.open_path(output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def compute_metrics(rows: list[dict], judgments: dict[str, dict]) -> dict:
    """Per-method and per-chooser accuracy against the human-judged correct code.

    Only rows with a judgment count; a method's accuracy denominator is the
    number of reviewed rows where that method actually produced a code (a
    None/failed prediction is excluded rather than counted wrong, since it's
    a distinct failure mode already visible via evaluate_eval_set_multi_method's
    own timing/failure logs).
    """
    n_reviewed = 0
    ground_truth_correct = 0
    chooser_total = chooser_correct = 0
    method_totals = {name: 0 for name in METHOD_LABELS}
    method_correct = {name: 0 for name in METHOD_LABELS}

    for row in rows:
        judgment = judgments.get(row["row_id"])
        if judgment is None:
            continue
        n_reviewed += 1
        correct_norm = normalize_code(judgment["correct_code"])

        if normalize_code(row["ground_truth_code"]) == correct_norm:
            ground_truth_correct += 1

        if row["chosen_code"] is not None:
            chooser_total += 1
            if normalize_code(row["chosen_code"]) == correct_norm:
                chooser_correct += 1

        for name in METHOD_LABELS:
            code = row["method_codes"].get(name)
            if code is None:
                continue
            method_totals[name] += 1
            if normalize_code(code) == correct_norm:
                method_correct[name] += 1

    def rate(correct: int, total: int) -> float | None:
        return correct / total if total else None

    return {
        "n_total": len(rows),
        "n_reviewed": n_reviewed,
        "ground_truth_accuracy": rate(ground_truth_correct, n_reviewed),
        "chooser_accuracy": rate(chooser_correct, chooser_total),
        "chooser_n": chooser_total,
        "methods": {
            name: {
                "accuracy": rate(method_correct[name], method_totals[name]),
                "n": method_totals[name],
                "label": label,
            }
            for name, label in METHOD_LABELS.items()
        },
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
    max-width: 820px; margin: 2rem auto; padding: 0 1.25rem;
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
  .muted { color: #777; font-size: 0.9rem; }
  .sources { font-size: 0.8rem; color: #777; margin-top: 0.2rem; }
  .chooser-badge { display: inline-block; padding: 0.1rem 0.5rem; border-radius: 999px;
    background: #eef2ff; color: #3949ab; font-size: 0.75rem; font-weight: 700; margin-left: 0.4rem; }
  @media (prefers-color-scheme: dark) { .chooser-badge { background: #2a3350; color: #cfd9ff; } }
  .candidate {
    border: 1px solid #ddd; border-radius: 10px; padding: 1rem 1.25rem;
    margin-bottom: 0.75rem; background: #fafafa; display: flex; justify-content: space-between;
    align-items: flex-start; gap: 1rem;
  }
  .candidate-info { flex: 1; }
  .candidate form { flex-shrink: 0; }
  button.pick {
    font-size: 0.95rem; font-weight: 700; padding: 0.6rem 1.2rem; border-radius: 8px;
    border: none; cursor: pointer; background: #16a34a; color: white;
  }
  button.pick:hover { opacity: 0.88; }
  .other-row { display: flex; gap: 0.5rem; margin-top: 1rem; }
  .other-row input[type=text] {
    flex: 1; padding: 0.5rem 0.7rem; border-radius: 8px; border: 1px solid #ccc; font-family: monospace;
  }
  .other-row button {
    padding: 0.5rem 1rem; border-radius: 8px; border: 1px solid #ccc; background: #f1f1f1; cursor: pointer;
  }
  .existing-banner { padding: 0.5rem 0.9rem; border-radius: 8px; margin-bottom: 1rem;
    font-weight: 600; background: #d7f5df; color: #166534; }
  @media (prefers-color-scheme: dark) { .existing-banner { background: #14351f; color: #7be0a0; } }
  .nav-links { display: flex; justify-content: space-between; margin-top: 1rem; }
  .nav-links a.disabled { pointer-events: none; color: #bbb; }
  table { border-collapse: collapse; width: 100%; margin: 1rem 0; }
  th, td { text-align: left; padding: 0.4rem 0.6rem; border-bottom: 1px solid #e5e5e5; font-size: 0.9rem; }
  @media (prefers-color-scheme: dark) { th, td { border-color: #3a3a3a; } }
  .stat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 0.75rem; margin: 1.25rem 0; }
  .stat { border: 1px solid #ddd; border-radius: 10px; padding: 0.8rem; text-align: center; background: #fafafa; }
  .stat .n { font-size: 1.5rem; font-weight: 700; display: block; }
  .stat .l { font-size: 0.75rem; color: #777; text-transform: uppercase; letter-spacing: 0.03em; }
</style>
"""

NAV = """
<nav>
  <a href="{{ url_for('index') }}">&larr; Revue</a>
  <span class="progress">{{ n_reviewed }} / {{ n_total }} jugés</span>
  <a href="{{ url_for('metrics') }}">Métriques &rarr;</a>
</nav>
"""

REVIEW_TEMPLATE = (
    STYLE
    + NAV
    + """
{% if existing_code %}
  <div class="existing-banner">Déjà jugé : code correct marqué <span class="code-badge">{{ existing_code }}</span></div>
{% endif %}

<div class="card">
  <div class="label">Activité ({{ idx }} / {{ total }})</div>
  <div class="activite">{{ row.libelle }}</div>
  {% if row.ground_truth_is_match is not none %}
    <div class="muted">
      MatchVerifier sur la vérité terrain :
      {{ "match" if row.ground_truth_is_match else "no match" }}
      {% if row.ground_truth_verify_confidence is not none %}
        (confiance {{ "%.0f"|format(row.ground_truth_verify_confidence * 100) }}%)
      {% endif %}
      — {{ row.ground_truth_verify_explanation }}
    </div>
  {% endif %}
</div>

<div class="label" style="margin-bottom: 0.5rem;">
  Quel code est réellement correct ?
  {% if row.n_unique_candidates %}<span class="muted">({{ row.n_unique_candidates }} candidat(s) distinct(s))</span>{% endif %}
</div>

{% for c in row.candidates %}
<div class="candidate">
  <div class="candidate-info">
    <span class="code-badge">{{ c.code }}</span>
    {% if c.norm == chosen_norm %}<span class="chooser-badge">choix CodeChooser</span>{% endif %}
    {% set notice = notices.get(c.norm) %}
    {% if notice %} &mdash; {{ notice.name }}{% endif %}
    <div class="sources">Proposé par : {{ c.sources | join(", ") }}</div>
    {% if notice and notice.description %}<div class="notice">{{ notice.description }}</div>{% endif %}
  </div>
  <form method="post" action="{{ url_for('judge', row_id=row.row_id) }}">
    <input type="hidden" name="correct_code" value="{{ c.code }}">
    <button type="submit" class="pick">Correct</button>
  </form>
</div>
{% endfor %}

{% if row.chosen_code and row.chosen_norm not in row.candidate_norms %}
<div class="candidate">
  <div class="candidate-info">
    <span class="code-badge">{{ row.chosen_code }}</span>
    <span class="chooser-badge">choix CodeChooser</span>
    <div class="sources">{{ row.chooser_explanation }}</div>
  </div>
  <form method="post" action="{{ url_for('judge', row_id=row.row_id) }}">
    <input type="hidden" name="correct_code" value="{{ row.chosen_code }}">
    <button type="submit" class="pick">Correct</button>
  </form>
</div>
{% endif %}

<form method="post" action="{{ url_for('judge', row_id=row.row_id) }}" class="other-row">
  <input type="text" name="correct_code" placeholder="Autre code...">
  <button type="submit">Valider</button>
</form>

<div class="nav-links">
  <a href="{{ url_for('review', row_id=prev_id) if prev_id else '#' }}"
     class="{{ '' if prev_id else 'disabled' }}">&larr; Précédent</a>
  <a href="{{ url_for('review', row_id=next_id) if next_id else '#' }}"
     class="{{ '' if next_id else 'disabled' }}">Passer &rarr;</a>
</div>
"""
)

METRICS_TEMPLATE = (
    STYLE
    + NAV
    + """
<h2>Métriques multi-méthode</h2>

<div class="stat-grid">
  <div class="stat"><span class="n">{{ m.n_reviewed }}/{{ m.n_total }}</span><span class="l">Jugés</span></div>
  <div class="stat"><span class="n">{{ "%.0f%%"|format(m.ground_truth_accuracy * 100) if m.ground_truth_accuracy is not none else "—" }}</span><span class="l">Vérité terrain</span></div>
  <div class="stat"><span class="n">{{ "%.0f%%"|format(m.chooser_accuracy * 100) if m.chooser_accuracy is not none else "—" }}</span><span class="l">CodeChooser ({{ m.chooser_n }})</span></div>
  {% for name, s in m.methods.items() %}
  <div class="stat"><span class="n">{{ "%.0f%%"|format(s.accuracy * 100) if s.accuracy is not none else "—" }}</span><span class="l">{{ s.label }} ({{ s.n }})</span></div>
  {% endfor %}
</div>

<p class="muted">
  Accuracy = part des lignes jugées où le code proposé (vérité terrain / choix
  CodeChooser / prédiction de la méthode) correspond au code marqué correct par
  l'humain. Le dénominateur de chaque méthode exclut les lignes où elle n'a
  produit aucun code (échec de classification).
</p>

<h3>Lignes jugées</h3>
<table>
  <tr><th>Activité</th><th>Vérité terrain</th><th>Choix CodeChooser</th><th>Code jugé correct</th><th></th></tr>
  {% for r in rows %}
  <tr>
    <td>{{ r.libelle[:60] }}{{ "…" if r.libelle|length > 60 else "" }}</td>
    <td><span class="code-badge">{{ r.ground_truth_code }}</span></td>
    <td>{% if r.chosen_code %}<span class="code-badge">{{ r.chosen_code }}</span>{% else %}—{% endif %}</td>
    <td><span class="code-badge">{{ r.correct_code }}</span></td>
    <td><a href="{{ url_for('review', row_id=r.row_id) }}">revoir</a></td>
  </tr>
  {% endfor %}
</table>
"""
)


class PrefixMiddleware:
    """Cf. human_review_app.py's PrefixMiddleware — same Onyxia reverse-proxy handling."""

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
    n: int | None,
    seed: int,
    url_prefix: str = "",
) -> Flask:
    app = Flask(__name__)
    if url_prefix:
        app.wsgi_app = PrefixMiddleware(app.wsgi_app, url_prefix)
    rows = load_rows(input_path, code_column, n, seed)
    rows_by_id = {r["row_id"]: r for r in rows}
    order = [r["row_id"] for r in rows]
    logger.info(f"Loaded {len(rows)} rows from {input_path}")

    def first_unreviewed() -> str | None:
        judgments = load_judgments(output_path)
        for row_id in order:
            if row_id not in judgments:
                return row_id
        return None

    @app.route("/")
    def index():
        next_id = first_unreviewed()
        if next_id is None:
            return redirect(url_for("metrics"))
        return redirect(url_for("review", row_id=next_id))

    @app.route("/review/<row_id>")
    def review(row_id):
        if row_id not in rows_by_id:
            return redirect(url_for("index"))
        row = rows_by_id[row_id]
        judgments = load_judgments(output_path)
        chosen_norm = normalize_code(row["chosen_code"])
        candidate_norms = {c["norm"] for c in row["candidates"]}
        notices = {c["norm"]: get_code_notice(c["code"]) for c in row["candidates"]}
        idx = order.index(row_id)
        existing = judgments.get(row_id)
        return render_template_string(
            REVIEW_TEMPLATE,
            row={**row, "chosen_norm": chosen_norm, "candidate_norms": candidate_norms},
            chosen_norm=chosen_norm,
            notices=notices,
            idx=idx + 1,
            total=len(order),
            n_reviewed=len(judgments),
            n_total=len(order),
            prev_id=order[idx - 1] if idx > 0 else None,
            next_id=order[idx + 1] if idx + 1 < len(order) else None,
            existing_code=existing["correct_code"] if existing else None,
        )

    @app.route("/judge/<row_id>", methods=["POST"])
    def judge(row_id):
        if row_id not in rows_by_id:
            return redirect(url_for("index"))
        correct_code = request.form.get("correct_code", "").strip()
        if not correct_code:
            return redirect(url_for("review", row_id=row_id))

        row = rows_by_id[row_id]
        append_judgment(
            output_path,
            {
                "row_id": row_id,
                "libelle": row["libelle"],
                "ground_truth_code": row["ground_truth_code"],
                "chosen_code": row["chosen_code"],
                "method_codes": row["method_codes"],
                "correct_code": correct_code,
                "reviewed_at": datetime.now(timezone.utc).isoformat(),
            },
        )

        idx = order.index(row_id)
        if idx + 1 < len(order):
            return redirect(url_for("review", row_id=order[idx + 1]))
        return redirect(url_for("index"))

    @app.route("/metrics")
    def metrics():
        judgments = load_judgments(output_path)
        m = compute_metrics(rows, judgments)
        reviewed_rows = [
            {**row, **judgments[row["row_id"]]} for row in rows if row["row_id"] in judgments
        ]
        return render_template_string(
            METRICS_TEMPLATE,
            m=m,
            rows=reviewed_rows,
            n_reviewed=m["n_reviewed"],
            n_total=m["n_total"],
        )

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
    parser.add_argument("--n", type=int, default=None, help="Subsample size (default: all rows)")
    parser.add_argument("--seed", type=int, default=42)
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

    if args.url_prefix is not None:
        url_prefix = args.url_prefix
    elif os.environ.get("VSCODE_PROXY_URI"):
        url_prefix = f"/proxy/{args.port}"
    else:
        url_prefix = ""

    app = create_app(
        args.input, args.output, args.code_column, args.n, args.seed, url_prefix=url_prefix
    )
    if url_prefix:
        logger.info(f"Prefixing generated links with {url_prefix!r} (Onyxia reverse proxy)")
    app.run(host=args.host, port=args.port, debug=args.debug)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
