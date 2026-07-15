"""Petite appli web pour l'évaluation humaine du MatchVerifier.

Pour un échantillon de labels du jeu d'entraînement déjà jugés par
MatchVerifier (cf. src.evaluation.verify_train_labels), affiche l'activité,
le code de la vérité terrain avec sa notice officielle NACE, et le verdict
du vérificateur (is_match, confiance, justification). L'humain indique si ce
verdict est correct ("right") ou non ("wrong") ; les jugements sont
journalisés en JSONL (append-only, ré-exécutable) et servent à calculer
l'accuracy, la précision et le rappel du vérificateur.

Le rappel/précision sont dérivés en traitant le jugement humain comme la
vérité terrain sur le match réel : si l'humain dit "right", le label réel vaut
le verdict du vérificateur ; s'il dit "wrong", le label réel est l'inverse.

Quand le vérificateur affirme un mismatch (llm_is_match = False), la revue
affiche en plus, pour aider à juger, la proposition indépendante du
SummaryAgenticClassifier pour ce même libellé, avec la notice officielle du
code qu'il propose. Cette proposition n'est pas calculée ici : elle est déjà
présente dans le parquet d'entrée (colonnes summary_code/summary_confidence/
summary_explanation), calculée une fois pour toutes par
src.evaluation.verify_train_labels au moment même où celui-ci appelle
MatchVerifier — pas d'appel agentique dans cette appli, juste une lecture de
colonnes (absentes ou nulles : pas de proposition affichée, sans erreur).

Nécessite Neo4j configuré dans l'environnement pour afficher les notices (du
code de vérité terrain et du code proposé, comme src.main) ; sans connexion,
la revue reste possible mais sans notice.

Usage :
    uv run -m src.evaluation.human_review_app \
        --input data/eval/train_verification/train_label_verification.parquet \
        --output data/eval/human_review/match_verifier_review.jsonl \
        --port 5050
"""

import argparse
import hashlib
import json
import logging
import os
from datetime import datetime, timezone

import polars as pl
from flask import Flask, redirect, render_template_string, url_for

from src.agents.closers.match_verifier import MatchVerificationInput
from src.config import neo4j_config
from src.neo4j_graph.graph import Graph
from src.utils import storage
from src.utils.logging import configure_logging

configure_logging()
logger = logging.getLogger(__name__)

DEFAULT_INPUT = "data/eval/train_verification/train_label_verification.parquet"
DEFAULT_OUTPUT = "data/eval/human_review/match_verifier_review.jsonl"


def row_id_for(libelle: str, code: str) -> str:
    return hashlib.sha1(f"{libelle}|{code}".encode("utf-8")).hexdigest()[:12]


def load_rows(input_path: str, n: int | None, seed: int) -> list[dict]:
    with storage.open_path(input_path, "rb") as f:
        df = pl.read_parquet(f)
    if n is not None and n < len(df):
        df = df.sample(n=n, seed=seed)
    # summary_code/summary_confidence/summary_explanation (SummaryAgenticClassifier's
    # second opinion, precomputed by verify_train_labels for mismatch rows) are
    # optional: absent on a parquet produced before that column was added.
    has_summary_columns = "summary_code" in df.columns
    return [
        {
            "row_id": row_id_for(r["libelle"], r["nace2025"]),
            "libelle": r["libelle"],
            "nace2025": r["nace2025"],
            "llm_is_match": bool(r["llm_is_match"]),
            "llm_confidence": r["llm_confidence"],
            "llm_explanation": r["llm_explanation"],
            "summary_code": r["summary_code"] if has_summary_columns else None,
            "summary_confidence": r["summary_confidence"] if has_summary_columns else None,
            "summary_explanation": r["summary_explanation"] if has_summary_columns else None,
        }
        for r in df.to_dicts()
    ]


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
    """Confusion matrix of llm_is_match (predicted) vs. human-derived actual label.

    actual = predicted if the human marked the verdict "right", else `not predicted`
    (the human is asserting the verifier got it backwards).
    """
    tp = fp = fn = tn = 0
    n_right = 0
    n_reviewed = 0
    for row in rows:
        judgment = judgments.get(row["row_id"])
        if judgment is None:
            continue
        n_reviewed += 1
        predicted = row["llm_is_match"]
        is_right = judgment["human_verdict"] == "right"
        n_right += int(is_right)
        actual = predicted if is_right else not predicted
        if predicted and actual:
            tp += 1
        elif predicted and not actual:
            fp += 1
        elif not predicted and actual:
            fn += 1
        else:
            tn += 1

    accuracy = n_right / n_reviewed if n_reviewed else None
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = 2 * precision * recall / (precision + recall) if precision and recall else None

    return {
        "n_total": len(rows),
        "n_reviewed": n_reviewed,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
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


def summary_prediction_for(row: dict) -> MatchVerificationInput | None:
    """Reconstructs the SummaryAgenticClassifier second opinion from the row's
    summary_code/summary_confidence/summary_explanation columns (precomputed by
    verify_train_labels, cf. module docstring). Returns None when the row has no
    proposal: a match row (never computed), an older parquet without the columns, or
    a mismatch row where the classifier call failed at verification time.
    """
    code = row.get("summary_code")
    if not code:
        return None
    return MatchVerificationInput(
        activity=row["libelle"],
        code=code,
        proposed_explanation=row.get("summary_explanation"),
        proposed_confidence=row.get("summary_confidence"),
    )


STYLE = """
<style>
  :root { color-scheme: light dark; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    max-width: 780px; margin: 2rem auto; padding: 0 1.25rem;
    line-height: 1.5; color: #1a1a1a; background: #fff;
  }
  @media (prefers-color-scheme: dark) {
    body { color: #e8e8e8; background: #1b1b1b; }
    .card { background: #262626 !important; border-color: #3a3a3a !important; }
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
  .code-badge { display: inline-block; font-family: monospace; font-size: 1.1rem;
    background: #eef2ff; padding: 0.15rem 0.5rem; border-radius: 6px; font-weight: 700; }
  @media (prefers-color-scheme: dark) { .code-badge { background: #2a3350; color: #cfd9ff; } }
  .notice { white-space: pre-line; font-size: 0.95rem; }
  .verdict-row { display: flex; align-items: center; gap: 0.6rem; margin-bottom: 0.5rem; }
  .verdict-badge { padding: 0.15rem 0.6rem; border-radius: 999px; font-weight: 700; font-size: 0.85rem; }
  .verdict-match { background: #d7f5df; color: #166534; }
  .verdict-nomatch { background: #fde2e2; color: #991b1b; }
  @media (prefers-color-scheme: dark) {
    .verdict-match { background: #14351f; color: #7be0a0; }
    .verdict-nomatch { background: #3a1414; color: #ff9d9d; }
  }
  .muted { color: #777; font-size: 0.9rem; }
  .actions { display: flex; gap: 0.75rem; margin: 1.5rem 0; }
  .actions a { font-size: 1rem; font-weight: 700; padding: 0.7rem 1.4rem; border-radius: 8px;
    border: none; cursor: pointer; flex: 1; text-align: center; text-decoration: none; display: block; }
  .btn-right { background: #16a34a; color: white; }
  .btn-wrong { background: #dc2626; color: white; }
  .actions a:hover { opacity: 0.88; }
  .existing-banner { padding: 0.5rem 0.9rem; border-radius: 8px; margin-bottom: 1rem; font-weight: 600; }
  .existing-right { background: #d7f5df; color: #166534; }
  .existing-wrong { background: #fde2e2; color: #991b1b; }
  @media (prefers-color-scheme: dark) {
    .existing-right { background: #14351f; color: #7be0a0; }
    .existing-wrong { background: #3a1414; color: #ff9d9d; }
  }
  .nav-links { display: flex; justify-content: space-between; }
  .nav-links a.disabled { pointer-events: none; color: #bbb; }
  table { border-collapse: collapse; width: 100%; margin: 1rem 0; }
  th, td { text-align: left; padding: 0.4rem 0.6rem; border-bottom: 1px solid #e5e5e5; font-size: 0.9rem; }
  @media (prefers-color-scheme: dark) { th, td { border-color: #3a3a3a; } }
  .stat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 0.75rem; margin: 1.25rem 0; }
  .stat { border: 1px solid #ddd; border-radius: 10px; padding: 0.8rem; text-align: center; background: #fafafa; }
  .stat .n { font-size: 1.6rem; font-weight: 700; display: block; }
  .stat .l { font-size: 0.78rem; color: #777; text-transform: uppercase; letter-spacing: 0.03em; }
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
{% if existing_verdict %}
  <div class="existing-banner existing-{{ existing_verdict }}">
    Déjà jugé : verdict du vérificateur marqué <strong>{{ "correct" if existing_verdict == "right" else "incorrect" }}</strong>
  </div>
{% endif %}

<div class="card">
  <div class="label">Activité ({{ idx }} / {{ total }})</div>
  <div class="activite">{{ row.libelle }}</div>

  <div class="label">Code de la vérité terrain</div>
  <div style="margin-bottom: 1rem;">
    <span class="code-badge">{{ row.nace2025 }}</span>
    {% if notice %} &mdash; {{ notice.name }}{% endif %}
  </div>

  {% if notice %}
    <div class="label">Notice officielle NACE</div>
    <div class="notice">{{ notice.description or "(pas de description)" }}</div>
  {% else %}
    <div class="muted">Notice indisponible (pas de connexion Neo4j ou code introuvable).</div>
  {% endif %}
</div>

<div class="card">
  <div class="verdict-row">
    {% if row.llm_is_match %}
      <span class="verdict-badge verdict-match">MATCH</span>
    {% else %}
      <span class="verdict-badge verdict-nomatch">NO MATCH</span>
    {% endif %}
    <span class="muted">confiance : {{ "%.0f"|format(row.llm_confidence * 100) }}%</span>
  </div>
  <div class="label">Justification du MatchVerifier</div>
  <div>{{ row.llm_explanation }}</div>
</div>

{% if not row.llm_is_match %}
<div class="card">
  <div class="label">Proposition indépendante (Summary Agentic Classifier)</div>
  {% if summary_prediction %}
    <div style="margin-bottom: 1rem;">
      <span class="code-badge">{{ summary_prediction.code }}</span>
      {% if summary_code_notice %} &mdash; {{ summary_code_notice.name }}{% endif %}
      {% if summary_prediction.proposed_confidence is not none %}
        <span class="muted"> (confiance : {{ "%.0f"|format(summary_prediction.proposed_confidence * 100) }}%)</span>
      {% endif %}
    </div>
    <div>{{ summary_prediction.proposed_explanation }}</div>
    {% if summary_code_notice %}
      <div class="label" style="margin-top: 1rem;">Notice officielle de ce code</div>
      <div class="notice">{{ summary_code_notice.description or "(pas de description)" }}</div>
    {% endif %}
  {% else %}
    <div class="muted">Proposition indisponible (non calculée pour ce libellé, cf. verify_train_labels).</div>
  {% endif %}
</div>
{% endif %}

<div class="actions">
  <a href="{{ url_for('judge', row_id=row.row_id, verdict='right') }}" class="btn-right">&#10003; Verdict correct</a>
  <a href="{{ url_for('judge', row_id=row.row_id, verdict='wrong') }}" class="btn-wrong">&#10007; Verdict incorrect</a>
</div>

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
<h2>Métriques du MatchVerifier</h2>

<div class="stat-grid">
  <div class="stat"><span class="n">{{ m.n_reviewed }}/{{ m.n_total }}</span><span class="l">Jugés</span></div>
  <div class="stat"><span class="n">{{ "%.0f%%"|format(m.accuracy * 100) if m.accuracy is not none else "—" }}</span><span class="l">Accuracy</span></div>
  <div class="stat"><span class="n">{{ "%.0f%%"|format(m.precision * 100) if m.precision is not none else "—" }}</span><span class="l">Précision</span></div>
  <div class="stat"><span class="n">{{ "%.0f%%"|format(m.recall * 100) if m.recall is not none else "—" }}</span><span class="l">Rappel</span></div>
  <div class="stat"><span class="n">{{ "%.0f%%"|format(m.f1 * 100) if m.f1 is not none else "—" }}</span><span class="l">F1</span></div>
</div>

<p class="muted">
  Accuracy = part des verdicts jugés corrects. Précision/rappel traitent
  "is_match" comme prédiction positive, avec un label réel dérivé du jugement
  humain (verdict inversé quand il est jugé incorrect).
</p>

<table>
  <tr><th></th><th>Réel : match</th><th>Réel : no match</th></tr>
  <tr><th>Prédit : match</th><td>TP = {{ m.tp }}</td><td>FP = {{ m.fp }}</td></tr>
  <tr><th>Prédit : no match</th><td>FN = {{ m.fn }}</td><td>TN = {{ m.tn }}</td></tr>
</table>

<h3>Lignes jugées</h3>
<table>
  <tr><th>Activité</th><th>Code</th><th>Verdict LLM</th><th>Jugement humain</th><th></th></tr>
  {% for r in rows %}
  <tr>
    <td>{{ r.libelle[:60] }}{{ "…" if r.libelle|length > 60 else "" }}</td>
    <td><span class="code-badge">{{ r.nace2025 }}</span></td>
    <td>{{ "MATCH" if r.llm_is_match else "NO MATCH" }}</td>
    <td>{{ "correct" if r.human_verdict == "right" else "incorrect" }}</td>
    <td><a href="{{ url_for('review', row_id=r.row_id) }}">revoir</a></td>
  </tr>
  {% endfor %}
</table>
"""
)


class PrefixMiddleware:
    """Make the app work whether or not the reverse proxy strips its own path prefix.

    Onyxia/code-server serves this app through a path-based reverse proxy
    (`/proxy/<port>/...`). Whether that proxy strips the prefix before
    forwarding to us, or forwards the full path unstripped, is not something
    we can rely on: it 404s one way and 404s the other way if we hard-code
    either assumption. So: if an incoming request's PATH_INFO still carries
    the prefix, strip it ourselves before Werkzeug routes it (handles the
    "forwarded unstripped" case; a no-op if the proxy already stripped it).
    Either way, set SCRIPT_NAME so url_for()/redirect() re-add the prefix to
    every generated link and Location header, since plain `<a href>` links in
    the HTML body are never rewritten by the proxy itself.
    """

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
    input_path: str, output_path: str, n: int | None, seed: int, url_prefix: str = ""
) -> Flask:
    app = Flask(__name__)
    if url_prefix:
        app.wsgi_app = PrefixMiddleware(app.wsgi_app, url_prefix)
    rows = load_rows(input_path, n, seed)
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
        notice = get_code_notice(row["nace2025"])
        summary_prediction = summary_prediction_for(row)
        summary_code_notice = (
            get_code_notice(summary_prediction.code) if summary_prediction is not None else None
        )
        idx = order.index(row_id)
        existing = judgments.get(row_id)
        return render_template_string(
            REVIEW_TEMPLATE,
            row=row,
            notice=notice,
            summary_prediction=summary_prediction,
            summary_code_notice=summary_code_notice,
            idx=idx + 1,
            total=len(order),
            n_reviewed=len(judgments),
            n_total=len(order),
            prev_id=order[idx - 1] if idx > 0 else None,
            next_id=order[idx + 1] if idx + 1 < len(order) else None,
            existing_verdict=existing["human_verdict"] if existing else None,
        )

    @app.route("/judge/<row_id>/<verdict>")
    def judge(row_id, verdict):
        if row_id not in rows_by_id or verdict not in ("right", "wrong"):
            return redirect(url_for("index"))

        row = rows_by_id[row_id]
        append_judgment(
            output_path,
            {
                "row_id": row_id,
                "libelle": row["libelle"],
                "nace2025": row["nace2025"],
                "llm_is_match": row["llm_is_match"],
                "llm_confidence": row["llm_confidence"],
                "llm_explanation": row["llm_explanation"],
                "human_verdict": verdict,
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
    parser = argparse.ArgumentParser(description="Human review app for MatchVerifier")
    parser.add_argument(
        "--input", default=DEFAULT_INPUT, help="Parquet with libelle/nace2025/llm_* columns"
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="JSONL log of human judgments")
    parser.add_argument("--n", type=int, default=None, help="Subsample size (default: all rows)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5050)
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

    app = create_app(args.input, args.output, args.n, args.seed, url_prefix=url_prefix)
    if url_prefix:
        logger.info(f"Prefixing generated links with {url_prefix!r} (Onyxia reverse proxy)")
    app.run(host=args.host, port=args.port, debug=args.debug)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
