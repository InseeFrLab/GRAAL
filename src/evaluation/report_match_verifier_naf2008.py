"""Rapport agrégé de l'évaluation MatchVerifier NAF 2008 (code de production vs humain).

Pendant de docs/eval/match_verifier_coicop.md pour la NAF : mêmes métriques (cf.
metrics.verifier_metrics), mêmes tableaux, avec ce qui change d'une nomenclature à
l'autre :

- la vérité « de branche » de la COICOP n'a pas d'objet ici — codes de production et
  codes humains sont tous des sous-classes — et cède la place à l'**accord de classe**
  (4 premiers caractères) : un écart de sous-classe (`68.20A` / `68.20B`) est-il une
  erreur que l'on veut attraper au même titre qu'un changement de section ?
- les types d'écart suivent les niveaux NAF (sous-classe, classe, groupe, division,
  section) ;
- le mode d'attribution du code de production (`mode_calcul_ape` : `AUTO` par règles,
  `FASTTEXT` par le modèle) découpe la population, les deux ayant des taux d'erreur
  très différents.

Lit le checkpoint au fil de l'eau : un run en cours donne un rapport partiel, dont
l'état est indiqué en tête. Aucun libellé n'est écrit (secret statistique).

Usage :
    uv run -m src.evaluation.report_match_verifier_naf2008 \
        --checkpoint data/eval/match_verifier_eval_naf2008/full/<commit>/<modèle>/match_verifier_eval.checkpoint.jsonl \
        --pairs data/eval/naf2008/pairs_test_NAF2008.parquet \
        --output docs/eval/match_verifier_naf2008.md \
        --site-json presentation/data/match_verifier_naf2008.json
"""

import argparse
import os
import json
import math
from datetime import datetime

import polars as pl
from dotenv import load_dotenv

from src.evaluation.build_eval_set import load_dataframe
from src.evaluation.match_verifier_eval import (
    SUMMARY_FILENAME,
    ALTERNATIVE_CODE_COLUMN,
    CODE_COLUMN_OUT,
    DURATION_COLUMN,
    EXPECTED_CODE_COLUMN,
    EXPECTED_MATCH_COLUMN,
    P_MATCH_COLUMN,
    P_MATCH_THRESHOLDS,
    TEXT_COLUMN_OUT,
    VERDICT_COLUMN,
)
from src.evaluation.metrics import is_within_branch, normalize_code, verifier_metrics

load_dotenv(override=True)

HIERARCHY_PATH = "projet-ape/data/naf2008_extended.parquet"

GAP_TYPES = [
    "Accord exact",
    "Même classe, sous-classe différente",
    "Même groupe, classe différente",
    "Même division, groupe différent",
    "Même section, division différente",
    "Section différente",
]


def gap_type(code: str, expected: str, section_of: dict[str, str]) -> str:
    code, expected = normalize_code(code), normalize_code(expected)
    if code == expected:
        return GAP_TYPES[0]
    for depth, label in ((4, GAP_TYPES[1]), (3, GAP_TYPES[2]), (2, GAP_TYPES[3])):
        if code[:depth] == expected[:depth]:
            return label
    if section_of.get(code) and section_of.get(code) == section_of.get(expected):
        return GAP_TYPES[4]
    return GAP_TYPES[5]


def load_results(checkpoint: str) -> pl.DataFrame:
    with open(checkpoint, encoding="utf-8") as f:
        rows = []
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # dernière ligne tronquée d'un run en cours
    return pl.DataFrame(rows, infer_schema_length=None).unique(
        [TEXT_COLUMN_OUT, CODE_COLUMN_OUT, EXPECTED_CODE_COLUMN], keep="last"
    )


def pct(x: float) -> str:
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.1%}"


def metrics_row(frame: pl.DataFrame, truth: str, prevalence: float) -> dict:
    m = verifier_metrics(
        frame[truth].to_list(),
        frame[VERDICT_COLUMN].to_list(),
        p_match=frame[P_MATCH_COLUMN].to_list(),
        error_prevalence=prevalence,
    )
    caught = frame.filter(~pl.col(truth) & ~pl.col(VERDICT_COLUMN))
    recovered = [
        is_within_branch(alt, exp)
        for alt, exp in zip(caught[ALTERNATIVE_CODE_COLUMN], caught[EXPECTED_CODE_COLUMN])
    ]
    m["alternative_recovers"] = sum(recovered) / len(recovered) if recovered else float("nan")
    return m


def prepare(
    results: pl.DataFrame, pairs_rows: pl.DataFrame, section_of: dict
) -> tuple[pl.DataFrame, pl.DataFrame, float, float, int]:
    """Joint les verdicts à la population ; rend (paires évaluées, lignes, prévalences
    exacte et de classe, nombre de paires distinctes)."""
    # Population (lignes) : prévalences d'erreur et mode majoritaire de chaque paire.
    rows = pairs_rows.select(
        pl.col("libelle").alias(TEXT_COLUMN_OUT),
        pl.col("apet_finale").alias(CODE_COLUMN_OUT),
        pl.col("apet_manual").alias(EXPECTED_CODE_COLUMN),
        "mode_calcul_ape",
    )
    prevalence_exact = float((rows[CODE_COLUMN_OUT] != rows[EXPECTED_CODE_COLUMN]).mean())
    prevalence_class = float(
        (rows[CODE_COLUMN_OUT].str.slice(0, 4) != rows[EXPECTED_CODE_COLUMN].str.slice(0, 4)).mean()
    )
    keys = [TEXT_COLUMN_OUT, CODE_COLUMN_OUT, EXPECTED_CODE_COLUMN]
    pair_mode = (
        rows.group_by(keys + ["mode_calcul_ape"])
        .len()
        # Égalité de lignes : AUTO avant FASTTEXT, pour un rapport reproductible
        # (l'ordre de sortie de group_by ne l'est pas).
        .sort(["len", "mode_calcul_ape"], descending=[True, False])
        .unique(keys, keep="first", maintain_order=True)
        .drop("len")
    )
    n_pairs_total = rows.unique(keys).height

    df = results.join(pair_mode, on=keys, how="left").with_columns(
        (
            pl.col(CODE_COLUMN_OUT).str.slice(0, 4) == pl.col(EXPECTED_CODE_COLUMN).str.slice(0, 4)
        ).alias("expected_match_class"),
        pl.struct(CODE_COLUMN_OUT, EXPECTED_CODE_COLUMN)
        .map_elements(
            lambda r: gap_type(r[CODE_COLUMN_OUT], r[EXPECTED_CODE_COLUMN], section_of),
            return_dtype=pl.Utf8,
        )
        .alias("gap_type"),
    )
    return df, rows, prevalence_exact, prevalence_class, n_pairs_total


def build_report(
    results: pl.DataFrame,
    pairs_rows: pl.DataFrame,
    section_of: dict,
    run: str = "",
    finished: bool = False,
) -> str:
    df, rows, prevalence_exact, prevalence_class, n_pairs_total = prepare(
        results, pairs_rows, section_of
    )
    n = df.height
    # Un run fini peut laisser quelques paires en échec (réponse vide, etc.) : c'est le
    # résumé écrit en fin de run, pas le compte, qui dit qu'il est terminé.
    done = "terminé" if finished or n >= n_pairs_total else "en cours"
    lines = [
        "# Évaluation du MatchVerifier sur la NAF 2008 (code de production vs annotation humaine)",
        "",
        f"**État : {done}** — {n} / {n_pairs_total} paires ({n / n_pairs_total:.1%}), "
        f"mis à jour le {datetime.now():%Y-%m-%d %H:%M}.",
        "",
        "## Protocole",
        "",
        "- Données : test de la campagne d'annotation 2024 "
        "(`label-studio/annotation-campaign-2024/NAF2008/preprocessed/test_data_NAF2008.parquet`), "
        "joint par liasse aux exports Label Studio pour retrouver le code attribué en production "
        "(cf. `src.evaluation.prepare_naf2008_pairs`).",
        "- Code jugé : `apet_finale`, code de production (`AUTO` par règles ou `FASTTEXT` par "
        "le modèle). Référence : `apet_manual`, code retenu par l'annotateur. Les deux en NAF rév. 2, "
        "au niveau sous-classe.",
        f"- Version : `{run}` (`<commit>/<modèle>` sous lequel match_verifier_eval a rangé "
        "le run, cf. `src.utils.run_provenance`).",
        "- Graphe : NAF rév. 2 construite depuis les notes explicatives Insee "
        "(cf. `graph_builder.utils.naf2008_notices`), profil `NOMENCLATURE=naf2008`.",
        "- Deux vérités : **exacte** (même sous-classe) et **de classe** (mêmes 4 premiers "
        "caractères : un écart de sous-classe n'y compte pas comme une erreur).",
        f"- Population : {rows.height} lignes, {n_pairs_total} paires distinctes ; tous les codes "
        f"sont dans le graphe. Taux d'erreur de la production : {pct(prevalence_exact)} (exact), "
        f"{pct(prevalence_class)} (classe).",
        "- Toutes les paires distinctes sont vérifiées (pas d'échantillonnage) ; métriques "
        "calculées par paire distincte (libellé, code de production, code humain).",
        "",
        "## Résultats principaux",
        "",
        "| Vérité | Paires | dont erreurs | Rappel des erreurs | Rejets à tort | Précision du rejet "
        "| Précision à la prévalence réelle | AUC p_match | Alternative = humain (erreurs détectées) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for label, truth, prev in (
        ("Exacte", EXPECTED_MATCH_COLUMN, prevalence_exact),
        ("Classe", "expected_match_class", prevalence_class),
    ):
        m = metrics_row(df, truth, prev)
        lines.append(
            f"| {label} | {m['n']} | {m['n_errors']} | {pct(m['error_recall'])} | "
            f"{pct(m['false_rejection_rate'])} | {pct(m['rejection_precision'])} | "
            f"{pct(m.get('rejection_precision_at_prevalence'))} | {m['roc_auc']:.3f} | "
            f"{pct(m['alternative_recovers'])} |"
        )
    lines += [
        "",
        "- **Rappel des erreurs** : part des erreurs de la production que le vérificateur rejette.",
        "- **Rejets à tort** : part des codes justes qu'il rejette.",
        "- **Précision à la prévalence réelle** : part des rejets qui portent sur de vraies "
        "erreurs, ramenée au taux d'erreur de la population.",
        "- **Alternative = humain** : quand il rejette à raison, son code alternatif est-il "
        "le code humain ?",
        "",
        "## Par type d'écart production / humain",
        "",
        "| Type | Paires | Rejetées par le vérificateur |",
        "|---|---|---|",
    ]
    for gap in GAP_TYPES:
        sub = df.filter(pl.col("gap_type") == gap)
        if sub.height:
            lines.append(f"| {gap} | {sub.height} | {pct(float((~sub[VERDICT_COLUMN]).mean()))} |")
    lines += [
        "",
        "Pour « Accord exact », le taux de rejet est un taux de rejet à tort ; pour les autres "
        "lignes, plus il est haut, mieux le vérificateur détecte ce type d'écart.",
        "",
        "## Par mode d'attribution du code de production (vérité exacte)",
        "",
        "| Mode | Paires | Taux d'erreur de la production | Rappel des erreurs | Rejets à tort "
        "| AUC p_match |",
        "|---|---|---|---|---|---|",
    ]
    for mode in ("AUTO", "FASTTEXT"):
        sub = df.filter(pl.col("mode_calcul_ape") == mode)
        if not sub.height:
            continue
        mode_rows = rows.filter(pl.col("mode_calcul_ape") == mode)
        prev = float((mode_rows[CODE_COLUMN_OUT] != mode_rows[EXPECTED_CODE_COLUMN]).mean())
        m = metrics_row(sub, EXPECTED_MATCH_COLUMN, prev)
        lines.append(
            f"| {mode} | {m['n']} | {pct(prev)} | {pct(m['error_recall'])} | "
            f"{pct(m['false_rejection_rate'])} | {m['roc_auc']:.3f} |"
        )
    lines += [
        "",
        "## Point de fonctionnement (vérité exacte)",
        "",
        "Le verdict `is_match` est un seuil parmi d'autres ; en seuillant `p_match` :",
        "",
        "| Seuil (rejet si p_match < t) | Rappel des erreurs | Rejets à tort |",
        "|---|---|---|",
    ]
    scored = df.filter(pl.col(P_MATCH_COLUMN).is_not_null())
    errors = scored.filter(~pl.col(EXPECTED_MATCH_COLUMN))
    correct = scored.filter(pl.col(EXPECTED_MATCH_COLUMN))
    for t in P_MATCH_THRESHOLDS:
        recall = float((errors[P_MATCH_COLUMN] < t).mean()) if errors.height else float("nan")
        frr = float((correct[P_MATCH_COLUMN] < t).mean()) if correct.height else float("nan")
        lines.append(f"| {t} | {pct(recall)} | {pct(frr)} |")
    extreme = float(((scored[P_MATCH_COLUMN] < 0.1) | (scored[P_MATCH_COLUMN] > 0.9)).mean())
    lines += [
        "",
        f"{pct(extreme)} des `p_match` sont sous 0,1 ou au-dessus de 0,9 : le modèle tranche "
        "presque toujours avec certitude, et seuiller `p_match` ne déplace guère le point de "
        "fonctionnement. L'AUC n'en mesure donc qu'une version quasi binaire du verdict.",
        "",
        "_Version agrégée, sans aucun libellé (données soumises au secret statistique)._",
        "",
        "## Coût",
        "",
        f"Durée moyenne par paire : {df[DURATION_COLUMN].mean():.1f} s ; temps d'inférence "
        f"cumulé : {df[DURATION_COLUMN].sum() / 3600:.1f} h.",
        "",
    ]
    return "\n".join(lines)


def _num(x: float) -> float | None:
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else x


def build_site_data(
    results: pl.DataFrame,
    pairs_rows: pl.DataFrame,
    section_of: dict,
    run: str = "",
    finished: bool = False,
) -> dict:
    """Agrégats lus par presentation/match_verifier_naf2008.qmd (aucun libellé)."""
    df, rows, prevalence_exact, prevalence_class, n_pairs_total = prepare(
        results, pairs_rows, section_of
    )
    n = df.height
    truths = []
    for key, name, truth, prev in (
        ("exact", "Exacte", EXPECTED_MATCH_COLUMN, prevalence_exact),
        ("class", "Classe", "expected_match_class", prevalence_class),
    ):
        m = metrics_row(df, truth, prev)
        truths.append(
            {
                "key": key,
                "name": name,
                "n": m["n"],
                "n_errors": m["n_errors"],
                **{
                    k: _num(m.get(k))
                    for k in (
                        "error_recall",
                        "false_rejection_rate",
                        "rejection_precision",
                        "rejection_precision_at_prevalence",
                        "roc_auc",
                    )
                },
                "alt_recovers": _num(m["alternative_recovers"]),
            }
        )
    by_type = [
        {
            "type": gap,
            "n": sub.height,
            "is_error": gap != GAP_TYPES[0],
            "rejection_rate": float((~sub[VERDICT_COLUMN]).mean()),
        }
        for gap in GAP_TYPES
        if (sub := df.filter(pl.col("gap_type") == gap)).height
    ]
    by_mode = []
    for mode in ("AUTO", "FASTTEXT"):
        sub = df.filter(pl.col("mode_calcul_ape") == mode)
        if not sub.height:
            continue
        mode_rows = rows.filter(pl.col("mode_calcul_ape") == mode)
        prev = float((mode_rows[CODE_COLUMN_OUT] != mode_rows[EXPECTED_CODE_COLUMN]).mean())
        m = metrics_row(sub, EXPECTED_MATCH_COLUMN, prev)
        by_mode.append(
            {
                "mode": mode,
                "n_rows": mode_rows.height,
                "n": m["n"],
                "error_prevalence": prev,
                "flag_rate": float((~sub[VERDICT_COLUMN]).mean()),
                **{
                    k: _num(m.get(k))
                    for k in (
                        "error_recall",
                        "false_rejection_rate",
                        "rejection_precision_at_prevalence",
                        "roc_auc",
                    )
                },
            }
        )

    # Courbe rappel / rejets à tort en seuillant p_match (vérité exacte).
    scored = df.filter(pl.col(P_MATCH_COLUMN).is_not_null())
    is_error = ~scored[EXPECTED_MATCH_COLUMN]
    p = scored[P_MATCH_COLUMN]
    n_err, n_ok = int(is_error.sum()), int((~is_error).sum())
    curve = [{"threshold": 0.0, "recall": 0.0, "false_rejection": 0.0}]
    for t in sorted(set(p.to_list())):
        flagged = p <= t
        point = {
            "threshold": t,
            "recall": float((flagged & is_error).sum()) / n_err,
            "false_rejection": float((flagged & ~is_error).sum()) / n_ok,
        }
        # p_match est quasi continu près de 0 et 1 : un point par pas de 0,5 % suffit.
        last = curve[-1]
        if max(
            point["recall"] - last["recall"], point["false_rejection"] - last["false_rejection"]
        ) >= 0.005 or t == max(p):
            curve.append(point)
    exact = truths[0]
    hist = []
    bins = [i / 10 for i in range(11)]
    for cls, mask in (("Erreur de la production", is_error), ("Code juste", ~is_error)):
        values = p.filter(mask)
        for lo, hi in zip(bins[:-1], bins[1:]):
            count = int(((values >= lo) & ((values < hi) | (hi == 1.0))).sum())
            hist.append(
                {"class": cls, "lo": lo, "hi": hi, "share": count / len(values), "count": count}
            )
    return {
        "updated": f"{datetime.now():%Y-%m-%d %H:%M}",
        "state": "terminé" if finished or n >= n_pairs_total else "en cours",
        "run": run,
        "n_done": n,
        "n_total": n_pairs_total,
        "population": {
            "n_rows": rows.height,
            "n_pairs": n_pairs_total,
            "error_prevalence": prevalence_exact,
            "error_prevalence_class": prevalence_class,
        },
        "truths": truths,
        "by_type": by_type,
        "by_mode": by_mode,
        "curve": curve,
        "verdict_point": {
            "recall": exact["error_recall"],
            "false_rejection": exact["false_rejection_rate"],
        },
        "p_match_hist": hist,
        "p_match_extreme": float(((p < 0.1) | (p > 0.9)).mean()),
        "mean_duration_s": float(df[DURATION_COLUMN].mean()),
        "total_duration_h": float(df[DURATION_COLUMN].sum() / 3600),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True, help="match_verifier_eval checkpoint .jsonl")
    parser.add_argument("--pairs", required=True, help="Sortie de prepare_naf2008_pairs")
    parser.add_argument("--hierarchy", default=HIERARCHY_PATH)
    parser.add_argument("--output", required=True, help="Rapport markdown")
    parser.add_argument(
        "--site-json",
        help="Agrégats pour la page du site (presentation/data/match_verifier_naf2008.json)",
    )
    args = parser.parse_args()

    hierarchy = load_dataframe(args.hierarchy)
    section_of = dict(zip(hierarchy["APE_NIV5"], hierarchy["APE_NIV1"]))
    # Le checkpoint vit sous <output>/<commit>/<modèle>/ : ses deux dossiers parents
    # identifient la version du prompt et le modèle jugés.
    run = "/".join(os.path.normpath(args.checkpoint).split(os.sep)[-3:-1])
    results, pairs_rows = load_results(args.checkpoint), load_dataframe(args.pairs)
    finished = os.path.exists(os.path.join(os.path.dirname(args.checkpoint), SUMMARY_FILENAME))
    report = build_report(results, pairs_rows, section_of, run, finished=finished)
    if args.site_json:
        site = build_site_data(results, pairs_rows, section_of, run, finished=finished)
        with open(args.site_json, "w", encoding="utf-8") as f:
            json.dump(site, f, ensure_ascii=False, indent=1)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(report)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
