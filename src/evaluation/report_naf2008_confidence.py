"""Erreurs du MatchVerifier NAF 2008 selon l'indice de confiance du modèle fastText.

Question (note technique, chapitre 6) : le MatchVerifier est-il meilleur là où le
modèle de production est peu confiant, et apporte-t-il un signal que l'indice de
confiance (IC) n'a pas déjà ?

Source de l'IC : le tableau de bord de la campagne d'annotation 2024
(`dashboard/current-model`), qui contient pour chaque liasse du jeu de test les
5 meilleures prédictions d'un modèle fastText et son IC (`Response.IC`). C'est une
**re-prédiction** du modèle courant au moment de la campagne, pas nécessairement
l'IC qui a décidé du codage en production : sur les liasses `FASTTEXT`, son top-1
ne coïncide avec `apet_finale` que dans environ 64 % des cas.

L'analyse est faite **par liasse** (l'IC est propre à chaque liasse) : chaque
liasse reçoit le verdict du MatchVerifier sur sa paire (libellé, code de
production, code humain). Les intervalles de confiance sont obtenus par bootstrap
en grappes de paires (les liasses d'une même paire partagent leur verdict).

Aucun libellé n'est écrit (secret statistique).

Usage :
    uv run -m src.evaluation.report_naf2008_confidence \
        --results data/eval/match_verifier_eval_naf2008/full/<commit>/<modèle>/match_verifier_eval.parquet \
        --pairs data/eval/naf2008/pairs_test_NAF2008.parquet \
        --output docs/eval/match_verifier_naf2008_confidence.md
"""

import argparse
import logging

import numpy as np
import polars as pl

from src.evaluation.build_eval_set import load_dataframe
from src.evaluation.metrics import roc_auc
from src.utils.logging import configure_logging
from src.utils.storage import get_file_system

configure_logging()
logger = logging.getLogger(__name__)

DASHBOARD_PATH = "projet-ape/label-studio/annotation-campaign-2024/NAF2008/dashboard/current-model"
IC_BINS = [0.2, 0.4, 0.6, 0.8]
IC_LOW = 0.4
N_RESAMPLES = 1000
SEED = 42


def load_dashboard(path: str) -> pl.DataFrame:
    """(liasse_numero, ic, top1) du modèle courant ; une ligne par liasse."""
    fs = get_file_system()
    files = sorted(p for p in fs.find(path) if p.endswith(".parquet"))
    logger.info(f"Reading {len(files)} dashboard partitions under {path}")
    frames = []
    for p in files:
        with fs.open(p, "rb") as f:
            frames.append(
                pl.read_parquet(
                    f, columns=["liasse_numero", "Response.IC", "Response.1.code"]
                ).cast({"Response.1.code": pl.Utf8})
            )
    # Quelques liasses figurent dans plusieurs partitions : on garde la première.
    return (
        pl.concat(frames)
        .unique("liasse_numero", keep="first", maintain_order=True)
        .rename({"Response.IC": "ic", "Response.1.code": "top1"})
    )


def build_rows(results: pl.DataFrame, pairs: pl.DataFrame, dashboard: pl.DataFrame) -> pl.DataFrame:
    verdicts = results.select(
        pl.col("libelle"),
        pl.col("current_code").alias("apet_finale"),
        pl.col("expected_code").alias("apet_manual"),
        pl.col("match_verifier_verdict").alias("verdict"),
        (pl.col("match_verifier_alternative_score") - pl.col("match_verifier_match_score"))
        .cast(pl.Float64)
        .alias("gap"),
    )
    keys = ["libelle", "apet_finale", "apet_manual"]
    return (
        pairs.join(dashboard, on="liasse_numero", how="inner")
        .join(verdicts, on=keys, how="inner")
        .with_columns(
            err=pl.col("apet_finale") != pl.col("apet_manual"),
            rej=~pl.col("verdict"),
            pair=pl.struct(keys).hash(seed=0),
        )
    )


def rates(err: np.ndarray, flag: np.ndarray) -> dict:
    n_err, n_ok, n_flag = err.sum(), (~err).sum(), flag.sum()
    nan = float("nan")
    return {
        "n": len(err),
        "err": err.mean() if len(err) else nan,
        "flag": flag.mean() if len(err) else nan,
        "recall": (flag & err).sum() / n_err if n_err else nan,
        "fr": (flag & ~err).sum() / n_ok if n_ok else nan,
        "prec": (flag & err).sum() / n_flag if n_flag else nan,
    }


def _rank(x: np.ndarray) -> np.ndarray:
    return pl.Series(x).rank("average").to_numpy() / len(x)


def scores(frame: pl.DataFrame) -> dict[str, np.ndarray]:
    """Scores d'erreur (plus haut = plus suspect)."""
    ic = -frame["ic"].to_numpy()
    mv = frame["gap"].to_numpy()
    return {"ic": ic, "mv": mv, "comb": _rank(ic) + _rank(mv)}


def auc_with_ci(frame: pl.DataFrame) -> dict:
    """AUC de chaque score et différences appariées, IC 95 % par bootstrap en grappes de paires."""
    err = frame["err"].to_numpy()
    s = scores(frame)
    point = {k: roc_auc(err.tolist(), v.tolist()) for k, v in s.items()}
    groups = [
        np.array(idx)
        for idx in frame.with_row_index().group_by("pair").agg("index")["index"].to_list()
    ]
    rng = np.random.default_rng(SEED)
    draws = []
    for _ in range(N_RESAMPLES):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        draws.append({k: roc_auc(err[idx].tolist(), v[idx].tolist()) for k, v in s.items()})
    d = {k: np.array([x[k] for x in draws]) for k in s}

    def ci(x):
        return float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))

    return {
        **{k: (point[k], ci(d[k])) for k in s},
        "mv-ic": (point["mv"] - point["ic"], ci(d["mv"] - d["ic"])),
        "comb-mv": (point["comb"] - point["mv"], ci(d["comb"] - d["mv"])),
        "rank_corr": float(np.corrcoef(_rank(s["ic"]), _rank(s["mv"]))[0, 1]),
    }


def pct(x: float) -> str:
    return "–" if x is None or np.isnan(x) else f"{x:.1%}".replace(".", ",")


def num(x: float, d: int = 2) -> str:
    return f"{x:.{d}f}".replace(".", ",")


def signed(x: float) -> str:
    return f"{x:+.3f}".replace(".", ",")


def auc_cell(v) -> str:
    p, (lo, hi) = v
    return f"{num(p)} [{num(lo)} ; {num(hi)}]"


def build_report(rows: pl.DataFrame, n_test: int, run: str) -> str:
    lines = [
        "# MatchVerifier NAF 2008 : erreurs selon l'indice de confiance fastText",
        "",
        f"- Run MatchVerifier : `{run}`. Liasses analysées : {rows.height} sur {n_test} "
        "(jointure jeu de test × tableau de bord × verdicts).",
        f"- IC : `{DASHBOARD_PATH}` (`Response.IC`), re-prédiction du modèle courant au moment "
        "de la campagne. Analyse par liasse ; IC 95 % par bootstrap en grappes de paires "
        f"({N_RESAMPLES} tirages).",
        "- Score MatchVerifier : `alternative_score − match_score`. Score combiné : somme des rangs "
        "des deux scores.",
        "",
        "## Concordance du modèle du tableau de bord avec la production",
        "",
        "| Mode | Liasses | IC médian | Top-1 = `apet_finale` | Erreur de production si top-1 = `apet_finale` | si top-1 ≠ `apet_finale` |",
        "|---|---|---|---|---|---|",
    ]
    for mode in ["AUTO", "FASTTEXT"]:
        f = rows.filter(pl.col("mode_calcul_ape") == mode)
        agree = (f["top1"] == f["apet_finale"]).to_numpy()
        err = f["err"].to_numpy()
        lines.append(
            f"| `{mode}` | {f.height} | {num(float(f['ic'].median()))} | {pct(agree.mean())} "
            f"| {pct(err[agree].mean())} | {pct(err[~agree].mean())} |"
        )

    for mode in ["FASTTEXT", "AUTO"]:
        f = rows.filter(pl.col("mode_calcul_ape") == mode)
        err, rej, ic = f["err"].to_numpy(), f["rej"].to_numpy(), f["ic"].to_numpy()
        lines += [
            "",
            f"## Mode `{mode}`",
            "",
            "### Par tranche d'IC",
            "",
            "| IC | Liasses | Erreurs de production | Rejets | Erreurs détectées | Rejets à tort | Précision |",
            "|---|---|---|---|---|---|---|",
        ]
        edges = [0.0] + IC_BINS + [1.0001]
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (ic >= lo) & (ic < hi)
            if not m.any():
                continue
            r = rates(err[m], rej[m])
            label = f"[{num(lo, 1)} ; {num(min(hi, 1.0), 1)}{']' if hi > 1 else '['}"
            lines.append(
                f"| {label} | {r['n']} | {pct(r['err'])} | {pct(r['flag'])} | {pct(r['recall'])} "
                f"| {pct(r['fr'])} | {pct(r['prec'])} |"
            )
        r = rates(err, rej)
        lines.append(
            f"| **Ensemble** | **{r['n']}** | **{pct(r['err'])}** | **{pct(r['flag'])}** "
            f"| **{pct(r['recall'])}** | **{pct(r['fr'])}** | **{pct(r['prec'])}** |"
        )

        a = auc_with_ci(f)
        lines += [
            "",
            "### Pouvoir de détection des erreurs de production (AUC)",
            "",
            "| Score | AUC [IC 95 %] |",
            "|---|---|",
            f"| IC fastText (plus bas = plus suspect) | {auc_cell(a['ic'])} |",
            f"| MatchVerifier | {auc_cell(a['mv'])} |",
            f"| Combiné | {auc_cell(a['comb'])} |",
            f"| Différence MatchVerifier − IC | {signed(a['mv-ic'][0])} [{signed(a['mv-ic'][1][0])} ; {signed(a['mv-ic'][1][1])}] |",
            f"| Différence combiné − MatchVerifier | {signed(a['comb-mv'][0])} [{signed(a['comb-mv'][1][0])} ; {signed(a['comb-mv'][1][1])}] |",
            "",
            f"Corrélation des rangs des deux scores : {num(a['rank_corr'])}.",
            "",
            "### Signaux comparés",
            "",
            "| Signal | Liasses signalées | Erreurs détectées | Rejets à tort | Précision |",
            "|---|---|---|---|---|",
        ]
        n_flag = int(rej.sum())
        s = scores(f)
        candidates = [("Rejet MatchVerifier", rej)]
        for name, key in [
            ("IC le plus bas, même volume", "ic"),
            ("Score combiné, même volume", "comb"),
        ]:
            flag = np.zeros(len(err), bool)
            flag[np.argsort(-s[key], kind="stable")[:n_flag]] = True
            candidates.append((name, flag))
        low = ic < IC_LOW
        ic_low = num(IC_LOW, 1)
        candidates += [
            (f"IC < {ic_low}", low),
            (f"Rejet **et** IC < {ic_low}", rej & low),
            (f"Rejet **ou** IC < {ic_low}", rej | low),
        ]
        for name, flag in candidates:
            r = rates(err, flag)
            lines.append(
                f"| {name} | {pct(r['flag'])} | {pct(r['recall'])} | {pct(r['fr'])} | {pct(r['prec'])} |"
            )
        lines += [
            "",
            f"Erreurs de production repérées par le seul rejet : {int((rej & ~low & err).sum())} ; "
            f"par le seul IC < {ic_low} : {int((low & ~rej & err).sum())} ; par les deux : "
            f"{int((rej & low & err).sum())} ; par aucun : {int((~rej & ~low & err).sum())}.",
        ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", required=True, help="match_verifier_eval.parquet du run")
    parser.add_argument(
        "--pairs", required=True, help="pairs_test_NAF2008.parquet (une ligne par liasse)"
    )
    parser.add_argument("--dashboard", default=DASHBOARD_PATH)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    pairs = load_dataframe(args.pairs)
    rows = build_rows(load_dataframe(args.results), pairs, load_dashboard(args.dashboard))
    run = "/".join(args.results.split("/")[-3:-1])
    report = build_report(rows, pairs.height, run)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(report)
    logger.info(f"Report written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
