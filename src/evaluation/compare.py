"""Comparaison appariée de deux méthodes exécutées sur le même jeu d'évaluation.

cf. cadrage_2026-07.md §3.3-B. Deux méthodes évaluées sur le même jeu (mêmes
lignes, même vérité terrain) forment un plan **apparié** : comparer deux
intervalles de confiance indépendants et vérifier s'ils se chevauchent est un
test conservateur et statistiquement inadapté ici (il ignore la corrélation
ligne à ligne entre les deux méthodes). On utilise donc :

- un bootstrap apparié en grappes (par strate) sur la différence d'exactitude
  (`accuracy_b - accuracy_a`) ;
- le test de McNemar sur les paires discordantes, en complément rapide et
  sans rééchantillonnage, standard pour comparer deux classifieurs sur les
  mêmes observations.

Usage :
    uv run -m src.evaluation.compare \
        --a data/eval/results/predictions_navigator.parquet \
        --b data/eval/results/predictions_agentic-rag.parquet
"""

import argparse
import json

import numpy as np
import polars as pl
from scipy.stats import binomtest

from src.evaluation.metrics import accuracy_at_depth, normalize_code


def _is_correct(true_code, pred_code, depth: int | None = None) -> bool | None:
    """True/False si comparable, None si la vérité terrain est manquante (paire ignorée)."""
    true_norm = normalize_code(true_code)
    pred_norm = normalize_code(pred_code)
    if true_norm is None:
        return None
    if pred_norm is None:
        return False
    if depth is None:
        return true_norm == pred_norm
    return true_norm[:depth] == pred_norm[:depth]


def paired_bootstrap_diff(
    y_true: list,
    y_pred_a: list,
    y_pred_b: list,
    strata: list,
    depth: int | None = None,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, tuple[float, float]]:
    """IC bootstrap apparié (grappes par strate) sur `accuracy(b) - accuracy(a)`.

    Chaque rééchantillonnage tire les mêmes indices de lignes pour les deux
    méthodes (le rééchantillonnage porte sur les lignes, pas sur les
    prédictions séparément), ce qui préserve l'appariement.
    """
    n = len(y_true)
    if n != len(y_pred_a) or n != len(y_pred_b) or n != len(strata):
        raise ValueError("y_true, y_pred_a, y_pred_b et strata doivent avoir la même taille")

    y_true_arr = np.asarray(y_true, dtype=object)
    y_pred_a_arr = np.asarray(y_pred_a, dtype=object)
    y_pred_b_arr = np.asarray(y_pred_b, dtype=object)

    strata_indices: dict = {}
    for i, s in enumerate(strata):
        strata_indices.setdefault(s, []).append(i)
    strata_index_arrays = [np.array(idx) for idx in strata_indices.values()]

    def diff_for(idx: np.ndarray) -> float:
        t = y_true_arr[idx].tolist()
        acc_a = accuracy_at_depth(t, y_pred_a_arr[idx].tolist(), depth=depth)
        acc_b = accuracy_at_depth(t, y_pred_b_arr[idx].tolist(), depth=depth)
        return acc_b - acc_a

    point_diff = diff_for(np.arange(n))

    rng = np.random.default_rng(seed)
    diffs = np.empty(n_resamples)
    for r in range(n_resamples):
        resampled_idx = np.concatenate(
            [rng.choice(idx, size=len(idx), replace=True) for idx in strata_index_arrays]
        )
        diffs[r] = diff_for(resampled_idx)

    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    return float(point_diff), (float(lo), float(hi))


def mcnemar_test(y_true: list, y_pred_a: list, y_pred_b: list, depth: int | None = None) -> float:
    """P-valeur du test de McNemar exact sur les paires discordantes entre deux méthodes.

    Une paire est discordante quand une méthode a raison et l'autre a tort sur
    la même ligne ; les lignes sans vérité terrain exploitable sont ignorées.
    """
    n_a_only = 0  # a correct, b incorrect
    n_b_only = 0  # a incorrect, b correct
    for true_code, pred_a, pred_b in zip(y_true, y_pred_a, y_pred_b):
        correct_a = _is_correct(true_code, pred_a, depth=depth)
        correct_b = _is_correct(true_code, pred_b, depth=depth)
        if correct_a is None or correct_b is None:
            continue
        if correct_a and not correct_b:
            n_a_only += 1
        elif correct_b and not correct_a:
            n_b_only += 1

    discordant = n_a_only + n_b_only
    if discordant == 0:
        return 1.0
    return float(binomtest(min(n_a_only, n_b_only), discordant, p=0.5).pvalue)


def compare(
    df_a: pl.DataFrame,
    df_b: pl.DataFrame,
    code_column: str = "apet2025",
    prediction_column: str = "prediction",
    stratum_column: str = "eval_stratum",
    depth: int | None = None,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    """Compare deux campagnes exécutées sur le même jeu d'évaluation (même ordre de lignes)."""
    if len(df_a) != len(df_b):
        raise ValueError("Les deux jeux de prédictions doivent avoir le même nombre de lignes")

    y_true = df_a[code_column].to_list()
    if y_true != df_b[code_column].to_list():
        raise ValueError(
            "Les vérités terrain diffèrent entre les deux fichiers : lignes non alignées"
        )

    y_pred_a = df_a[prediction_column].to_list()
    y_pred_b = df_b[prediction_column].to_list()
    strata = (
        df_a[stratum_column].to_list() if stratum_column in df_a.columns else list(range(len(df_a)))
    )

    accuracy_a = accuracy_at_depth(y_true, y_pred_a, depth=depth)
    accuracy_b = accuracy_at_depth(y_true, y_pred_b, depth=depth)
    diff_point, diff_ci = paired_bootstrap_diff(
        y_true,
        y_pred_a,
        y_pred_b,
        strata,
        depth=depth,
        n_resamples=n_resamples,
        alpha=alpha,
        seed=seed,
    )
    p_value = mcnemar_test(y_true, y_pred_a, y_pred_b, depth=depth)

    return {
        "n": len(y_true),
        "accuracy_a": accuracy_a,
        "accuracy_b": accuracy_b,
        "diff": diff_point,
        "diff_ci": list(diff_ci),
        "mcnemar_p_value": p_value,
        "significant": p_value < alpha,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Paired statistical comparison of two evaluation campaigns"
    )
    parser.add_argument("--a", required=True, help="predictions_<method>.parquet for method A")
    parser.add_argument("--b", required=True, help="predictions_<method>.parquet for method B")
    parser.add_argument("--code-column", default="apet2025")
    parser.add_argument("--prediction-column", default="prediction")
    parser.add_argument("--stratum-column", default="eval_stratum")
    parser.add_argument(
        "--depth", type=int, default=None, help="Prefix depth to compare (default: leaf)"
    )
    parser.add_argument("--n-resamples", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None, help="Optional path to write the JSON report")
    args = parser.parse_args()

    df_a = pl.read_parquet(args.a)
    df_b = pl.read_parquet(args.b)
    report = compare(
        df_a,
        df_b,
        code_column=args.code_column,
        prediction_column=args.prediction_column,
        stratum_column=args.stratum_column,
        depth=args.depth,
        n_resamples=args.n_resamples,
        alpha=args.alpha,
        seed=args.seed,
    )

    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
