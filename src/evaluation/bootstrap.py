"""Intervalles de confiance bootstrap, respectant la stratification du jeu d'évaluation.

cf. cadrage_2026-07.md §3.3-B : ne pas se limiter à un delta d'exactitude entre
méthodes sans intervalle de confiance, en particulier sur les strates à faible
effectif. Le jeu d'évaluation (`build_eval_set.stratified_sample`) est construit
par strate (tirage plafonné par code) : un bootstrap qui rééchantillonne les
lignes empilées sans distinction de strate ne représente pas ce plan
d'échantillonnage. `bootstrap_ci` rééchantillonne donc à l'intérieur de chaque
strate (bootstrap en grappes), jamais entre strates.
"""

import numpy as np


def bootstrap_ci(
    y_true: list,
    y_pred: list,
    strata: list,
    metric_fn,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
    weights: list | None = None,
) -> tuple[float, tuple[float, float]]:
    """Intervalle de confiance bootstrap pour une métrique, par rééchantillonnage en grappes.

    Args:
        y_true: Codes de référence.
        y_pred: Codes prédits, alignés avec `y_true`.
        strata: Clé de strate par ligne (ex. colonne `eval_stratum` de l'eval
            set) ; chaque rééchantillonnage tire, pour chaque strate, autant
            de lignes que la strate en contient, avec remise, uniquement
            parmi les lignes de cette même strate.
        metric_fn: Fonction `(y_true, y_pred, weights) -> float` (ex.
            `lambda t, p, w: accuracy_at_depth(t, p, weights=w)`).
        n_resamples: Nombre de tirages bootstrap.
        alpha: Taux de risque (défaut 5 %, IC à 95 %).
        seed: Graine, pour un IC reproductible.
        weights: Poids par ligne, propagés au rééchantillonnage (ex.
            `ipw_weight`) ; None = poids uniformes.

    Returns:
        `(estimation ponctuelle, (borne basse, borne haute))`, l'IC étant
        calculé par la méthode des percentiles sur la distribution des
        rééchantillonnages.
    """
    n = len(y_true)
    if n != len(y_pred) or n != len(strata):
        raise ValueError("y_true, y_pred et strata doivent avoir la même taille")
    if weights is not None and len(weights) != n:
        raise ValueError("weights doit avoir la même taille que y_true")

    y_true_arr = np.asarray(y_true, dtype=object)
    y_pred_arr = np.asarray(y_pred, dtype=object)
    weights_arr = np.asarray(weights, dtype=float) if weights is not None else None

    strata_indices: dict = {}
    for i, s in enumerate(strata):
        strata_indices.setdefault(s, []).append(i)
    strata_index_arrays = [np.array(idx) for idx in strata_indices.values()]

    point_estimate = metric_fn(list(y_true), list(y_pred), weights)

    rng = np.random.default_rng(seed)
    resample_values = np.empty(n_resamples)
    for r in range(n_resamples):
        resampled_idx = np.concatenate(
            [rng.choice(idx, size=len(idx), replace=True) for idx in strata_index_arrays]
        )
        resampled_y_true = y_true_arr[resampled_idx].tolist()
        resampled_y_pred = y_pred_arr[resampled_idx].tolist()
        resampled_weights = weights_arr[resampled_idx].tolist() if weights_arr is not None else None
        resample_values[r] = metric_fn(resampled_y_true, resampled_y_pred, resampled_weights)

    lo, hi = np.quantile(resample_values, [alpha / 2, 1 - alpha / 2])
    return float(point_estimate), (float(lo), float(hi))
