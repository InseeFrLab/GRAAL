"""Détection de dérive de distribution, sans étiquettes (cf. cadrage §3.3-C).

Sert le cas d'usage monitoring : surveiller la distribution des prédictions
d'un modèle en production (scores de confiance, fréquence des codes prédits)
et détecter un décalage par rapport à une période de référence, sans
nécessiter d'annotation humaine continue.

Trois métriques usuelles en surveillance de modèles :
- **distance de Wasserstein** et **test de Kolmogorov-Smirnov** : pour des
  signaux continus (ex. scores de confiance du *CodeChooser*/*MatchVerifier*).
- **Population Stability Index (PSI)** : pour un signal continu (bins par
  quantiles de la référence) ou catégoriel (ex. fréquence des codes prédits).

Les seuils d'alerte ne sont pas des constantes arbitraires : `calibrate_threshold`
les dérive empiriquement de la référence elle-même (rééchantillonnage sous
l'hypothèse « pas de dérive »), pour un taux de fausse alerte contrôlé.
"""


import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance


def wasserstein_drift(reference: np.ndarray, current: np.ndarray) -> float:
    """Distance de Wasserstein (terre déplacée) entre deux échantillons continus.

    0 = distributions identiques ; croît avec l'ampleur du décalage. Sensible
    à l'échelle du signal (ex. pas comparable entre un score de confiance
    dans [0, 1] et un signal à plus grande échelle).
    """
    return float(wasserstein_distance(reference, current))


def ks_drift(reference: np.ndarray, current: np.ndarray) -> dict:
    """Test de Kolmogorov-Smirnov à deux échantillons.

    Returns:
        {"statistic": écart maximal entre fonctions de répartition empiriques,
         "p_value": probabilité d'observer un tel écart sous H0 (même distribution)}
    """
    statistic, p_value = ks_2samp(reference, current)
    return {"statistic": float(statistic), "p_value": float(p_value)}


def psi(reference: np.ndarray, current: np.ndarray, n_bins: int = 10, eps: float = 1e-4) -> float:
    """Population Stability Index entre deux échantillons continus.

    Les bins sont définis par les quantiles de `reference` (pas de `current`),
    pour que le découpage ne dépende que de la période de référence.

    Repères usuels en gouvernance de modèle : PSI < 0.1 = stable,
    0.1-0.25 = dérive modérée à surveiller, > 0.25 = dérive significative.
    """
    quantile_edges = np.quantile(reference, np.linspace(0, 1, n_bins + 1))
    quantile_edges[0], quantile_edges[-1] = -np.inf, np.inf

    ref_counts, _ = np.histogram(reference, bins=quantile_edges)
    cur_counts, _ = np.histogram(current, bins=quantile_edges)

    return _psi_from_counts(ref_counts, cur_counts, eps=eps)


def psi_categorical(reference_labels: list, current_labels: list, eps: float = 1e-4) -> float:
    """Population Stability Index entre deux échantillons catégoriels.

    Cas d'usage typique ici : comparer la distribution des codes prédits sur
    deux fenêtres temporelles. Les catégories vues dans l'une des deux
    fenêtres mais pas l'autre sont incluses (fréquence 0 lissée par `eps`).
    """
    categories = sorted(set(reference_labels) | set(current_labels))
    ref_counts = np.array([reference_labels.count(c) for c in categories])
    cur_counts = np.array([current_labels.count(c) for c in categories])
    return _psi_from_counts(ref_counts, cur_counts, eps=eps)


def _psi_from_counts(ref_counts: np.ndarray, cur_counts: np.ndarray, eps: float) -> float:
    ref_pct = ref_counts / ref_counts.sum() if ref_counts.sum() else ref_counts.astype(float)
    cur_pct = cur_counts / cur_counts.sum() if cur_counts.sum() else cur_counts.astype(float)
    ref_pct = np.clip(ref_pct, eps, None)
    cur_pct = np.clip(cur_pct, eps, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def calibrate_threshold(
    reference: np.ndarray,
    metric_fn,
    alpha: float = 0.05,
    n_resamples: int = 500,
    seed: int = 42,
) -> float:
    """Calibre un seuil d'alerte empirique à partir de la seule référence.

    Principe : sous l'hypothèse « pas de dérive », deux sous-échantillons
    aléatoires de la référence devraient donner une valeur de métrique
    proche de 0. On simule cette hypothèse par rééchantillonnage (bootstrap)
    et on prend le quantile (1 - alpha) de la distribution obtenue comme
    seuil : au-delà, la dérive observée est trop grande pour être due au
    seul bruit d'échantillonnage, au taux de faux positifs `alpha` près.

    Args:
        reference: Échantillon de la période de référence.
        metric_fn: Fonction (échantillon_a, échantillon_b) -> métrique scalaire
            (ex. `wasserstein_drift`, ou `lambda a, b: ks_drift(a, b)["statistic"]`).
        alpha: Taux de fausse alerte visé (défaut : 5 %).
        n_resamples: Nombre de tirages bootstrap.
        seed: Graine, pour un seuil reproductible.

    Returns:
        Seuil au-delà duquel une valeur de métrique est considérée comme une
        dérive réelle plutôt qu'une fluctuation d'échantillonnage.
    """
    rng = np.random.default_rng(seed)
    reference = np.asarray(reference)
    half = len(reference) // 2
    if half < 1:
        raise ValueError("reference must have at least 2 elements to calibrate a threshold")

    null_values = []
    for _ in range(n_resamples):
        shuffled = rng.permutation(reference)
        null_values.append(metric_fn(shuffled[:half], shuffled[half:]))

    return float(np.quantile(null_values, 1 - alpha))


def drift_report(
    reference: np.ndarray,
    current: np.ndarray,
    alpha: float = 0.05,
    n_resamples: int = 500,
    seed: int = 42,
) -> dict:
    """Rapport de dérive combinant les trois métriques sur un signal continu.

    Chaque métrique a son propre seuil calibré empiriquement sur `reference`
    (cf. `calibrate_threshold`) plutôt qu'un seuil arbitraire partagé.

    Returns:
        Dictionnaire avec, pour chaque métrique, la valeur observée, le seuil
        calibré et un booléen `is_drift`, plus `any_drift` (majorité simple).
    """
    reference = np.asarray(reference)
    current = np.asarray(current)

    wasserstein_value = wasserstein_drift(reference, current)
    wasserstein_threshold = calibrate_threshold(
        reference, wasserstein_drift, alpha=alpha, n_resamples=n_resamples, seed=seed
    )

    ks_result = ks_drift(reference, current)

    psi_value = psi(reference, current)
    psi_threshold = calibrate_threshold(
        reference, psi, alpha=alpha, n_resamples=n_resamples, seed=seed
    )

    report = {
        "wasserstein": {
            "value": wasserstein_value,
            "threshold": wasserstein_threshold,
            "is_drift": wasserstein_value > wasserstein_threshold,
        },
        "ks": {
            "statistic": ks_result["statistic"],
            "p_value": ks_result["p_value"],
            "is_drift": ks_result["p_value"] < alpha,
        },
        "psi": {
            "value": psi_value,
            "threshold": psi_threshold,
            "is_drift": psi_value > psi_threshold,
        },
    }
    n_alerts = sum(m["is_drift"] for m in report.values())
    report["any_drift"] = n_alerts >= 2  # majorité sur les 3 métriques
    return report


def rolling_drift_reports(
    reference: np.ndarray,
    windows: list,
    alpha: float = 0.05,
    n_resamples: int = 500,
    seed: int = 42,
) -> list:
    """Applique `drift_report` sur une suite de fenêtres temporelles glissantes.

    Args:
        reference: Échantillon de la période de référence (fixe).
        windows: Liste d'échantillons, un par fenêtre temporelle successive.

    Returns:
        Liste de rapports (un par fenêtre), dans l'ordre fourni.
    """
    return [
        drift_report(reference, window, alpha=alpha, n_resamples=n_resamples, seed=seed)
        for window in windows
    ]
