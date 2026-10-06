"""Métriques d'évaluation pour la classification hiérarchique.

Volontairement sans dépendance tierce : le module reste importable et testable
dans un environnement de CI minimal, sans Neo4j ni client LLM.

Les codes sont normalisés (suppression des points/espaces, majuscules) avant
comparaison, si bien que "10.71C" et "1071c" sont considérés identiques.
Pour la NAF/NACE, les préfixes du code normalisé correspondent aux niveaux
de la hiérarchie : 2 caractères = division, 3 = groupe, 4 = classe,
code complet = sous-classe (feuille).
"""


def normalize_code(code) -> str | None:
    """Normalise un code de nomenclature pour comparaison.

    Retourne None pour une prédiction manquante (None, chaîne vide),
    ce qui matérialise un échec du classifieur (pas de code final atteint).
    """
    if code is None:
        return None
    normalized = str(code).replace(".", "").replace(" ", "").upper()
    return normalized or None


def accuracy_at_depth(
    y_true: list, y_pred: list, depth: int | None = None, weights: list | None = None
) -> float:
    """Part des prédictions égales à la vérité terrain sur les `depth` premiers caractères.

    Args:
        y_true: Codes de référence.
        y_pred: Codes prédits (None = échec du classifieur, compté comme erreur).
        depth: Profondeur de comparaison ; None compare les codes complets
            (exactitude à la feuille).
        weights: Poids par ligne (ex. correction de sur-échantillonnage des
            strates rares, cf. `build_eval_set.ipw_weight`). None = poids
            uniformes (comportement historique : exactitude moyenne par
            ligne du jeu d'évaluation, pas par la fréquence réelle des codes).

    Returns:
        Exactitude entre 0 et 1 ; NaN si aucune paire exploitable. Les paires
        dont la vérité terrain est manquante sont ignorées.
    """
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true ({len(y_true)}) et y_pred ({len(y_pred)}) doivent avoir la même taille"
        )
    if weights is not None and len(weights) != len(y_true):
        raise ValueError(
            f"weights ({len(weights)}) et y_true ({len(y_true)}) doivent avoir la même taille"
        )

    correct = 0.0
    total = 0.0
    for i, (true_code, pred_code) in enumerate(zip(y_true, y_pred)):
        true_norm = normalize_code(true_code)
        pred_norm = normalize_code(pred_code)
        if true_norm is None:
            continue
        weight = weights[i] if weights is not None else 1.0
        total += weight
        if pred_norm is None:
            continue
        if depth is None:
            is_correct = true_norm == pred_norm
        else:
            is_correct = true_norm[:depth] == pred_norm[:depth]
        correct += weight if is_correct else 0.0

    return correct / total if total else float("nan")


def failure_rate(y_pred: list) -> float:
    """Part des prédictions manquantes (le classifieur n'a pas atteint de code final)."""
    if not y_pred:
        return float("nan")
    return sum(1 for pred in y_pred if normalize_code(pred) is None) / len(y_pred)


def low_confidence_rate(confidences: list, threshold: float = 0.0) -> float:
    """Part des prédictions à confiance faible ou nulle.

    Distinct de `failure_rate` : un classifieur peut renvoyer un code réel
    (pas None) avec une confiance nulle quand la finalisation a échoué et
    qu'un repli sur la dernière position atteinte a été utilisé (cf.
    `_fallback_output`). Ces cas ne sont pas des échecs au sens de
    `failure_rate` (il y a bien un code) mais ne doivent pas être confondus
    avec une prédiction confiante lors de l'analyse d'erreurs.

    Args:
        confidences: Scores de confiance par prédiction (None = non renseigné,
            ignoré du calcul).
        threshold: Seuil (inclusif) en dessous duquel une confiance est
            considérée comme faible (défaut : 0.0, le sentinel de repli).

    Returns:
        Part des prédictions dont la confiance est renseignée et <= threshold,
        rapportée à l'ensemble des prédictions (confiances manquantes comptées
        au dénominateur, jamais au numérateur) ; NaN si la liste est vide.
    """
    if not confidences:
        return float("nan")
    return sum(1 for c in confidences if c is not None and c <= threshold) / len(confidences)


def fallback_rate(explanations: list, marker: str = "[non-final fallback]") -> float:
    """Part des prédictions dont l'explication porte le marqueur de repli non-final.

    Distinct de `low_confidence_rate` : une confiance nulle peut aussi venir d'un
    échec générique (ex. le fallback de run_eval.py sur une exception LLM), ce qui ne
    permet pas d'isoler spécifiquement les cas où un classifieur sans garde-fou Python
    (ex. SummaryAgenticClassifier) a dû être corrigé après avoir renvoyé un code non
    terminal. Compte les explications portant `marker`, un préfixe posé explicitement
    à cette fin par ce genre de repli.

    Args:
        explanations: Explications par prédiction (None = non renseignée, ignorée du
            calcul).
        marker: Sous-chaîne identifiant un repli de ce type (défaut : le préfixe posé
            par SummaryAgenticClassifier).

    Returns:
        Part des explications renseignées contenant `marker`, rapportée à l'ensemble
        des prédictions (explications manquantes comptées au dénominateur, jamais au
        numérateur) ; NaN si la liste est vide.
    """
    if not explanations:
        return float("nan")
    return sum(1 for e in explanations if e is not None and marker in e) / len(explanations)


def retry_rate(attempt_counts: list) -> float:
    """Part des prédictions ayant eu besoin d'au moins une nouvelle tentative.

    Distinct de `failure_rate` : un appel qui échoue puis réussit au rattrapage
    (cf. `src.utils.retry.call_with_retries`) n'est pas un échec, mais mérite d'être
    distingué d'un succès dès la première tentative — un taux élevé signale une
    instabilité (timeouts, endpoint LLM saturé) qui resterait invisible dans les
    seules métriques d'exactitude.

    Args:
        attempt_counts: Nombre de tentatives ayant réussi par prédiction (None =
            non renseigné — call_with_retries n'a pas été utilisé, ou l'attribut
            n'existe pas sur ce type de sortie — ignoré du calcul).

    Returns:
        Part des tentatives renseignées strictement supérieures à 1, rapportée à
        l'ensemble des prédictions (valeurs manquantes comptées au dénominateur,
        jamais au numérateur) ; NaN si la liste est vide.
    """
    if not attempt_counts:
        return float("nan")
    return sum(1 for a in attempt_counts if a is not None and a > 1) / len(attempt_counts)


def evaluate(
    y_true: list,
    y_pred: list,
    depths: tuple = (2, 3, 4),
    weights: list | None = None,
    confidences: list | None = None,
    explanations: list | None = None,
    attempt_counts: list | None = None,
) -> dict:
    """Rapport d'évaluation complet d'un classifieur.

    Args:
        y_true: Codes de référence.
        y_pred: Codes prédits (None = échec).
        depths: Profondeurs de préfixe pour l'exactitude par niveau
            (par défaut : division, groupe, classe pour la NAF).
        weights: Poids par ligne pour une lecture pondérée (représentative de
            la fréquence réelle des codes) en complément de la lecture non
            pondérée (moyenne égale par code) — les deux sont toujours
            calculées quand `weights` est fourni, aucune n'écrase l'autre.
        confidences: Scores de confiance par prédiction, pour reporter
            `low_confidence_rate` en plus de `failure_rate` (optionnel).
        explanations: Explications par prédiction, pour reporter
            `non_final_fallback_rate` (optionnel, cf. `fallback_rate`).
        attempt_counts: Nombre de tentatives par prédiction, pour reporter
            `retry_rate` (optionnel, cf. `retry_rate`).

    Returns:
        Dictionnaire {n, leaf_accuracy, failure_rate, accuracy_depth_<d>...},
        avec en plus {leaf_accuracy_weighted, accuracy_depth_<d>_weighted...}
        si `weights` est fourni, {low_confidence_rate} si `confidences` est
        fourni, {non_final_fallback_rate} si `explanations` est fourni, et
        {retry_rate} si `attempt_counts` est fourni.
    """
    report = {
        "n": len(y_true),
        "leaf_accuracy": accuracy_at_depth(y_true, y_pred),
        "failure_rate": failure_rate(y_pred),
    }
    for depth in depths:
        report[f"accuracy_depth_{depth}"] = accuracy_at_depth(y_true, y_pred, depth=depth)

    if weights is not None:
        report["leaf_accuracy_weighted"] = accuracy_at_depth(y_true, y_pred, weights=weights)
        for depth in depths:
            report[f"accuracy_depth_{depth}_weighted"] = accuracy_at_depth(
                y_true, y_pred, depth=depth, weights=weights
            )

    if confidences is not None:
        report["low_confidence_rate"] = low_confidence_rate(confidences)

    if explanations is not None:
        report["non_final_fallback_rate"] = fallback_rate(explanations)

    if attempt_counts is not None:
        report["retry_rate"] = retry_rate(attempt_counts)

    return report


def roc_auc(labels: list, scores: list) -> float:
    """Aire sous la courbe ROC : P(score d'un positif > score d'un négatif).

    Calculée par les rangs (statistique de Mann-Whitney), les ex-aequo comptant pour
    moitié. Les paires dont le score est None sont ignorées.

    Returns:
        AUC entre 0 et 1 ; NaN s'il manque des positifs ou des négatifs.
    """
    pairs = [(s, bool(y)) for y, s in zip(labels, scores) if s is not None]
    n_pos = sum(1 for _, y in pairs if y)
    n_neg = len(pairs) - n_pos
    if not n_pos or not n_neg:
        return float("nan")
    pairs.sort(key=lambda p: p[0])
    rank_sum_pos = 0.0
    i = 0
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        mean_rank = (i + 1 + j) / 2  # rangs 1-indexés i+1..j
        rank_sum_pos += mean_rank * sum(1 for _, y in pairs[i:j] if y)
        i = j
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def verifier_metrics(
    expected_match: list,
    verdicts: list,
    p_match: list | None = None,
    error_prevalence: float | None = None,
) -> dict:
    """Qualité d'un vérificateur de paires (libellé, code) face à une vérité terrain.

    La classe d'intérêt est l'**erreur** : un vérificateur sert à attraper les codes
    faux, donc « positif » veut dire ici « le code jugé n'est pas le bon »
    (`expected_match` faux) et « détecté » veut dire « rejeté » (`verdict` faux).

    Args:
        expected_match: Par paire, le code jugé est-il le bon (vérité terrain) ?
        verdicts: Par paire, `is_match` rendu par le vérificateur.
        p_match: Par paire, P(is_match) lue dans les logprobs (None = absente), pour
            l'AUC — qui ne dépend pas du seuil, contrairement au verdict.
        error_prevalence: Part d'erreurs dans la population d'où vient l'échantillon.
            Un échantillon équilibré entre bons et mauvais codes surestime la
            précision du rejet ; elle est reportée en plus à cette prévalence.

    Returns:
        Dictionnaire {n, n_errors, n_correct, error_recall, false_rejection_rate,
        rejection_precision, balanced_accuracy, accuracy, roc_auc}, plus
        {rejection_precision_at_prevalence} si `error_prevalence` est fourni.
        Les taux sans dénominateur valent NaN.
    """
    if len(expected_match) != len(verdicts):
        raise ValueError(
            f"expected_match ({len(expected_match)}) et verdicts ({len(verdicts)}) "
            "doivent avoir la même taille"
        )
    is_error = [not bool(e) for e in expected_match]
    rejected = [not bool(v) for v in verdicts]
    n_errors = sum(is_error)
    n_correct = len(is_error) - n_errors
    caught = sum(1 for e, r in zip(is_error, rejected) if e and r)
    false_rejections = sum(1 for e, r in zip(is_error, rejected) if not e and r)
    nan = float("nan")
    error_recall = caught / n_errors if n_errors else nan
    false_rejection_rate = false_rejections / n_correct if n_correct else nan
    n_rejected = caught + false_rejections
    report = {
        "n": len(is_error),
        "n_errors": n_errors,
        "n_correct": n_correct,
        "error_recall": error_recall,
        "false_rejection_rate": false_rejection_rate,
        "rejection_precision": caught / n_rejected if n_rejected else nan,
        "balanced_accuracy": (error_recall + 1 - false_rejection_rate) / 2,
        "accuracy": sum(1 for e, r in zip(is_error, rejected) if e == r) / len(is_error)
        if is_error
        else nan,
        "roc_auc": roc_auc(is_error, [None if p is None else 1 - p for p in p_match])
        if p_match is not None
        else nan,
    }
    if error_prevalence is not None:
        flagged_errors = error_prevalence * error_recall
        flagged_correct = (1 - error_prevalence) * false_rejection_rate
        report["rejection_precision_at_prevalence"] = (
            flagged_errors / (flagged_errors + flagged_correct)
            if flagged_errors + flagged_correct
            else nan
        )
    return report
