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


def evaluate(
    y_true: list,
    y_pred: list,
    depths: tuple = (2, 3, 4),
    weights: list | None = None,
    confidences: list | None = None,
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

    Returns:
        Dictionnaire {n, leaf_accuracy, failure_rate, accuracy_depth_<d>...},
        avec en plus {leaf_accuracy_weighted, accuracy_depth_<d>_weighted...}
        si `weights` est fourni, et {low_confidence_rate} si `confidences`
        est fourni.
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

    return report
