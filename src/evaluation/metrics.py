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


def accuracy_at_depth(y_true: list, y_pred: list, depth: int | None = None) -> float:
    """Part des prédictions égales à la vérité terrain sur les `depth` premiers caractères.

    Args:
        y_true: Codes de référence.
        y_pred: Codes prédits (None = échec du classifieur, compté comme erreur).
        depth: Profondeur de comparaison ; None compare les codes complets
            (exactitude à la feuille).

    Returns:
        Exactitude entre 0 et 1 ; NaN si aucune paire exploitable. Les paires
        dont la vérité terrain est manquante sont ignorées.
    """
    if len(y_true) != len(y_pred):
        raise ValueError(f"y_true ({len(y_true)}) et y_pred ({len(y_pred)}) doivent avoir la même taille")

    correct = 0
    total = 0
    for true_code, pred_code in zip(y_true, y_pred):
        true_norm = normalize_code(true_code)
        pred_norm = normalize_code(pred_code)
        if true_norm is None:
            continue
        total += 1
        if pred_norm is None:
            continue
        if depth is None:
            correct += true_norm == pred_norm
        else:
            correct += true_norm[:depth] == pred_norm[:depth]

    return correct / total if total else float("nan")


def failure_rate(y_pred: list) -> float:
    """Part des prédictions manquantes (le classifieur n'a pas atteint de code final)."""
    if not y_pred:
        return float("nan")
    return sum(1 for pred in y_pred if normalize_code(pred) is None) / len(y_pred)


def evaluate(y_true: list, y_pred: list, depths: tuple = (2, 3, 4)) -> dict:
    """Rapport d'évaluation complet d'un classifieur.

    Args:
        y_true: Codes de référence.
        y_pred: Codes prédits (None = échec).
        depths: Profondeurs de préfixe pour l'exactitude par niveau
            (par défaut : division, groupe, classe pour la NAF).

    Returns:
        Dictionnaire {n, leaf_accuracy, failure_rate, accuracy_depth_<d>...}.
    """
    report = {
        "n": len(y_true),
        "leaf_accuracy": accuracy_at_depth(y_true, y_pred),
        "failure_rate": failure_rate(y_pred),
    }
    for depth in depths:
        report[f"accuracy_depth_{depth}"] = accuracy_at_depth(y_true, y_pred, depth=depth)
    return report
