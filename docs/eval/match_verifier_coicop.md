# Évaluation du MatchVerifier sur la COICOP (Sirus vs annotation humaine)

**État : en cours** — 8106 / 8981 paires (90.3 %), mis à jour le 2026-10-13 04:57.

Rythme ≈ 14.5 paires/min, fin estimée dans ≈ 1.0 h.

## Protocole

- Données : `projet-budget-famille/.../2026-09-08/codif-c9vjm/reconcile-sirus/predictions.parquet`.
- Code jugé : `sirus_code` (prédiction Sirus, niveau ≤ 4). Référence : `code_lvl4`, le code humain (`code`, annotations vague 1) ramené au niveau 4.
- Deux vérités : **exacte** (Sirus = humain au niveau 4) et **de branche** (idem, ou Sirus sous-catégorie d'un code humain plus court).
- Erreur = Sirus ≠ humain ; le vérificateur « détecte » une erreur quand il rejette le code.
- Population : 14695 lignes, 400 écartées (code Sirus absent du graphe), 8981 paires distinctes. Taux d'erreur de Sirus : 19.6 % (exact), 17.3 % (branche).
- Métriques calculées par paire distincte (libellé, code Sirus, code humain).

## Résultats principaux

| Vérité | Paires | dont erreurs | Rappel des erreurs | Rejets à tort | Précision du rejet | Précision à la prévalence réelle | AUC p_match | Alternative = humain (erreurs détectées) |
|---|---|---|---|---|---|---|---|---|
| Exacte | 8106 | 1953 | 50.0 % | 7.2 % | 68.7 % | 62.9 % | 0.729 | 44.3 % |
| Branche | 8106 | 1808 | 51.5 % | 7.7 % | 65.6 % | 58.2 % | 0.736 | 42.4 % |
| Branche, hors codes 98/99 | 7799 | 1501 | 51.1 % | 7.7 % | 61.1 % | 57.9 % | 0.730 | 51.5 % |

- **Rappel des erreurs** : part des erreurs de Sirus que le vérificateur rejette.
- **Rejets à tort** : part des codes justes qu'il rejette.
- **Précision à la prévalence réelle** : part des rejets qui portent sur de vraies erreurs, ramenée au taux d'erreur de la population.
- **Alternative = humain** : quand il rejette à raison, son code alternatif tombe-t-il dans la branche du code humain ?

## Par type d'écart Sirus / humain

| Type | Paires | Rejetées par le vérificateur |
|---|---|---|
| Accord exact | 6153 | 7.2 % |
| Même groupe, classe/sous-classe différente | 702 | 49.6 % |
| Division différente | 616 | 57.0 % |
| Code humain technique (98/99) | 307 | 53.7 % |
| Humain moins précis (Sirus = sous-catégorie) | 145 | 30.3 % |
| Même division, groupe différent | 100 | 64.0 % |
| Sirus moins précis que l'humain | 83 | 4.8 % |

Pour « Accord exact », le taux de rejet est un taux de rejet à tort ; pour les autres lignes, plus il est haut, mieux le vérificateur détecte ce type d'écart.

## Point de fonctionnement (vérité de branche)

Le verdict `is_match` est un seuil parmi d'autres ; en seuillant `p_match` :

| Seuil (rejet si p_match < t) | Rappel des erreurs | Rejets à tort |
|---|---|---|
| 0.1 | 51.1 % | 7.7 % |
| 0.25 | 51.3 % | 7.7 % |
| 0.5 | 51.5 % | 7.8 % |
| 0.75 | 51.6 % | 7.9 % |
| 0.9 | 51.7 % | 7.9 % |

_Version agrégée, sans aucun libellé (données d'enquête soumises au secret statistique) : les exemples sont dans le rapport local, non versionné._

## Coût

Durée moyenne par paire : 19.9 s ; temps d'inférence cumulé : 44.8 h.
