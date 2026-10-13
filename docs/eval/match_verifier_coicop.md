# Évaluation du MatchVerifier sur la COICOP (Sirus vs annotation humaine)

**État : en cours** — 6311 / 8981 paires (70.3 %), mis à jour le 2026-10-13 02:51.

Rythme ≈ 14.5 paires/min, fin estimée dans ≈ 3.1 h.

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
| Exacte | 6311 | 1522 | 49.1 % | 7.4 % | 67.7 % | 61.7 % | 0.723 | 43.4 % |
| Branche | 6311 | 1411 | 50.4 % | 8.0 % | 64.5 % | 56.8 % | 0.729 | 41.4 % |
| Branche, hors codes 98/99 | 6078 | 1178 | 50.1 % | 8.0 % | 60.1 % | 56.7 % | 0.725 | 49.8 % |

- **Rappel des erreurs** : part des erreurs de Sirus que le vérificateur rejette.
- **Rejets à tort** : part des codes justes qu'il rejette.
- **Précision à la prévalence réelle** : part des rejets qui portent sur de vraies erreurs, ramenée au taux d'erreur de la population.
- **Alternative = humain** : quand il rejette à raison, son code alternatif tombe-t-il dans la branche du code humain ?

## Par type d'écart Sirus / humain

| Type | Paires | Rejetées par le vérificateur |
|---|---|---|
| Accord exact | 4789 | 7.4 % |
| Même groupe, classe/sous-classe différente | 553 | 47.9 % |
| Division différente | 478 | 56.1 % |
| Code humain technique (98/99) | 233 | 51.9 % |
| Humain moins précis (Sirus = sous-catégorie) | 111 | 32.4 % |
| Même division, groupe différent | 85 | 63.5 % |
| Sirus moins précis que l'humain | 62 | 4.8 % |

Pour « Accord exact », le taux de rejet est un taux de rejet à tort ; pour les autres lignes, plus il est haut, mieux le vérificateur détecte ce type d'écart.

## Point de fonctionnement (vérité de branche)

Le verdict `is_match` est un seuil parmi d'autres ; en seuillant `p_match` :

| Seuil (rejet si p_match < t) | Rappel des erreurs | Rejets à tort |
|---|---|---|
| 0.1 | 49.9 % | 7.9 % |
| 0.25 | 50.0 % | 7.9 % |
| 0.5 | 50.4 % | 8.0 % |
| 0.75 | 50.4 % | 8.1 % |
| 0.9 | 50.5 % | 8.2 % |

_Version agrégée, sans aucun libellé (données d'enquête soumises au secret statistique) : les exemples sont dans le rapport local, non versionné._

## Coût

Durée moyenne par paire : 19.9 s ; temps d'inférence cumulé : 34.9 h.
