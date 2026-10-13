# Évaluation du MatchVerifier sur la COICOP (Sirus vs annotation humaine)

**État : en cours** — 7235 / 8981 paires (80.6 %), mis à jour le 2026-10-13 03:56.

Rythme ≈ 14.4 paires/min, fin estimée dans ≈ 2.0 h.

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
| Exacte | 7235 | 1740 | 49.2 % | 7.1 % | 68.6 % | 62.8 % | 0.727 | 44.5 % |
| Branche | 7235 | 1613 | 50.6 % | 7.7 % | 65.4 % | 58.0 % | 0.733 | 42.6 % |
| Branche, hors codes 98/99 | 6971 | 1349 | 50.4 % | 7.7 % | 61.2 % | 57.9 % | 0.729 | 51.2 % |

- **Rappel des erreurs** : part des erreurs de Sirus que le vérificateur rejette.
- **Rejets à tort** : part des codes justes qu'il rejette.
- **Précision à la prévalence réelle** : part des rejets qui portent sur de vraies erreurs, ramenée au taux d'erreur de la population.
- **Alternative = humain** : quand il rejette à raison, son code alternatif tombe-t-il dans la branche du code humain ?

## Par type d'écart Sirus / humain

| Type | Paires | Rejetées par le vérificateur |
|---|---|---|
| Accord exact | 5495 | 7.1 % |
| Même groupe, classe/sous-classe différente | 627 | 48.6 % |
| Division différente | 558 | 56.3 % |
| Code humain technique (98/99) | 264 | 51.5 % |
| Humain moins précis (Sirus = sous-catégorie) | 127 | 31.5 % |
| Même division, groupe différent | 90 | 64.4 % |
| Sirus moins précis que l'humain | 74 | 4.1 % |

Pour « Accord exact », le taux de rejet est un taux de rejet à tort ; pour les autres lignes, plus il est haut, mieux le vérificateur détecte ce type d'écart.

## Point de fonctionnement (vérité de branche)

Le verdict `is_match` est un seuil parmi d'autres ; en seuillant `p_match` :

| Seuil (rejet si p_match < t) | Rappel des erreurs | Rejets à tort |
|---|---|---|
| 0.1 | 50.1 % | 7.6 % |
| 0.25 | 50.3 % | 7.6 % |
| 0.5 | 50.6 % | 7.7 % |
| 0.75 | 50.7 % | 7.8 % |
| 0.9 | 50.7 % | 7.8 % |

_Version agrégée, sans aucun libellé (données d'enquête soumises au secret statistique) : les exemples sont dans le rapport local, non versionné._

## Coût

Durée moyenne par paire : 19.9 s ; temps d'inférence cumulé : 40.1 h.
