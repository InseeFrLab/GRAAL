# Évaluation du MatchVerifier sur la COICOP (Sirus vs annotation humaine)

**État : en cours** — 5247 / 8981 paires (58.4 %), mis à jour le 2026-10-13 01:36.

Rythme ≈ 14.5 paires/min, fin estimée dans ≈ 4.3 h.

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
| Exacte | 5247 | 1278 | 49.1 % | 7.6 % | 67.5 % | 61.2 % | 0.723 | 43.3 % |
| Branche | 5247 | 1193 | 50.4 % | 8.1 % | 64.6 % | 56.5 % | 0.731 | 41.4 % |
| Branche, hors codes 98/99 | 5044 | 990 | 50.4 % | 8.1 % | 60.3 % | 56.5 % | 0.728 | 49.9 % |

- **Rappel des erreurs** : part des erreurs de Sirus que le vérificateur rejette.
- **Rejets à tort** : part des codes justes qu'il rejette.
- **Précision à la prévalence réelle** : part des rejets qui portent sur de vraies erreurs, ramenée au taux d'erreur de la population.
- **Alternative = humain** : quand il rejette à raison, son code alternatif tombe-t-il dans la branche du code humain ?

## Par type d'écart Sirus / humain

| Type | Paires | Rejetées par le vérificateur |
|---|---|---|
| Accord exact | 3969 | 7.6 % |
| Même groupe, classe/sous-classe différente | 467 | 48.8 % |
| Division différente | 402 | 55.7 % |
| Code humain technique (98/99) | 203 | 50.2 % |
| Humain moins précis (Sirus = sous-catégorie) | 85 | 31.8 % |
| Même division, groupe différent | 72 | 62.5 % |
| Sirus moins précis que l'humain | 49 | 4.1 % |

Pour « Accord exact », le taux de rejet est un taux de rejet à tort ; pour les autres lignes, plus il est haut, mieux le vérificateur détecte ce type d'écart.

## Point de fonctionnement (vérité de branche)

Le verdict `is_match` est un seuil parmi d'autres ; en seuillant `p_match` :

| Seuil (rejet si p_match < t) | Rappel des erreurs | Rejets à tort |
|---|---|---|
| 0.1 | 49.8 % | 8.0 % |
| 0.25 | 50.0 % | 8.0 % |
| 0.5 | 50.4 % | 8.1 % |
| 0.75 | 50.4 % | 8.2 % |
| 0.9 | 50.5 % | 8.3 % |

_Version agrégée, sans aucun libellé (données d'enquête soumises au secret statistique) : les exemples sont dans le rapport local, non versionné._

## Coût

Durée moyenne par paire : 19.8 s ; temps d'inférence cumulé : 28.9 h.
