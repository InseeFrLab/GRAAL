# Évaluation du MatchVerifier sur la COICOP (Sirus vs annotation humaine)

**État : en cours** — 3594 / 8981 paires (40.0 %), mis à jour le 2026-10-12 23:43.

Rythme ≈ 14.5 paires/min, fin estimée dans ≈ 6.2 h.

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
| Exacte | 3594 | 894 | 50.1 % | 7.6 % | 68.7 % | 61.8 % | 0.727 | 43.5 % |
| Branche | 3594 | 829 | 51.6 % | 8.1 % | 65.6 % | 57.1 % | 0.738 | 41.4 % |
| Branche, hors codes 98/99 | 3440 | 675 | 51.9 % | 8.1 % | 61.0 % | 57.2 % | 0.736 | 50.6 % |

- **Rappel des erreurs** : part des erreurs de Sirus que le vérificateur rejette.
- **Rejets à tort** : part des codes justes qu'il rejette.
- **Précision à la prévalence réelle** : part des rejets qui portent sur de vraies erreurs, ramenée au taux d'erreur de la population.
- **Alternative = humain** : quand il rejette à raison, son code alternatif tombe-t-il dans la branche du code humain ?

## Par type d'écart Sirus / humain

| Type | Paires | Rejetées par le vérificateur |
|---|---|---|
| Accord exact | 2700 | 7.6 % |
| Même groupe, classe/sous-classe différente | 324 | 48.5 % |
| Division différente | 266 | 58.6 % |
| Code humain technique (98/99) | 154 | 50.6 % |
| Humain moins précis (Sirus = sous-catégorie) | 65 | 30.8 % |
| Même division, groupe différent | 56 | 64.3 % |
| Sirus moins précis que l'humain | 29 | 3.4 % |

Pour « Accord exact », le taux de rejet est un taux de rejet à tort ; pour les autres lignes, plus il est haut, mieux le vérificateur détecte ce type d'écart.

## Point de fonctionnement (vérité de branche)

Le verdict `is_match` est un seuil parmi d'autres ; en seuillant `p_match` :

| Seuil (rejet si p_match < t) | Rappel des erreurs | Rejets à tort |
|---|---|---|
| 0.1 | 51.0 % | 8.0 % |
| 0.25 | 51.1 % | 8.0 % |
| 0.5 | 51.6 % | 8.1 % |
| 0.75 | 51.6 % | 8.2 % |
| 0.9 | 51.7 % | 8.2 % |

_Version agrégée, sans aucun libellé (données d'enquête soumises au secret statistique) : les exemples sont dans le rapport local, non versionné._

## Coût

Durée moyenne par paire : 19.8 s ; temps d'inférence cumulé : 19.8 h.
