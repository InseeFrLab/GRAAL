# Évaluation du MatchVerifier sur la COICOP (Sirus vs annotation humaine)

**État : en cours** — 3691 / 8981 paires (41.1 %), mis à jour le 2026-10-12 23:50.

Rythme ≈ 14.5 paires/min, fin estimée dans ≈ 6.1 h.

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
| Exacte | 3691 | 922 | 50.0 % | 7.5 % | 69.0 % | 62.0 % | 0.726 | 43.2 % |
| Branche | 3691 | 855 | 51.5 % | 8.0 % | 65.9 % | 57.2 % | 0.737 | 40.9 % |
| Branche, hors codes 98/99 | 3532 | 696 | 51.6 % | 8.0 % | 61.2 % | 57.3 % | 0.735 | 50.1 % |

- **Rappel des erreurs** : part des erreurs de Sirus que le vérificateur rejette.
- **Rejets à tort** : part des codes justes qu'il rejette.
- **Précision à la prévalence réelle** : part des rejets qui portent sur de vraies erreurs, ramenée au taux d'erreur de la population.
- **Alternative = humain** : quand il rejette à raison, son code alternatif tombe-t-il dans la branche du code humain ?

## Par type d'écart Sirus / humain

| Type | Paires | Rejetées par le vérificateur |
|---|---|---|
| Accord exact | 2769 | 7.5 % |
| Même groupe, classe/sous-classe différente | 332 | 48.2 % |
| Division différente | 277 | 58.5 % |
| Code humain technique (98/99) | 159 | 50.9 % |
| Humain moins précis (Sirus = sous-catégorie) | 67 | 31.3 % |
| Même division, groupe différent | 58 | 62.1 % |
| Sirus moins précis que l'humain | 29 | 3.4 % |

Pour « Accord exact », le taux de rejet est un taux de rejet à tort ; pour les autres lignes, plus il est haut, mieux le vérificateur détecte ce type d'écart.

## Point de fonctionnement (vérité de branche)

Le verdict `is_match` est un seuil parmi d'autres ; en seuillant `p_match` :

| Seuil (rejet si p_match < t) | Rappel des erreurs | Rejets à tort |
|---|---|---|
| 0.1 | 50.9 % | 7.9 % |
| 0.25 | 51.0 % | 8.0 % |
| 0.5 | 51.5 % | 8.0 % |
| 0.75 | 51.5 % | 8.1 % |
| 0.9 | 51.6 % | 8.2 % |

_Version agrégée, sans aucun libellé (données d'enquête soumises au secret statistique) : les exemples sont dans le rapport local, non versionné._

## Coût

Durée moyenne par paire : 19.8 s ; temps d'inférence cumulé : 20.3 h.
