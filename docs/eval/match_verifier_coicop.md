# Évaluation du MatchVerifier sur la COICOP (Sirus vs annotation humaine)

**État : en cours** — 4546 / 8981 paires (50.6 %), mis à jour le 2026-10-13 00:50.

Rythme ≈ 14.5 paires/min, fin estimée dans ≈ 5.1 h.

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
| Exacte | 4546 | 1109 | 49.7 % | 7.4 % | 68.4 % | 62.1 % | 0.724 | 42.5 % |
| Branche | 4546 | 1031 | 51.2 % | 7.9 % | 65.5 % | 57.5 % | 0.734 | 40.5 % |
| Branche, hors codes 98/99 | 4364 | 849 | 51.2 % | 7.9 % | 61.0 % | 57.5 % | 0.732 | 49.2 % |

- **Rappel des erreurs** : part des erreurs de Sirus que le vérificateur rejette.
- **Rejets à tort** : part des codes justes qu'il rejette.
- **Précision à la prévalence réelle** : part des rejets qui portent sur de vraies erreurs, ramenée au taux d'erreur de la population.
- **Alternative = humain** : quand il rejette à raison, son code alternatif tombe-t-il dans la branche du code humain ?

## Par type d'écart Sirus / humain

| Type | Paires | Rejetées par le vérificateur |
|---|---|---|
| Accord exact | 3437 | 7.4 % |
| Même groupe, classe/sous-classe différente | 407 | 49.4 % |
| Division différente | 340 | 56.5 % |
| Code humain technique (98/99) | 182 | 51.1 % |
| Humain moins précis (Sirus = sous-catégorie) | 78 | 29.5 % |
| Même division, groupe différent | 64 | 62.5 % |
| Sirus moins précis que l'humain | 38 | 5.3 % |

Pour « Accord exact », le taux de rejet est un taux de rejet à tort ; pour les autres lignes, plus il est haut, mieux le vérificateur détecte ce type d'écart.

## Point de fonctionnement (vérité de branche)

Le verdict `is_match` est un seuil parmi d'autres ; en seuillant `p_match` :

| Seuil (rejet si p_match < t) | Rappel des erreurs | Rejets à tort |
|---|---|---|
| 0.1 | 50.7 % | 7.8 % |
| 0.25 | 50.8 % | 7.8 % |
| 0.5 | 51.2 % | 7.9 % |
| 0.75 | 51.2 % | 8.1 % |
| 0.9 | 51.3 % | 8.1 % |

_Version agrégée, sans aucun libellé (données d'enquête soumises au secret statistique) : les exemples sont dans le rapport local, non versionné._

## Coût

Durée moyenne par paire : 19.9 s ; temps d'inférence cumulé : 25.2 h.
