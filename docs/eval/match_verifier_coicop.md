# Évaluation du MatchVerifier sur la COICOP (Sirus vs annotation humaine)

**État : terminé** — 8909 / 8981 paires (99.2 %), mis à jour le 2026-10-13 22:55.

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
| Exacte | 8909 | 2148 | 49.8 % | 7.1 % | 68.9 % | 63.0 % | 0.729 | 44.2 % |
| Branche | 8909 | 1988 | 51.4 % | 7.6 % | 65.9 % | 58.4 % | 0.738 | 42.4 % |
| Branche, hors codes 98/99 | 8568 | 1647 | 51.1 % | 7.6 % | 61.4 % | 58.3 % | 0.732 | 51.5 % |

- **Rappel des erreurs** : part des erreurs de Sirus que le vérificateur rejette.
- **Rejets à tort** : part des codes justes qu'il rejette.
- **Précision à la prévalence réelle** : part des rejets qui portent sur de vraies erreurs, ramenée au taux d'erreur de la population.
- **Alternative = humain** : quand il rejette à raison, son code alternatif tombe-t-il dans la branche du code humain ?

## Par type d'écart Sirus / humain

| Type | Paires | Rejetées par le vérificateur |
|---|---|---|
| Accord exact | 6761 | 7.1 % |
| Même groupe, classe/sous-classe différente | 768 | 48.8 % |
| Division différente | 675 | 57.5 % |
| Code humain technique (98/99) | 341 | 53.1 % |
| Humain moins précis (Sirus = sous-catégorie) | 160 | 29.4 % |
| Même division, groupe différent | 109 | 66.1 % |
| Sirus moins précis que l'humain | 95 | 6.3 % |

Pour « Accord exact », le taux de rejet est un taux de rejet à tort ; pour les autres lignes, plus il est haut, mieux le vérificateur détecte ce type d'écart.

## Point de fonctionnement (vérité de branche)

Le verdict `is_match` est un seuil parmi d'autres ; en seuillant `p_match` :

| Seuil (rejet si p_match < t) | Rappel des erreurs | Rejets à tort |
|---|---|---|
| 0.1 | 51.0 % | 7.5 % |
| 0.25 | 51.2 % | 7.6 % |
| 0.5 | 51.4 % | 7.7 % |
| 0.75 | 51.5 % | 7.8 % |
| 0.9 | 51.5 % | 7.8 % |

_Version agrégée, sans aucun libellé (données d'enquête soumises au secret statistique) : les exemples sont dans le rapport local, non versionné._

## Second classifieur : SummaryAgenticClassifier

**État : en cours** — 5264 / 8717 libellés classés (60.4 %), 5437 paires couvertes.

Le Summary classifier reçoit le libellé seul et le résumé complet de la COICOP, et choisit un code terminal. Son code est ramené au niveau 4 pour être comparé au code humain, comme Sirus.

### Exactitude face à l'humain

| Paires | n | Sirus exact | Summary exact | Sirus branche | Summary branche | Sirus division | Summary division | Sirus groupe | Summary groupe |
|---|---|---|---|---|---|---|---|---|---|
| Toutes les paires | 5437 | 75.3 % | 53.8 % | 77.0 % | 67.6 % | 88.3 % | 83.7 % | 86.9 % | 81.3 % |
| Hors codes 98/99 | 5231 | 78.3 % | 55.9 % | 80.0 % | 70.3 % | 91.8 % | 87.0 % | 90.3 % | 84.5 % |

### Détecter les erreurs de Sirus (vérité de branche, hors 98/99)

| Signal | Lignes signalées | Erreurs détectées | Codes justes signalés | Précision au taux réel |
|---|---|---|---|---|
| MatchVerifier (rejet) | 16.4 % | 49.8 % | 8.0 % | 60.7 % |
| Summary ≠ Sirus | 31.9 % | 70.7 % | 22.2 % | 44.3 % |
| Les deux signalent | 14.3 % | 46.0 % | 6.4 % | 64.2 % |
| L'un ou l'autre signale | 34.0 % | 74.4 % | 23.9 % | 43.8 % |

### Corriger

- Sur les erreurs de Sirus, le Summary retrouve le code humain dans 37.1 % des cas (1045 paires).
- Sur les erreurs que le vérificateur détecte (520 paires) : son alternative retrouve le code humain dans 49.6 % des cas, le Summary dans 48.1 %.

Repli sur une feuille (code non terminal renvoyé) : 34.1 % des libellés. Durée moyenne : 38.0 s par libellé.

## Coût

Durée moyenne par paire : 19.9 s ; temps d'inférence cumulé : 49.2 h.
