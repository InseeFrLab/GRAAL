# Évaluation du MatchVerifier sur la COICOP (Sirus vs annotation humaine)

**État : terminé** — 8909 / 8981 paires (99.2 %), mis à jour le 2026-10-13 15:40.

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

**État : en cours** — 1789 / 8717 libellés classés (20.5 %), 1898 paires couvertes.

Le Summary classifier reçoit le libellé seul et le résumé complet de la COICOP, et choisit un code terminal. Son code est ramené au niveau 4 pour être comparé au code humain, comme Sirus.

### Exactitude face à l'humain

| Paires | n | Sirus exact | Summary exact | Sirus branche | Summary branche | Sirus division | Summary division | Sirus groupe | Summary groupe |
|---|---|---|---|---|---|---|---|---|---|
| Toutes les paires | 1898 | 73.4 % | 51.6 % | 75.6 % | 66.0 % | 86.7 % | 82.7 % | 85.4 % | 80.5 % |
| Hors codes 98/99 | 1805 | 77.2 % | 54.2 % | 79.4 % | 69.4 % | 91.2 % | 86.9 % | 89.8 % | 84.7 % |

### Détecter les erreurs de Sirus (vérité de branche, hors 98/99)

| Signal | Lignes signalées | Erreurs détectées | Codes justes signalés | Précision au taux réel |
|---|---|---|---|---|
| MatchVerifier (rejet) | 16.5 % | 48.2 % | 8.2 % | 60.3 % |
| Summary ≠ Sirus | 32.6 % | 70.6 % | 22.7 % | 44.6 % |
| Les deux signalent | 14.1 % | 44.2 % | 6.3 % | 64.6 % |
| L'un ou l'autre signale | 35.0 % | 74.7 % | 24.7 % | 43.9 % |

### Corriger

- Sur les erreurs de Sirus, le Summary retrouve le code humain dans 34.5 % des cas (371 paires).
- Sur les erreurs que le vérificateur détecte (179 paires) : son alternative retrouve le code humain dans 49.2 % des cas, le Summary dans 45.8 %.

Repli sur une feuille (code non terminal renvoyé) : 34.3 % des libellés. Durée moyenne : 42.8 s par libellé.

## Coût

Durée moyenne par paire : 19.9 s ; temps d'inférence cumulé : 49.2 h.
