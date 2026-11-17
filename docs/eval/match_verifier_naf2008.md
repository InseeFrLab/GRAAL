# Évaluation du MatchVerifier sur la NAF 2008 (code de production vs annotation humaine)

**État : en cours** — 1476 / 6735 paires (21.9%), mis à jour le 2026-11-17 15:20.

## Protocole

- Données : test de la campagne d'annotation 2024 (`label-studio/annotation-campaign-2024/NAF2008/preprocessed/test_data_NAF2008.parquet`), joint par liasse aux exports Label Studio pour retrouver le code attribué en production (cf. `src.evaluation.prepare_naf2008_pairs`).
- Code jugé : `apet_finale`, code de production (`AUTO` par règles ou `FASTTEXT` par le modèle). Référence : `apet_manual`, code retenu par l'annotateur. Les deux en NAF rév. 2, au niveau sous-classe.
- Graphe : NAF rév. 2 construite depuis les notes explicatives Insee (cf. `graph_builder.utils.naf2008_notices`), profil `NOMENCLATURE=naf2008`.
- Deux vérités : **exacte** (même sous-classe) et **de classe** (mêmes 4 premiers caractères : un écart de sous-classe n'y compte pas comme une erreur).
- Population : 8431 lignes, 6735 paires distinctes ; tous les codes sont dans le graphe. Taux d'erreur de la production : 20.3% (exact), 17.7% (classe).
- Toutes les paires distinctes sont vérifiées (pas d'échantillonnage) ; métriques calculées par paire distincte (libellé, code de production, code humain).

## Résultats principaux

| Vérité | Paires | dont erreurs | Rappel des erreurs | Rejets à tort | Précision du rejet | Précision à la prévalence réelle | AUC p_match | Alternative = humain (erreurs détectées) |
|---|---|---|---|---|---|---|---|---|
| Exacte | 1476 | 361 | 54.3% | 17.7% | 49.9% | 44.0% | 0.694 | 36.2% |
| Classe | 1476 | 315 | 54.9% | 18.9% | 44.0% | 38.4% | 0.691 | 33.5% |

- **Rappel des erreurs** : part des erreurs de la production que le vérificateur rejette.
- **Rejets à tort** : part des codes justes qu'il rejette.
- **Précision à la prévalence réelle** : part des rejets qui portent sur de vraies erreurs, ramenée au taux d'erreur de la population.
- **Alternative = humain** : quand il rejette à raison, son code alternatif est-il le code humain ?

## Par type d'écart production / humain

| Type | Paires | Rejetées par le vérificateur |
|---|---|---|
| Accord exact | 1115 | 17.7% |
| Même classe, sous-classe différente | 46 | 50.0% |
| Même groupe, classe différente | 51 | 39.2% |
| Même division, groupe différent | 59 | 55.9% |
| Même section, division différente | 46 | 60.9% |
| Section différente | 159 | 57.9% |

Pour « Accord exact », le taux de rejet est un taux de rejet à tort ; pour les autres lignes, plus il est haut, mieux le vérificateur détecte ce type d'écart.

## Par mode d'attribution du code de production (vérité exacte)

| Mode | Paires | Taux d'erreur de la production | Rappel des erreurs | Rejets à tort | AUC p_match |
|---|---|---|---|---|---|
| AUTO | 871 | 10.3% | 46.4% | 9.6% | 0.699 |
| FASTTEXT | 605 | 40.7% | 57.8% | 34.8% | 0.616 |

## Point de fonctionnement (vérité exacte)

Le verdict `is_match` est un seuil parmi d'autres ; en seuillant `p_match` :

| Seuil (rejet si p_match < t) | Rappel des erreurs | Rejets à tort |
|---|---|---|
| 0.1 | 54.3% | 17.7% |
| 0.25 | 54.3% | 17.7% |
| 0.5 | 54.3% | 17.7% |
| 0.75 | 54.3% | 17.7% |
| 0.9 | 54.3% | 17.7% |

_Version agrégée, sans aucun libellé (données soumises au secret statistique)._

## Coût

Durée moyenne par paire : 25.5 s ; temps d'inférence cumulé : 10.5 h.
