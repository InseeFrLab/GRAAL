# Évaluation du MatchVerifier sur la NAF 2008 (code de production vs annotation humaine)

**État : en cours** — 2182 / 6735 paires (32.4%), mis à jour le 2026-11-17 16:21.

## Protocole

- Données : test de la campagne d'annotation 2024 (`label-studio/annotation-campaign-2024/NAF2008/preprocessed/test_data_NAF2008.parquet`), joint par liasse aux exports Label Studio pour retrouver le code attribué en production (cf. `src.evaluation.prepare_naf2008_pairs`).
- Code jugé : `apet_finale`, code de production (`AUTO` par règles ou `FASTTEXT` par le modèle). Référence : `apet_manual`, code retenu par l'annotateur. Les deux en NAF rév. 2, au niveau sous-classe.
- Version : `v0.0.1-46-gb753427/qwen3-6-35b-moe` (`<commit>/<modèle>` sous lequel match_verifier_eval a rangé le run, cf. `src.utils.run_provenance`).
- Graphe : NAF rév. 2 construite depuis les notes explicatives Insee (cf. `graph_builder.utils.naf2008_notices`), profil `NOMENCLATURE=naf2008`.
- Deux vérités : **exacte** (même sous-classe) et **de classe** (mêmes 4 premiers caractères : un écart de sous-classe n'y compte pas comme une erreur).
- Population : 8431 lignes, 6735 paires distinctes ; tous les codes sont dans le graphe. Taux d'erreur de la production : 20.3% (exact), 17.7% (classe).
- Toutes les paires distinctes sont vérifiées (pas d'échantillonnage) ; métriques calculées par paire distincte (libellé, code de production, code humain).

## Résultats principaux

| Vérité | Paires | dont erreurs | Rappel des erreurs | Rejets à tort | Précision du rejet | Précision à la prévalence réelle | AUC p_match | Alternative = humain (erreurs détectées) |
|---|---|---|---|---|---|---|---|---|
| Exacte | 2182 | 521 | 51.1% | 16.6% | 49.2% | 44.0% | 0.693 | 35.7% |
| Classe | 2182 | 455 | 51.9% | 17.7% | 43.6% | 38.7% | 0.690 | 32.6% |

- **Rappel des erreurs** : part des erreurs de la production que le vérificateur rejette.
- **Rejets à tort** : part des codes justes qu'il rejette.
- **Précision à la prévalence réelle** : part des rejets qui portent sur de vraies erreurs, ramenée au taux d'erreur de la population.
- **Alternative = humain** : quand il rejette à raison, son code alternatif est-il le code humain ?

## Par type d'écart production / humain

| Type | Paires | Rejetées par le vérificateur |
|---|---|---|
| Accord exact | 1661 | 16.6% |
| Même classe, sous-classe différente | 66 | 45.5% |
| Même groupe, classe différente | 69 | 34.8% |
| Même division, groupe différent | 79 | 49.4% |
| Même section, division différente | 72 | 56.9% |
| Section différente | 235 | 56.2% |

Pour « Accord exact », le taux de rejet est un taux de rejet à tort ; pour les autres lignes, plus il est haut, mieux le vérificateur détecte ce type d'écart.

## Par mode d'attribution du code de production (vérité exacte)

| Mode | Paires | Taux d'erreur de la production | Rappel des erreurs | Rejets à tort | AUC p_match |
|---|---|---|---|---|---|
| AUTO | 1294 | 10.3% | 41.4% | 8.7% | 0.689 |
| FASTTEXT | 888 | 40.7% | 55.0% | 33.9% | 0.612 |

## Point de fonctionnement (vérité exacte)

Le verdict `is_match` est un seuil parmi d'autres ; en seuillant `p_match` :

| Seuil (rejet si p_match < t) | Rappel des erreurs | Rejets à tort |
|---|---|---|
| 0.1 | 51.1% | 16.6% |
| 0.25 | 51.1% | 16.6% |
| 0.5 | 51.1% | 16.6% |
| 0.75 | 51.1% | 16.6% |
| 0.9 | 51.1% | 16.6% |

_Version agrégée, sans aucun libellé (données soumises au secret statistique)._

## Coût

Durée moyenne par paire : 25.6 s ; temps d'inférence cumulé : 15.5 h.
