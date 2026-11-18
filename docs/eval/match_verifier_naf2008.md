# Évaluation du MatchVerifier sur la NAF 2008 (code de production vs annotation humaine)

**État : terminé** — 6732 / 6735 paires (100.0%), mis à jour le 2026-11-18 08:32.

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
| Exacte | 6732 | 1595 | 51.0% | 16.6% | 48.8% | 43.9% | 0.695 | 35.8% |
| Classe | 6732 | 1385 | 53.3% | 17.4% | 44.3% | 39.8% | 0.704 | 34.3% |

- **Rappel des erreurs** : part des erreurs de la production que le vérificateur rejette.
- **Rejets à tort** : part des codes justes qu'il rejette.
- **Précision à la prévalence réelle** : part des rejets qui portent sur de vraies erreurs, ramenée au taux d'erreur de la population.
- **Alternative = humain** : quand il rejette à raison, son code alternatif est-il le code humain ?

## Par type d'écart production / humain

| Type | Paires | Rejetées par le vérificateur |
|---|---|---|
| Accord exact | 5137 | 16.6% |
| Même classe, sous-classe différente | 210 | 35.7% |
| Même groupe, classe différente | 197 | 40.1% |
| Même division, groupe différent | 244 | 45.9% |
| Même section, division différente | 220 | 57.3% |
| Section différente | 724 | 58.1% |

Pour « Accord exact », le taux de rejet est un taux de rejet à tort ; pour les autres lignes, plus il est haut, mieux le vérificateur détecte ce type d'écart.

## Par mode d'attribution du code de production (vérité exacte)

| Mode | Paires | Taux d'erreur de la production | Rappel des erreurs | Rejets à tort | AUC p_match |
|---|---|---|---|---|---|
| AUTO | 4071 | 10.3% | 41.1% | 9.8% | 0.688 |
| FASTTEXT | 2661 | 40.7% | 55.3% | 32.3% | 0.625 |

## Point de fonctionnement (vérité exacte)

Le verdict `is_match` est un seuil parmi d'autres ; en seuillant `p_match` :

| Seuil (rejet si p_match < t) | Rappel des erreurs | Rejets à tort |
|---|---|---|
| 0.1 | 51.0% | 16.6% |
| 0.25 | 51.0% | 16.6% |
| 0.5 | 51.0% | 16.6% |
| 0.75 | 51.0% | 16.6% |
| 0.9 | 51.0% | 16.6% |

100.0% des `p_match` sont sous 0,1 ou au-dessus de 0,9 : le modèle tranche presque toujours avec certitude, et seuiller `p_match` ne déplace guère le point de fonctionnement. L'AUC n'en mesure donc qu'une version quasi binaire du verdict.

_Version agrégée, sans aucun libellé (données soumises au secret statistique)._

## Coût

Durée moyenne par paire : 27.7 s ; temps d'inférence cumulé : 51.8 h.
