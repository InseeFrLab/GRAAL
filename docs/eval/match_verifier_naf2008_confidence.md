# MatchVerifier NAF 2008 : erreurs selon l'indice de confiance fastText

- Run MatchVerifier : `v0.0.1-46-gb753427/qwen3-6-35b-moe`. Liasses analysées : 8415 sur 8431 (jointure jeu de test × tableau de bord × verdicts).
- IC : `projet-ape/label-studio/annotation-campaign-2024/NAF2008/dashboard/current-model` (`Response.IC`), re-prédiction du modèle courant au moment de la campagne. Analyse par liasse ; IC 95 % par bootstrap en grappes de paires (1000 tirages).
- Score MatchVerifier : `alternative_score − match_score`. Score combiné : somme des rangs des deux scores.

## Concordance du modèle du tableau de bord avec la production

| Mode | Liasses | IC médian | Top-1 = `apet_finale` | Erreur de production si top-1 = `apet_finale` | si top-1 ≠ `apet_finale` |
|---|---|---|---|---|---|
| `AUTO` | 5638 | 0,99 | 99,2% | 9,8% | 76,7% |
| `FASTTEXT` | 2777 | 0,35 | 64,0% | 31,9% | 56,4% |

## Mode `FASTTEXT`

### Par tranche d'IC

| IC | Liasses | Erreurs de production | Rejets | Erreurs détectées | Rejets à tort | Précision |
|---|---|---|---|---|---|---|
| [0,0 ; 0,2[ | 922 | 47,4% | 47,4% | 60,2% | 35,9% | 60,2% |
| [0,2 ; 0,4[ | 589 | 45,8% | 43,3% | 55,9% | 32,6% | 59,2% |
| [0,4 ; 0,6[ | 523 | 37,3% | 39,8% | 52,3% | 32,3% | 49,0% |
| [0,6 ; 0,8[ | 625 | 31,0% | 34,7% | 48,5% | 28,5% | 43,3% |
| [0,8 ; 1,0] | 118 | 29,7% | 22,0% | 40,0% | 14,5% | 53,8% |
| **Ensemble** | **2777** | **40,7%** | **41,2%** | **55,2%** | **31,5%** | **54,6%** |

### Pouvoir de détection des erreurs de production (AUC)

| Score | AUC [IC 95 %] |
|---|---|
| IC fastText (plus bas = plus suspect) | 0,59 [0,56 ; 0,61] |
| MatchVerifier | 0,66 [0,64 ; 0,68] |
| Combiné | 0,66 [0,64 ; 0,68] |
| Différence MatchVerifier − IC | +0,071 [+0,041 ; +0,102] |
| Différence combiné − MatchVerifier | +0,003 [-0,012 ; +0,019] |

Corrélation des rangs des deux scores : 0,13.

### Signaux comparés

| Signal | Liasses signalées | Erreurs détectées | Rejets à tort | Précision |
|---|---|---|---|---|
| Rejet MatchVerifier | 41,2% | 55,2% | 31,5% | 54,6% |
| IC le plus bas, même volume | 41,2% | 48,5% | 36,1% | 48,0% |
| Score combiné, même volume | 41,2% | 54,0% | 32,3% | 53,5% |
| IC < 0,4 | 54,4% | 62,5% | 48,8% | 46,8% |
| Rejet **et** IC < 0,4 | 24,9% | 36,6% | 16,9% | 59,8% |
| Rejet **ou** IC < 0,4 | 70,7% | 81,1% | 63,5% | 46,7% |

Erreurs de production repérées par le seul rejet : 210 ; par le seul IC < 0,4 : 293 ; par les deux : 414 ; par aucun : 214.

## Mode `AUTO`

### Par tranche d'IC

| IC | Liasses | Erreurs de production | Rejets | Erreurs détectées | Rejets à tort | Précision |
|---|---|---|---|---|---|---|
| [0,0 ; 0,2[ | 5 | 40,0% | 0,0% | 0,0% | 0,0% | – |
| [0,2 ; 0,4[ | 3 | 0,0% | 33,3% | – | 33,3% | 0,0% |
| [0,4 ; 0,6[ | 19 | 26,3% | 31,6% | 60,0% | 21,4% | 50,0% |
| [0,6 ; 0,8[ | 77 | 39,0% | 29,9% | 56,7% | 12,8% | 73,9% |
| [0,8 ; 1,0] | 5534 | 9,8% | 12,3% | 45,0% | 8,8% | 35,9% |
| **Ensemble** | **5638** | **10,3%** | **12,6%** | **45,5%** | **8,8%** | **37,2%** |

### Pouvoir de détection des erreurs de production (AUC)

| Score | AUC [IC 95 %] |
|---|---|
| IC fastText (plus bas = plus suspect) | 0,71 [0,68 ; 0,75] |
| MatchVerifier | 0,75 [0,71 ; 0,79] |
| Combiné | 0,77 [0,74 ; 0,81] |
| Différence MatchVerifier − IC | +0,037 [-0,012 ; +0,089] |
| Différence combiné − MatchVerifier | +0,022 [-0,004 ; +0,047] |

Corrélation des rangs des deux scores : 0,38.

### Signaux comparés

| Signal | Liasses signalées | Erreurs détectées | Rejets à tort | Précision |
|---|---|---|---|---|
| Rejet MatchVerifier | 12,6% | 45,5% | 8,8% | 37,2% |
| IC le plus bas, même volume | 12,6% | 32,6% | 10,3% | 26,7% |
| Score combiné, même volume | 12,6% | 43,1% | 9,1% | 35,3% |
| IC < 0,4 | 0,1% | 0,3% | 0,1% | 25,0% |
| Rejet **et** IC < 0,4 | 0,0% | 0,0% | 0,0% | 0,0% |
| Rejet **ou** IC < 0,4 | 12,8% | 45,9% | 8,9% | 37,1% |

Erreurs de production repérées par le seul rejet : 265 ; par le seul IC < 0,4 : 2 ; par les deux : 0 ; par aucun : 315.
