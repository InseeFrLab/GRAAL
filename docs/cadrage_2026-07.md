# Document de cadrage — Projet GRAAL

---

## 1. Description du projet

**GRAAL** (*Graph-based Reasoning Agents for Automatic Labelling*) est un framework qui combine :

- une **base de données graphe** (Neo4j) représentant une nomenclature statistique hiérarchique (codes, libellés, notices explicatives, relations parent/enfant) ;
- des **agents LLM à outils** (*tool-calling*), qui interrogent et parcourent ce graphe pour raisonner sur la classification d'un texte libre vers un code de nomenclature.

Le projet est actuellement instancié sur la **NAF** (Nomenclature d'Activités Française, ~700 codes terminaux, 5 niveaux hiérarchiques) toutefois le pipeline a vocataion à être générique et la construction du graphe est déjà paramétrée pour d'autres nomenclatures (NACE en anglais, COICOP FR/EN).

### 1.1 Objectifs

1. **S'affranchir de la dépendance aux données labellisées.** Le modèle de production actuel (classification supervisée par *deep learning*, package `torchTextClassifiers`) est frugal et performant, mais nécessite 100k à 1M d'observations labellisées pour être entraîné — un pré-requis qui n'est pas toujours disponible (nouvelle nomenclature, nouveau domaine, etc.).
2. **Introduire du raisonnement et de la traçabilité.** Contrairement à une approche par similarité d'embeddings (RAG « plat »), un agent doit pouvoir justifier chaque étape de sa décision (*chain-of-thought*) et exploiter explicitement la structure hiérarchique de la nomenclature plutôt que de comparer des vecteurs.
3. **Généricité.** Le framework ne doit rien connaître de la nomenclature métier : il suffit de peupler une base Neo4j (codes, libellés, notices, hiérarchie) pour que les agents fonctionnent. GRAAL se positionne comme un « distributeur » d'outils, de prompts et de contrats d'appel LLM (validation stricte des entrées/sorties via Pydantic).

### 1.2 Enjeux

- **Enjeu métier** : disposer d'une solution de codification qui fonctionne (a) quand on n'a pas ou peu de données labellisées, (b) quand une nomenclature est révisée par le métier et qu'il faut recoder l'historique, (c) quand les données labellisées existantes sont de qualité incertaine et doivent être fiabilisées/corrigées.
- **Enjeu MLOps** : la chaîne de production actuelle (stockage S3/Datalab, entraînement distribué Argo Workflows, *serving* FastAPI conteneurisé) est mature sur les volets Data, Modèle et Déploiement, mais **le monitoring en production reste le point faible** : sans annotation humaine continue, il n'existe pas aujourd'hui de moyen de contrôler la dérive du modèle déployé. GRAAL est une piste pour combler ce manque.
- **Enjeu méthodologique** : un premier prototype zero-shot par RAG a été testé et a révélé plusieurs limites structurelles (dépendance au découpage des notices, dépendance au modèle d'embedding, saturation du contexte, hétérogénéité de la comparaison notice/libellé, absence de traçabilité, non prise en compte de la hiérarchie). Ces limites motivent directement le choix de l'approche agentique.

### 1.3 Positionnement par rapport aux besoins de codification automatique

GRAAL **ne vise pas à remplacer** le modèle supervisé en production : une architecture agentique multi-appels LLM est aujourd'hui trop coûteuse en latence et en ressources de calcul (pas de GPU dédié, temps d'inférence incompatible avec un usage de masse en production). GRAAL se positionne comme une **brique complémentaire**, mobilisable sur trois cas d'usage bien délimités :

| Cas d'usage | Description | Composant GRAAL mobilisé |
|---|---|---|
| **Monitoring du modèle en production** | Analyser en continu (ou en échantillonnage) les prédictions du modèle supervisé et détecter les codifications douteuses, sans annotation humaine | *MatchVerifier* (vérifie une correspondance libellé ↔ code), avec appel au *Navigator* pour proposer un recodage si besoin |
| **Recodage d'une base existante** | Recoder un stock de données suite à une révision de nomenclature, ou fiabiliser un historique de labels incertains | *Navigator* (classification hiérarchique agentique) |
| **Génération de données d'entraînement synthétiques** | Produire des libellés synthétiques par code, pour densifier les classes rares et ré-entraîner/améliorer le modèle supervisé | *Code2Text* / *NaiveCode2Text* |

Ce positionnement conditionne la roadmap : la priorité n'est pas de « battre » le modèle supervisé en production, mais de **livrer une évaluation chiffrée et honnête** de ce que chacun de ces trois cas d'usage peut apporter concrètement.

---

## 2. État d'avancement au 6 juillet 2026

### 2.1 Ce qui fonctionne

- **Le graphe Neo4j de la nomenclature** est opérationnel de bout en bout : construction depuis un fichier de notices (parquet), calcul et stockage des embeddings, relations hiérarchiques `HAS_PARENT`. Le pipeline de construction (`src/neo4j_graph/graph_builder`) est déjà paramétré pour plusieurs nomenclatures (NAF, NACE EN, COICOP FR/EN), preuve concrète de la généricité recherchée.
- **L'agent *Navigator*** (classification hiérarchique agentique) fonctionne en usage individuel et en usage batch (fichier de requêtes), avec traçage complet des appels via Langfuse (observabilité, sessions, coûts). Il expose un ensemble d'outils Neo4j génériques et réutilisables (`get_current_children`, `get_current_siblings`, `go_to_parent`, `go_to_child`, `navigate_to`, etc.), communs à tous les agents du framework.
- **Les agents « closers »** (*CodeChooser* : choix argumenté parmi K codes candidats ; *MatchVerifier* : validation d'une correspondance libellé/code) sont implémentés et fonctionnels **isolément**, avec des contrats de sortie stricts (schémas Pydantic, score de confiance, explication).
- **Une première chaîne de génération de données synthétiques** (*NaiveCode2Text*) produit des libellés synthétiques exploitables à partir des notices officielles (plusieurs itérations de prompt testées : zero-shot 1-par-1, puis génération par lots de 10, orientation « métier »), avec des exemples de résultats déjà générés et sauvegardés.
- **Une CLI** (`src/main.py`) permet de lancer une classification unitaire ou en lot avec le *Navigator*.

### 2.2 Ce qui ne fonctionne pas encore

- **Le classifieur « Agentic RAG »** (approche alternative combinant recherche par embeddings + *CodeChooser*) n'est pas branché dans le pipeline principal : `main.py` appelle aujourd'hui une fonction *stub* (`classify_agentic_rag`) qui renvoie une valeur codée en dur. → **Corrigé le 6/07** : le classifieur est branché dans la CLI (`--agentic-rag`) ; validation fonctionnelle sur la base Neo4j à faire. **Redesigné le 8/07** : l'approche n'arbitre plus entre des candidats via *CodeChooser*, mais utilise le code le plus proche par similarité d'embeddings comme simple *point de départ* (*warm start*), que l'agent *Navigator* vérifie et corrige ensuite via ses outils habituels (cf. §2.2 ci-dessous et `docs/framework.md` §5.2) — voir aussi le résultat préoccupant du diagnostic de l'espace d'embedding qui sous-tend ce point de départ.
- **Le chaînage *Navigator* → *MatchVerifier*** n'est pas encore implémenté : le *Navigator* s'arrête aujourd'hui dès qu'il atteint un code terminal, sans double vérification automatique par un second agent. C'est pourtant l'articulation nécessaire au cas d'usage « monitoring en production ». → **Implémenté le 6/07** via l'option `--verify` de la CLI (en mode unitaire comme en batch) ; validation fonctionnelle sur données réelles à faire.
- **Aucune méthodologie d'évaluation n'est encore formalisée** : ni jeu de données de référence versionné, ni métriques automatisées, ni tableau de bord de suivi. À ce stade, l'évaluation n'existe qu'à l'état de script exploratoire ponctuel (projection UMAP/PaCMAP/t-SNE/PCA des embeddings, calcul de k-plus-proches-voisins). C'est le chantier prioritaire du mois (cf. §3). → **Socle posé et jeu d'évaluation construit le 6/07** : module `src/evaluation/` (métriques d'exactitude à la feuille et par niveau hiérarchique testées unitairement, échantillonnage stratifié reproductible, harnais de campagne) ; **`data/eval/eval_set.parquet` est constitué et versionné** (5 181 lignes, stratifié par code complet `apet2025`, ~10 exemples/code). Restent à mener : les campagnes chiffrées elles-mêmes (S3).
- **L'espace d'embedding utilisé par l'Agentic RAG comme *warm start* n'avait jamais été évalué formellement**, seulement via le script exploratoire ponctuel ci-dessus (`explorations.py`). → **Diagnostic formalisé le 8/07** : `evaluate_embeddings.py` (script autonome, hors `src/evaluation/`) calcule l'accuracy@1 / recall@5 d'un k-NN cosinus libellé → notice sur `data/eval/eval_set_sample15.parquet` / `eval_set_sample30.parquet`, avec projections UMAP/PaCMAP/t-SNE/PCA et comparaison multi-modèles. **Premier résultat préoccupant** : avec le modèle actuellement configuré (`qwen3-embedding-8b`), accuracy@1 = recall@5 = 0 % sur l'échantillon de 30 libellés (ex. « FOOTBALL FEMININ » classé 373ᵉ/747 par similarité cosinus à sa propre notice). Cause probable non tranchée (modèle non adapté au domaine, ou format d'instruction spécifique attendu par ce modèle différent du préfixe générique `"query : "` câblé dans `Graph.get_closest_codes`) — à investiguer avant de considérer l'Agentic RAG comme fiable.
- **Le modèle supervisé de production n'était pas mobilisable comme référence de comparaison** : aucun wrapper ne l'exposait avec le même contrat que les classifieurs agentiques. → **Ajouté le 6/07** : `SupervisedClassifier` (`src/agents/Text2Code/classifiers/supervised_classifier.py`) charge le modèle (`torchTextClassifiers`) via MLflow pyfunc et l'expose en `--supervised` dans la CLI et dans `run_eval.py` ; **non validé fonctionnellement** (format de sortie du modèle réel non vérifiable dans cet environnement).
- **La détection de dérive (§3.3-C) n'existait qu'à l'état d'idée.** → **Prototype ajouté le 6/07** : `src/evaluation/drift.py` (Wasserstein, PSI, Kolmogorov-Smirnov, seuils calibrés empiriquement par rééchantillonnage), testé sur données synthétiques (dérive simulée). → **Retiré le 15/07** dans le cadre de la simplification du dépôt (fonctionnalité abandonnée, pas seulement le test).
- **La génération synthétique agentique** (*Code2Text*, par opposition à l'approche *Naive*) est au stade de squelette de code, non encore évaluée ni comparée à l'approche naïve.
- **Aucune intégration continue (CI) ne teste le code applicatif** : les deux workflows GitHub existants ne font que déployer les slides de présentation et les embeddings, sans exécuter de tests. → **Corrigé le 6/07** : workflow CI ajouté (lint `ruff`, vérification de syntaxe, tests unitaires).
- **Pas encore de documentation technique détaillée du framework**, au-delà du README et de la présentation de démonstration — c'est l'objet du chantier lancé cette semaine (cf. §4).
- **Le traçage Langfuse était incomplet** (audit du 8/07, cf. `docs/framework.md` §6.1) : `classify_navigator` (chemin agentique principal) n'était pas tracé (`@observe` commenté), le repli de finalisation ajouté à `_run_navigator_loop` masquait les échecs plutôt que de les faire remonter en erreur dans les traces, `--experiment-name` n'était pas réellement attaché aux traces Langfuse malgré ce qu'indiquait `framework.md` §4, et les variables `LANGFUSE_*` n'étaient pas documentées (documentation corrigée le 8/07). → **Correctifs de code faits le 9/07** : `@observe` ré-activé sur `classify_navigator`, échecs de finalisation remontés en `ERROR` sur le span courant, `--experiment-name` propagé à la trace (nom/tag/métadonnée) dans les quatre points d'entrée. Reste à faire : `langfuse.flush()`/`shutdown()` explicite avant sortie CLI (cf. `docs/framework.md` §6.1).
- **La boucle du *Navigator* (`_run_navigator_loop`, `BaseClassifier`) pouvait rester bloquée ou renvoyer silencieusement un code non terminal** : pas de mécanisme pour sortir d'une répétition du même appel d'outil, pas de garantie que la finalisation renvoie bien une feuille une fois le budget de pas épuisé, et un `try/except` limité à la finalisation qui laissait un échec pendant l'exploration remonter tel quel. → **Durci le 15/07** (cf. `docs/framework.md` §3.1 et §3.4) : relance à température plus élevée sur appel répété, descente forcée vers une feuille (outils restreints à `go_to_child`) quand l'exploration stagne sur une position non terminale, filet de sécurité rejetant toute sortie de finalisation non terminale, `try/except` élargi à toute la boucle, et timeout client LLM plafonné à 60 s (contre 10 min par défaut) pour échouer vite sur un endpoint figé. `run_eval.py` a été mis en cohérence : un échec de classification non intercepté par le classifieur lui-même (cas de `SummaryAgenticClassifier`, qui n'a pas de garde-fou Python équivalent) est désormais enregistré comme une prédiction d'échec au lieu de faire échouer toute la campagne.

### 2.3 Synthèse

L'architecture cible (graphe + agents génériques + closers de validation) est posée et sa faisabilité technique est démontrée sur le cas *Navigator*. Le mois de juillet doit servir à **consolider la base de code** (corriger le bug bloquant, brancher les composants manquants), puis, surtout, à **construire une évaluation chiffrée et reproductible**, condition nécessaire pour statuer sur la valeur ajoutée réelle de chacun des trois cas d'usage identifiés.

---

## 3. Roadmap détaillée — juillet 2026

### 3.1 Méthode d'évaluation envisagée

Méthode retenue pour l'évaluation du classifieur (*Navigator*, et *Agentic RAG* une fois réparé), comme identifié dès la présentation initiale du projet :

- **Si le jeu de test est jugé fiable** : évaluation automatique classique — exactitude (*accuracy*) au code terminal, et par niveau de hiérarchie (section, division, groupe, classe) pour distinguer une erreur « proche » (bon niveau supérieur, mauvaise feuille) d'une erreur « lointaine ».
- **Si le jeu de test n'est pas jugé fiable** (label bruité, notices ambiguës) : évaluation manuelle sur échantillon, en mettant bout à bout la prédiction *ground truth*, la prédiction du classifieur, et les jugements du *CodeChooser* et du *MatchVerifier* — évaluation conjointe de toute la chaîne plutôt que du seul classifieur.
- Métriques complémentaires : taux de requêtes n'atteignant **pas** un code terminal (`is_final = 0`), nombre moyen d'étapes de navigation, taux d'accord entre *Navigator* et *MatchVerifier*, comparaison au modèle supervisé de référence sur le même échantillon.

### 3.2 Jeu d'évaluation

- **Source de référence** : réutiliser le jeu de test déjà utilisé pour le modèle supervisé de production (split `df_test` du projet APE, colonnes `libelle` / `apet2025`), déjà mobilisé de façon exploratoire dans `explorations.py`. Cela permet une comparaison directe et un jeu de labels dont la qualité est déjà connue.
- **Construction d'un échantillon d'évaluation stratifié** : tirage représentatif par code (pas uniquement aléatoire), pour garantir une couverture des cas rares et des codes à fort volume. → **Fait le 6/07** : `data/eval/eval_set.parquet`, 5 181 lignes, ~10 exemples par code complet.
- **Isolation d'un sous-échantillon à ré-annoter manuellement** : pour les cas où le label historique est jugé incertain, base du protocole d'évaluation manuelle décrit en §3.1.
- Le jeu d'évaluation et son protocole de constitution sont versionnés dans le dépôt (et documentés dans `docs/framework.md`, cf. §4) pour être réutilisables et auditable.

### 3.3 Pistes à explorer

**A. Fiabilisation et industrialisation du framework**
- **Nettoyage du dépôt** : suppression du code mort, structuration des scripts exploratoires (`explorations.py`), mise en cohérence des modules d'agents, mise en place d'une CI minimale (lint, vérification d'import) pour éviter que des fichiers cassés comme ceux identifiés en §2.2 ne passent inaperçus.
- Réparation et **branchement effectif du chaînage *Navigator* → *MatchVerifier*** (nécessaire au cas d'usage monitoring).
- **Réparation du traçage Langfuse** (audit du 8/07, cf. `docs/framework.md` §6.1) : ré-activer `@observe` sur `classify_navigator`, faire remonter en erreur les échecs de finalisation capturés par `_fallback_output`, propager réellement `--experiment-name` à la trace (nom/tag/métadonnée plutôt que simple argument de log). → **Fait le 9/07** ; reste `langfuse.flush()`/`shutdown()` explicite avant sortie CLI.
- **Automatisation de la pipeline d'évaluation via Argo Workflows** (déjà utilisé pour l'entraînement du modèle supervisé de production) : exécuter le jeu d'évaluation de façon reproductible et programmée, avec des métriques versionnées à chaque évolution du framework, plutôt que des lancements manuels ponctuels.

**B. Diagnostic et évaluation**
- **Analyse des embeddings** : formaliser les explorations déjà amorcées (projections UMAP/PaCMAP/t-SNE/PCA, k-plus-proches-voisins dans `explorations.py`) en un diagnostic reproductible de la qualité de l'espace d'embedding utilisé par l'*Agentic RAG* — séparabilité des codes, zones de confusion, sensibilité au modèle d'embedding choisi. → **Formalisé le 8/07** (cf. §2.2) : `evaluate_embeddings.py`. Reste à faire : comprendre et corriger la cause du résultat quasi nul obtenu avec le modèle actuellement configuré.
- Comparaison **plusieurs modèles LLM** sous-jacents (le framework est agnostique au modèle via une API compatible OpenAI — variable d'environnement `GENERATION_MODEL`), pour arbitrer coût/latence/qualité.
- Étude du **nombre d'étapes de navigation et du taux d'échec** (requêtes n'atteignant pas de code terminal) en fonction de la formulation du prompt et des instructions du *Navigator*.
- **Rigueur statistique de la comparaison** : ne pas se limiter à un delta d'exactitude entre méthodes (*Navigator*, *Agentic RAG*, modèle supervisé) — calculer un intervalle de confiance (bootstrap sur le jeu d'évaluation) et un test de significativité avant de conclure qu'une méthode surpasse une autre, en particulier sur les strates à faible effectif (§3.2).
- **Analyse d'erreurs par segment** : ventiler l'exactitude et le taux d'échec par section/niveau de nomenclature plutôt que de s'arrêter à un score agrégé, pour identifier *où* et *pourquoi* chaque méthode échoue (ambiguïté de notice, code trop spécifique, hiérarchie profonde, etc.).
- **Note de conception (architecture) *Navigator* vs *Agentic RAG* vs modèle supervisé** : livrable écrit documentant les compromis mesurés (exactitude par niveau, latence, coût par requête, nombre d'appels LLM) et la recommandation d'usage qui en découle par cas d'usage — formalise la décision de conception plutôt que de la laisser implicite.

**C. Monitoring en production**
- Premiers tests du cas d'usage **monitoring** : simulation d'un flux de prédictions du modèle de production, détection par *MatchVerifier* des cas à recoder.
- **Détection de dérive de distribution**, non supervisée, en amont ou en complément de *MatchVerifier* — répond directement à l'enjeu « comment contrôler le modèle en production sans annotation humaine continue ? » (§1.2). Plan de travail :
  1. implémenter et comparer plusieurs métriques de dérive sur les distributions de prédictions/embeddings dans le temps : **distance de Wasserstein**, **Population Stability Index (PSI)** et **test de Kolmogorov-Smirnov** — trois métriques usuelles en surveillance de modèles de risque (proches de ce qui est attendu dans un cadre de gouvernance de modèle type SR 11-7) ; → **Fait le 6/07** : `src/evaluation/drift.py`. → **Retiré le 15/07** dans le cadre de la simplification du dépôt (fonctionnalité abandonnée, pas seulement le test).
  2. calibrer des seuils d'alerte sur des fenêtres temporelles glissantes, à partir de l'historique de prédictions du modèle de production ; → **Fait le 6/07** pour la calibration (empirique, par rééchantillonnage de la référence) ; **abandonné le 15/07** avec le retrait de `drift.py`.
  3. valider sur un cas de dérive simulée (ex. injection artificielle d'un décalage de distribution) avant tout test sur données réelles. → **Fait le 6/07** (tests unitaires avec dérive synthétique injectée) ; **abandonné le 15/07** avec le retrait de `drift.py`, cette piste n'est plus poursuivie.

**D. Génération de données et industrialisation du modèle agentique**
- Premiers tests du cas d'usage **génération synthétique pour ré-entraînement**, avec une métrique proxy (gain de performance du modèle supervisé sur les classes enrichies).
- **LoRA / QLoRA** : explorer le fine-tuning léger d'un modèle plus petit sur les trajectoires de raisonnement du *Navigator* (ou sur les données synthétiques de *Code2Text*), pour réduire le coût et la latence d'inférence — piste qui répondrait directement à la limite actuelle « un modèle agentique est trop lourd pour la production » (§1.3). Piste plus exploratoire, à horizon au-delà de juillet mais à garder en ligne de mire.

### 3.4 Objectifs hebdomadaires — juillet 2026

| Semaine | Dates | Objectifs |
|---|---|---|
| **S1** | 6 → 11 juillet | Corriger le bug bloquant (`agentic_rag.py`) et **nettoyer le dépôt** (code mort, artefacts de build committés par erreur, structuration des scripts exploratoires, CI minimale de lint/import) pour fiabiliser la base de code. Poser les bases de la **documentation détaillée du framework** (architecture, modèle de données, contrats des agents — voir §4), première brique valorisable en document de travail DMCSI. **Réparer le traçage Langfuse** (audit du 8/07, cf. §3.3-A) : tracer `classify_navigator`, faire remonter les échecs de finalisation, propager `--experiment-name` aux traces, documenter les variables `LANGFUSE_*` — ✅ fait le 9/07 (documentation le 8/07, correctifs de code le 9/07) ; reste `langfuse.flush()`/`shutdown()` explicite. |
| **S2** | 14 → 18 juillet | Formaliser la méthode d'évaluation (§3.1) et constituer le **jeu d'évaluation stratifié** (§3.2), versionné dans le dépôt — ✅ fait le 6/07 (`data/eval/eval_set.parquet`) — en s'appuyant sur un premier **diagnostic de l'espace d'embedding** — ✅ formalisé le 8/07 (`evaluate_embeddings.py`, cf. §2.2), résultat préoccupant à creuser. Brancher le classifieur *Agentic RAG* dans le pipeline principal — ✅ fait le 6/07. Nettoyage du dépôt (drift.py, scripts morts) — ✅ fait le 15/07. **Durcissement de la boucle *Navigator*** contre les blocages et les réponses non terminales silencieuses, et mise en cohérence de `run_eval.py` — ✅ fait le 15/07 (cf. §2.2). |
| **S3** | 21 → 25 juillet | Implémenter le chaînage *Navigator* → *MatchVerifier* — ✅ fait le 6/07. Lancer les **premières campagnes d'évaluation chiffrées** du *Navigator*, de l'*Agentic RAG* **et du modèle supervisé de référence** (`--supervised`, ✅ branché le 6/07, non validé fonctionnellement) sur le jeu d'évaluation, assorties d'intervalles de confiance et d'une analyse d'erreurs par segment (§3.3-B). Amorcer un **prototype d'automatisation de la pipeline d'évaluation via Argo Workflows**. Rédiger la **note de conception *Navigator* vs *Agentic RAG* vs supervisé** (compromis exactitude/latence/coût). |
| **S4** | 28 → 31 juillet | Évaluer la génération synthétique (*Code2Text* / *NaiveCode2Text*) via la métrique proxy de ré-entraînement. ~~**Prototype de détection de dérive** (Wasserstein, PSI, Kolmogorov-Smirnov)~~ — ✅ fait le 6/07 (`src/evaluation/drift.py`), validé sur dérive simulée ; **abandonné le 15/07** dans le cadre de la simplification du dépôt, cet objectif est retiré du périmètre de juillet. Synthèse des résultats du mois et note de recommandation sur les cas d'usage à prioriser pour la suite (monitoring, recodage, génération de données). |

> Pistes identifiées mais volontairement hors périmètre resserré de juillet, à ré-examiner ensuite : industrialisation complète (au-delà du prototype) de la pipeline Argo Workflows ; exploration **LoRA/QLoRA** pour un modèle agentique plus léger (cf. §3.3-D).

---

## 4. Documentation du framework

Un premier document technique a été créé : [`docs/framework.md`](./framework.md).
