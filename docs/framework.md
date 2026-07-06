# Documentation technique du framework GRAAL

> **Statut : document en cours de construction.** Cette première version pose les bases d'une documentation détaillée du framework (semaine du 6 juillet 2026, cf. [`docs/cadrage_2026-07.md`](./cadrage_2026-07.md)). Elle sera enrichie au fil des semaines de juillet — les sections marquées **[à compléter]** seront traitées lors des prochaines itérations — avant d'être valorisée sous la forme d'un document de travail DMCSI.

## 1. Vue d'ensemble

GRAAL combine trois briques :

1. **Un graphe de connaissance** (Neo4j) représentant une nomenclature hiérarchique : codes, libellés, notices explicatives, relations parent/enfant.
2. **Des agents LLM à outils** (*tool-calling*, librairie [`openai-agents`](https://github.com/openai/openai-agents-python)) qui interrogent ce graphe pour raisonner sur une tâche de classification ou de génération.
3. **Des contrats d'entrée/sortie stricts** (schémas Pydantic) qui structurent les échanges entre agents et garantissent la composabilité du pipeline.

Principe de conception central : **le framework ne connaît rien de la nomenclature métier**. Toute la connaissance (codes, hiérarchie, notices) vit dans la base Neo4j ; le code Python ne fait que peupler cette base et fournir des outils génériques de navigation/interrogation. Changer de nomenclature (NAF → COICOP, par exemple) ne nécessite donc pas de modifier le code des agents, seulement de reconstruire le graphe avec un nouveau jeu de notices.

## 2. Modèle de données Neo4j

Le graphe est construit par `src/neo4j_graph/graph_builder/build_graph_db.py` à partir d'un fichier de notices au format parquet (une ligne par code de nomenclature).

### 2.1 Colonnes sources (`COLUMNS_TO_KEEP`, `graph_builder/config/config.py`)

| Colonne | Rôle |
|---|---|
| `ID` | Identifiant unique du nœud |
| `CODE` | Code de nomenclature (ex. `62.01`, `J`) |
| `NAME` | Libellé du code |
| `PARENT_ID` / `PARENT_CODE` | Référence au nœud parent dans la hiérarchie |
| `LEVEL` | Niveau hiérarchique (0 = racine) |
| `FINAL` | Indique si le code est un code terminal (feuille) |
| `Implementation_rule` | Règle d'affectation officielle du code |
| `Includes` / `IncludesAlso` | Contenu inclus dans le code (notice officielle) |
| `Excludes` | Contenu explicitement exclu du code |
| `text_content` | Texte brut de la notice |

Le texte utilisé pour le calcul des embeddings (`text_to_embed`) est la concaténation de `NAME`, `Implementation_rule`, `Includes` et `IncludesAlso`.

### 2.2 Nomenclatures déjà paramétrées

Le pipeline de construction a déjà été testé/configuré pour plusieurs jeux de notices (`graph_builder/config/config.py`) :

- NAF 2025 (français) — nomenclature de référence actuelle du projet ;
- NACE Rev2.1 (anglais) ;
- COICOP 2018 (français et anglais).

**[à compléter]** : schéma exact des relations Neo4j (type `HAS_PARENT`, propriétés indexées), procédure de reconstruction complète du graphe pas à pas, gestion des mises à jour de nomenclature (recodage d'un graphe existant vers une nouvelle version).

## 3. Architecture des agents

### 3.1 `BaseAgent` (`src/agents/base_agent.py`)

Classe abstraite dont héritent tous les agents. Elle encapsule :

- la connexion à un client LLM compatible OpenAI (`OPENAI_BASE_URL` / `OPENAI_API_KEY`), avec tracing désactivé au niveau du SDK `agents` (le traçage applicatif passe par Langfuse, voir §5) ;
- le modèle utilisé pour la génération (`GENERATION_MODEL`), avec une température fixée à 0 par défaut (`get_model_settings`) ;
- un contrat commun : chaque sous-classe doit définir un nom d'agent (`get_agent_name`), des instructions système (`get_instructions`), un type de sortie structuré (`get_output_type`, un modèle Pydantic) et une méthode de construction du prompt (`build_prompt`) ;
- l'exécution (`__call__`) via `Runner.run` du SDK `agents`, avec un nombre maximal de tours (`MAX_TURNS`).

Chaque agent = **un prompt + un jeu d'outils + un contrat de sortie typé**. C'est cette homogénéité qui permet de composer des agents entre eux sans coder de logique de parsing ad hoc.

### 3.2 Outils partagés (`Graph.get_tools`, `src/neo4j_graph/graph.py`)

Un socle d'outils Neo4j est exposé à tous les agents via `@function_tool` (SDK `agents`) :

- `get_code_information(code)` — fiche complète d'un code (nom, niveau, description, inclusions/exclusions, parent, enfants) ;
- `get_children(code)` / `get_siblings(code)` / `get_descendants(code, levels)` — navigation locale dans la hiérarchie ;
- mise en cache (`functools.lru_cache`) des requêtes Neo4j les plus fréquentes, avec (dé)sérialisation dict ↔ tuple pour rendre les résultats hashables (`_freeze_dict` / `_unfreeze_dict`).

### 3.3 Le *Navigator* (`src/navigator/navigator.py`)

Le *Navigator* hérite de `Graph` et ajoute un **état de position courante** dans la hiérarchie (`current_code`) ainsi qu'un historique de navigation (`history`). Il expose un jeu d'outils dédié, avec état :

- `get_current_information` / `get_code_information(code)` — information sur la position courante ou sur un code arbitraire (sans déplacement) ;
- `get_current_children` / `get_current_siblings` / `get_current_parent` — exploration locale relative à la position courante ;
- `navigate_to(code)` / `go_to_child(child_code)` / `go_to_parent()` — déplacement dans la hiérarchie, avec validation (un `go_to_child` vers un code qui n'est pas un enfant direct échoue explicitement) ;
- `reset_to_root()` — réinitialisation avant une nouvelle requête.

Chaque appel d'outil est journalisé (position avant/après, données renvoyées au LLM), ce qui fournit une trace complète et rejouable du raisonnement de l'agent.

### 3.4 Classifieurs (`src/agents/Text2Code/`)

- `BaseClassifier` — spécialise `BaseAgent` en fixant le type de sortie à `MatchVerificationInput` (activité, code proposé, explication, confiance), le format commun attendu par les agents « closers ».
- `NavigatorAgenticClassifier` — classifieur concret : instructions demandant au *Navigator* de descendre jusqu'à un code terminal (`is_final = 1`) en justifiant chaque choix, en démarrant systématiquement par `get_current_children()`.
- `AgenticRAGClassifier` (`agentic_rag.py`) — approche alternative : récupération des *top-k* codes les plus proches par similarité d'embedding (`Graph.get_closest_codes`, recherche vectorielle Neo4j filtrée sur les codes finaux), puis arbitrage par l'agent `CodeChooser`. Branché dans la CLI via `--agentic-rag` ; le nombre de candidats est réglable par la variable d'environnement `AGENTIC_RAG_TOP_K` (défaut : 5).

### 3.5 Agents « closers » (`src/agents/closers/`)

Agents de validation, appelés en fin de chaîne :

- **`CodeChooser`** — arbitre entre plusieurs codes candidats pour une activité donnée ; sortie : code choisi, niveau de confiance, explication.
- **`MatchVerifier`** — vérifie qu'une correspondance libellé ↔ code proposée est valide ; sortie : booléen de validité, confiance, explication. C'est cet agent qui porte le cas d'usage « monitoring du modèle en production » (cf. cadrage §1.3).

Le chaînage classifieur → *MatchVerifier* est disponible via l'option `--verify` de la CLI : la sortie du classifieur (un `MatchVerificationInput` : activité, code proposé, explication, confiance) est passée telle quelle au *MatchVerifier*, qui rend un verdict indépendant (`is_match`, confiance, explication). Ce chaînage fonctionne en mode unitaire comme en mode batch et constitue la brique de base du cas d'usage « monitoring » (cf. cadrage §1.3). **[à compléter]** : retour d'expérience et calibrage des seuils de confiance après les premières campagnes d'évaluation.

### 3.6 Génération de données synthétiques (`src/agents/Code2Text/`, `src/agents/NaiveCode2Text/`)

Deux approches, à des stades de maturité différents :

- **`NaiveCode2Text`** — approche « classique » (non agentique) : échantillonnage aléatoire d'éléments de notice (loi géométrique, `code_retrieval/code_sampler.py`) pour construire des prompts de générations variées, plusieurs itérations testées (génération unitaire puis par lots de 10, orientation « métier » vs. « notice détaillée »). Des exemples de résultats sont disponibles dans `sample_results/`.
- **`Code2Text`** (`agent/code2text_agent.py`) — version agentique, au stade de squelette de code, non encore évaluée. **[à compléter]** une fois les premiers tests réalisés (semaine 4 de la roadmap).

## 4. Point d'entrée et CLI (`src/main.py`)

La CLI (`src/utils/parser.py`) expose deux méthodes de classification, avec vérification optionnelle :

```bash
uv run -m src.main --navigator "Boulangerie artisanale avec vente directe"
uv run -m src.main --agentic-rag "Boulangerie" --verify
uv run -m src.main --navigator --batch-file requetes.txt --experiment-name mon-experience
```

- `--navigator QUERY` — classification agentique par navigation hiérarchique (*Navigator*) ;
- `--agentic-rag QUERY` — classification par recherche vectorielle *top-k* + arbitrage `CodeChooser` (cf. §3.4) ;
- `--verify` — chaîne la prédiction dans le *MatchVerifier* pour double vérification (cf. §3.5) ;
- `--batch-file FILE` — traite un fichier de requêtes (une par ligne) avec la méthode choisie ;
- `--experiment-name` — nom d'expérience propagé au traçage Langfuse.

## 5. Module d'évaluation (`src/evaluation/`)

Socle du chantier prioritaire du mois (cf. cadrage §3.1–3.2), en trois briques :

- **`metrics.py`** — métriques pures Python (sans dépendance, testées unitairement dans `tests/`) : normalisation des codes (`"10.71C"` ≡ `"1071C"`), exactitude à la feuille, exactitude par niveau hiérarchique (préfixes : 2 = division, 3 = groupe, 4 = classe pour la NAF), taux d'échec (prédictions n'ayant pas atteint de code final, comptées comme erreurs).
- **`build_eval_set.py`** — construction du jeu d'évaluation stratifié : lecture du parquet labellisé (local ou S3/Datalab), stratification par préfixe de code (division par défaut), tirage plafonné par strate et reproductible (seed) — les strates plus petites que le plafond sont conservées en entier.
- **`run_eval.py`** — harnais de campagne : exécute une méthode (`navigator` ou `agentic-rag`) sur le jeu d'évaluation, écrit les prédictions détaillées (parquet) et le rapport de métriques (JSON). Nécessite Neo4j et l'API LLM à l'exécution.

```bash
uv run -m src.evaluation.build_eval_set --input <parquet S3/local> --output data/eval/eval_set.parquet
uv run -m src.evaluation.run_eval --eval-set data/eval/eval_set.parquet --method navigator
```

## 6. Configuration (variables d'environnement)

| Variable | Usage |
|---|---|
| `NEO4J_URL`, `NEO4J_USERNAME`, `NEO4J_PWD` | Connexion à la base de graphe |
| `OPENAI_BASE_URL`, `OPENAI_API_KEY` | Client LLM compatible OpenAI (génération) |
| `GENERATION_MODEL` | Modèle utilisé par les agents (`BaseAgent`) |
| `MAX_TURNS` | Nombre maximal de tours d'agent (boucle outil → réponse) |
| `EMBEDDING_MODEL`, `URL_EMBEDDING_API`, `MAX_TOKENS` | Modèle et service d'embedding utilisés lors de la construction du graphe |
| `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`, `AWS_ENDPOINT_URL` | Accès S3 (Datalab/Onyxia) pour les données sources et notices |

Le traçage applicatif (sessions, coûts, latence, arbre d'appels des agents) est assuré par **Langfuse** (`get_client`, `propagate_attributes`, `@observe` dans `src/main.py`).

## 7. Comment étendre GRAAL à une nouvelle nomenclature **[à compléter]**

Cette section documentera, une fois formalisée, le mode opératoire complet pour instancier GRAAL sur une nouvelle nomenclature : format attendu du fichier de notices, exécution du pipeline de construction du graphe, adaptation minimale des prompts si nécessaire.

## 8. Limites connues et dette technique

Recensées ici pour mémoire (suivi détaillé dans le document de cadrage) :

- Les composants branchés le 6/07 (classifieur *Agentic RAG* dans la CLI, chaînage `--verify`, harnais `run_eval`) sont vérifiés statiquement (lint, syntaxe, tests unitaires des métriques) mais **pas encore validés fonctionnellement** contre la base Neo4j et l'API LLM — à faire dès le retour sur l'environnement Datalab.
- La CI couvre lint, syntaxe et tests unitaires purs, mais **pas de tests d'intégration** (agents + graphe) : ils nécessiteraient un service Neo4j et un LLM de test dans le workflow.
- Le jeu d'évaluation lui-même n'est pas encore constitué ni versionné (l'outillage est prêt, il manque l'accès aux données — chantier semaine 2 de la roadmap de juillet).
- **Le projet requiert Python ≥ 3.12** (idéalement 3.13, cf. `pyproject.toml` et `.python-version`) : certains modules (ex. `prompt_builder.py`) utilisent des f-strings à guillemets imbriqués, syntaxe introduite par la PEP 701 et invalide sur des versions antérieures. Exécuter le projet avec un interpréteur plus ancien (3.11 par exemple) produit de fausses erreurs de syntaxe sur ces fichiers.

---

*Prochaine mise à jour prévue : semaine du 14 juillet 2026, à l'issue de la formalisation de la méthode et du jeu d'évaluation.*
