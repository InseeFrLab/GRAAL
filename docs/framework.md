# Documentation technique du framework GRAAL

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

### 2.3 Résumé de la nomenclature (`src/neo4j_graph/build_nace_summary.py`)

Script autonome qui aplatit la hiérarchie Neo4j (`Graph.get_summary_tree`) en un résumé texte indenté (par défaut niveau 5, soit la nomenclature complète jusqu'aux codes terminaux — code et nom seulement, pas la notice) et l'écrit dans un fichier (`data/nace_summary.txt` par défaut). Généré hors ligne, une fois par version de nomenclature — pas recalculé à chaque appel de classifieur. Sert de contexte d'orientation à `SummaryAgenticClassifier` (§3.4).

```bash
uv run -m src.neo4j_graph.build_nace_summary --max-level 5 --output data/nace_summary.txt
```

## 3. Architecture des agents

### 3.1 `BaseAgent` (`src/agents/base_agent.py`)

Classe abstraite dont héritent tous les agents. Elle encapsule :

- la connexion à un client LLM compatible OpenAI (`OPENAI_BASE_URL` / `OPENAI_API_KEY`), avec tracing désactivé au niveau du SDK `agents` (le traçage applicatif passe par Langfuse, voir §5) et un timeout de requête plafonné à 60 s (contre les 10 minutes par défaut du SDK), pour qu'une requête bloquée sur un endpoint LLM figé échoue vite plutôt que de bloquer toute la boucle appelante ;
- le modèle utilisé pour la génération (`GENERATION_MODEL`), avec une température fixée à 0 par défaut (`get_model_settings`) ;
- un contrat commun : chaque sous-classe doit définir un nom d'agent (`get_agent_name`), des instructions système (`get_instructions`), un type de sortie structuré (`get_output_type`, un modèle Pydantic) et une méthode de construction du prompt (`build_prompt`) ;
- l'exécution (`__call__`) via `Runner.run` du SDK `agents`, avec un nombre maximal de tours (`MAX_TURNS`). Cette description vaut pour `CodeChooser`/`MatchVerifier`/`SupervisedClassifier` ; `BaseClassifier` (§3.4) remplace cette boucle par sa propre boucle pas-à-pas.

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
- `navigate_to(code)` / `go_to_child(child_code)` / `go_to_parent()` — déplacement dans la hiérarchie, avec validation (un `go_to_child` vers un code qui n'est pas un enfant direct échoue explicitement) ; `go_to_child`/`go_to_parent` renvoient directement les enfants de la nouvelle position, pour fusionner « se déplacer » et « voir les options » en un seul appel ;
- `reset_to_root()` — réinitialisation avant une nouvelle requête ;
- `is_current_final()` — vérité terrain lue directement sur le graphe (jamais déduite d'une auto-évaluation du LLM), utilisée par `BaseClassifier` (§3.4) pour savoir quand arrêter l'exploration.

Chaque appel d'outil est journalisé (position avant/après, données renvoyées au LLM), ce qui fournit une trace complète et rejouable du raisonnement de l'agent.

### 3.4 Classifieurs (`src/agents/Text2Code/`)

- `BaseClassifier` — spécialise `BaseAgent` en fixant le type de sortie à `MatchVerificationInput` (activité, code proposé, explication, confiance), le format commun attendu par les agents « closers ». Remplace la boucle unique de `BaseAgent` par une boucle pas-à-pas pilotée en Python (`_run_navigator_loop`) : un seul `Runner.run` piloté par le LLM ne peut pas à la fois utiliser les outils de façon fiable et savoir quand s'arrêter (le SDK ne réinitialise `tool_choice` que sur « un outil a été utilisé », sans notion du critère métier `is_final`). La boucle alterne donc entre deux variantes d'`Agent` (`Agent.clone()`) : un agent d'exploration (outils forcés via `tool_choice="required"`, sans `output_type`) et un agent de finalisation (`tool_choice="none"`, outils gardés déclarés pour éviter un blocage du rendu du chat-template côté serveur, `output_type=MatchVerificationInput`). L'arrêt est décidé par `Navigator.is_current_final()` et seulement juste après un déplacement réel (jamais après une simple consultation), pour éviter qu'une position de départ RAG erronée soit « vérifiée » puis renvoyée telle quelle. **Durcissement du 15/07** contre les blocages et les réponses non terminales silencieuses : si le modèle répète exactement le même appel d'outil d'une étape à l'autre (aucune progression), la boucle relance une fois à température plus élevée avec une consigne explicite (`_step`) ; si le budget de pas est épuisé sur une position non terminale, un agent dédié (`forced_descent_agent`, outils restreints à `go_to_child`) force la descente jusqu'à une feuille (`_force_descent_to_leaf`, plafonné à 6 pas) plutôt que de laisser la finalisation accepter une catégorie comme réponse ; en tout dernier recours, une descente déterministe non-LLM (`_first_leaf_from`, premier enfant à chaque niveau) fournit une cible de repli qui ne peut jamais échouer. Un filet de sécurité final rejette toute sortie de finalisation qui atterrirait quand même sur un code non terminal. Le `try/except` englobe désormais toute la boucle (exploration, retry, descente forcée, finalisation) et non plus la seule finalisation, pour que tout mode d'échec (ex. `openai.APITimeoutError` sur l'endpoint LLM partagé) se dégrade de la même façon vers `_fallback_output`, en le signalant explicitement en erreur (`level="ERROR"`) sur le span Langfuse courant plutôt que de laisser l'exception avalée passer pour un succès normal.
- `NavigatorAgenticClassifier` — classifieur concret : instructions demandant au *Navigator* de descendre jusqu'à un code terminal (`is_final = 1`) en justifiant chaque choix, en démarrant systématiquement par `get_current_children()`.
- `AgenticRAGClassifier` (`agentic_rag.py`) — approche hybride : récupération du code le plus proche par similarité d'embedding (`Graph.get_closest_codes`, recherche vectorielle Neo4j filtrée sur les codes finaux), utilisé comme point de départ (*warm start*) pour le *Navigator* plutôt que la racine. L'agent vérifie ce point de départ avec les outils du *Navigator* (informations du noeud, enfants, frères, parent) et navigue pour le corriger si besoin, avant de rendre un `MatchVerificationInput`. Branché dans la CLI via `--agentic-rag`.
- `SummaryAgenticClassifier` (`summary_classifier.py`) — hérite directement de `BaseAgent` (pas de `BaseClassifier`) : le modèle reçoit d'emblée, dans son prompt système, un résumé texte de la nomenclature (par défaut la hiérarchie complète, code + nom seulement, généré hors ligne par `src/neo4j_graph/build_nace_summary.py` depuis Neo4j, cf. §2.3) et les outils *stateless* de `Graph` (`get_code_information(code)`, `get_children(code)`, etc., par opposition aux outils du *Navigator* qui opèrent sur une position courante). Contrairement aux deux classifieurs précédents, il n'y a ici qu'un seul `Runner.run` libre (`tool_choice` non forcé) : le modèle décide lui-même quels outils appeler, avec quel code, et quand conclure — choix de conception assumé, sans garde-fou Python empêchant la remontée d'un code non terminal (les instructions demandent seulement au modèle de ne conclure que sur `is_final = 1`). Branché dans la CLI via `--summary`.
- `SupervisedClassifier` (`supervised_classifier.py`) — **pas un agent LLM** : appelle le modèle supervisé de production via l'API déployée `codif-ape-API` (authentification HTTP Basic, `CODIF_APE_API_USERNAME` / `CODIF_APE_API_PASSWORD` / `CODIF_APE_API_URL`), plutôt que chargé en local via MLflow, pour éviter d'ajouter torch/transformers/torchfasttext aux dépendances de ce dépôt. L'expose avec le même contrat de sortie (`MatchVerificationInput`) que les deux classifieurs agentiques, pour servir de référence dans la comparaison chiffrée (cf. cadrage §3.3-B, note de conception). Branché dans la CLI via `--supervised`.

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

La CLI (`src/utils/parser.py`) expose quatre méthodes de classification, avec vérification optionnelle :

```bash
uv run -m src.main --navigator "Boulangerie artisanale avec vente directe"
uv run -m src.main --agentic-rag "Boulangerie" --verify
uv run -m src.main --summary "Boulangerie"
uv run -m src.main --supervised "Boulangerie"
uv run -m src.main --navigator --batch-file requetes.txt --experiment-name mon-experience
```

- `--navigator QUERY` — classification agentique par navigation hiérarchique (*Navigator*) ;
- `--agentic-rag QUERY` — classification par recherche vectorielle comme point de départ (*warm start*) du *Navigator* (cf. §3.4) ;
- `--summary QUERY` — classification à partir du résumé de la nomenclature donné d'emblée au modèle, qui choisit librement quels outils/codes interroger (*`SummaryAgenticClassifier`*, cf. §3.4) ;
- `--supervised QUERY` — classification par le modèle supervisé de production via MLflow (cf. §3.4) ;
- `--verify` — chaîne la prédiction dans le *MatchVerifier* pour double vérification (cf. §3.5) ;
- `--batch-file FILE` — traite un fichier de requêtes (une par ligne) avec la méthode choisie ;
- `--experiment-name` — nom d'expérience propagé au traçage Langfuse.

## 5. Module d'évaluation (`src/evaluation/`)

Socle du chantier prioritaire du mois (cf. cadrage §3.1–3.2), en trois briques :

- **`metrics.py`** — métriques pures Python (sans dépendance, testées unitairement dans `tests/`) : normalisation des codes (`"10.71C"` ≡ `"1071C"`), exactitude à la feuille, exactitude par niveau hiérarchique (préfixes : 2 = division, 3 = groupe, 4 = classe pour la NAF), taux d'échec (prédictions n'ayant pas atteint de code final, comptées comme erreurs), taux de faible confiance (`low_confidence_rate`, distinct du taux d'échec : un code peut être renvoyé avec une confiance nulle par le repli de finalisation `_fallback_output`). `accuracy_at_depth`/`evaluate` acceptent un paramètre `weights` optionnel pour une lecture pondérée en plus de la lecture non pondérée historique (jamais en remplacement, cf. ci-dessous).
- **`build_eval_set.py`** — construction du jeu d'évaluation stratifié : lecture du parquet labellisé (local ou S3/Datalab), stratification par préfixe de code (division par défaut), tirage plafonné par strate et reproductible (seed) — les strates plus petites que le plafond sont conservées en entier. Le sur-échantillonnage des strates rares casse la fréquence réelle des codes ; deux colonnes sont donc ajoutées au jeu produit : `eval_stratum` (clé de strate, réutilisée par le bootstrap stratifié) et `ipw_weight` (poids de repondération, population de la strate / lignes tirées) qui permet de reconstruire une exactitude représentative du trafic réel via `evaluate(..., weights=...)`.
- **`bootstrap.py`** — intervalle de confiance bootstrap (`bootstrap_ci`) pour une métrique, par rééchantillonnage en grappes **à l'intérieur de chaque strate** (jamais entre strates, pour respecter le plan d'échantillonnage de `build_eval_set.py`).
- **`compare.py`** — comparaison statistique appariée de deux campagnes exécutées sur le même jeu d'évaluation (mêmes lignes, même vérité terrain) : bootstrap apparié en grappes sur la différence d'exactitude, et test de McNemar en complément — répond au chantier « rigueur statistique » de la note de conception (cadrage §3.3-B) et évite l'erreur classique de comparer deux IC indépendants sur des données appariées.
- **`run_eval.py`** — harnais de campagne : exécute une méthode (`navigator`, `agentic-rag`, `summary` ou `supervised`) sur le jeu d'évaluation, écrit les prédictions détaillées (parquet) et le rapport de métriques (JSON), avec exactitude pondérée automatique si `ipw_weight` est présent dans le jeu d'évaluation. **Depuis le 15/07**, l'appel au classifieur pour chaque libellé est entouré d'un `try/except` : tous les classifieurs n'ont pas le garde-fou Python de `BaseClassifier` (§3.4) — `SummaryAgenticClassifier` notamment est un unique `Runner.run` libre, sans filet de sécurité équivalent — donc une exception non gérée (ex. `openai.APITimeoutError`) peut encore remonter jusqu'ici ; elle est désormais journalisée et transformée en prédiction d'échec (`code=""`, confiance 0.0), au lieu de faire échouer toute la campagne sur un seul libellé. Nécessite Neo4j et l'API LLM à l'exécution.

```bash
uv run -m src.evaluation.build_eval_set --input <parquet S3/local> --output data/eval/eval_set.parquet
uv run -m src.evaluation.run_eval --eval-set data/eval/eval_set.parquet --method navigator --bootstrap 1000
uv run -m src.evaluation.compare --a data/eval/results/predictions_navigator.parquet --b data/eval/results/predictions_agentic-rag.parquet
```

Le jeu d'évaluation est désormais construit (`data/eval/eval_set.parquet`, 5 181 lignes, stratifié par code complet — `apet2025`, ~10 exemples/code) ; `run_eval.py` propose quatre méthodes : `navigator`, `agentic-rag`, `summary`, `supervised`. **[à compléter]** : le jeu d'évaluation versionné a été construit avant l'ajout d'`ipw_weight`/`eval_stratum` — à reconstruire depuis la source (`df_test`) pour bénéficier de poids non triviaux (voir `stratified_sample`) et d'un bootstrap qui n'ait pas à se rabattre sur une strate unique.

### 5.1 Diagnostic de l'espace d'embedding (`evaluate_embeddings.py`)

Script autonome à la racine du dépôt (volontairement hors `src/evaluation/`, cf. cadrage §3.3-B), qui évalue la qualité de la recherche par similarité (notices NAF2025 ↔ libellés) utilisée comme *warm start* par l'Agentic RAG (`Graph.get_closest_codes`, §3.4), indépendamment de la navigation LLM qui la suit :

- **Quantitatif** : k-NN cosinus entre l'embedding d'un libellé (préfixé `"query : "`, comme au moment de l'inférence dans `graph.py`) et les embeddings des notices NAF2025 (codes terminaux uniquement) — accuracy@1, recall@5 et exactitude hiérarchique (section/division/groupe/classe/code) contre la vérité terrain (`apet2025`), sur `data/eval/eval_set_sample15.parquet` / `eval_set_sample30.parquet` (échantillon utilisé pour la projection 2D) et sur `data/eval/eval_set.parquet` (échantillon complet, 5 181 libellés, seule source des métriques chiffrées).
- **Visuel** : projection 2D (UMAP/PaCMAP/t-SNE/PCA) des notices et des libellés, arêtes k-NN correctes (vert) / incorrectes (bleu) et vérité terrain, une figure Plotly comparative par modèle écrite dans `data/eval/embedding_diagnostics/<modèle>_comparison.html`.
- **Comparaison multi-modèles** : éditer la liste `CANDIDATE_MODELS` en tête de script pour comparer plusieurs modèles d'embedding déployés derrière `URL_EMBEDDING_API`.
- **Page Quarto** ([`presentation/embeddings.qmd`](../presentation/embeddings.qmd)) : réutilise directement les fonctions du script (pas de logique dupliquée) pour publier le diagnostic sur le site, avec en plus la cohésion/confusion inter-groupes, l'écart requête/passage, et une expérience comparant le texte encodé actuel (`NAME + Implementation_rule + Includes + IncludesAlso`) à une variante ajoutant `Excludes`.

```bash
uv run python evaluate_embeddings.py
```

**Résultat (15/07, échantillon complet de 5 181 libellés)** avec le modèle configuré (`qwen3-embedding-8b`) : accuracy@1 = 48,1 %, recall@5 = 77,5 %, avec une exactitude hiérarchique qui croît proprement du code exact (48,1 %) jusqu'à la section (77,7 %) — signe que les erreurs de top-1 restent en général dans la bonne zone de la nomenclature plutôt que de partir dans une branche totalement différente. Ajouter `Excludes` au texte encodé (expérience de `presentation/embeddings.qmd`, hors production) améliore encore ces chiffres (accuracy@1 = 50,7 %, recall@5 = 80,4 %). Ce résultat remplace le 0 % rapporté par un diagnostic exploratoire antérieur au commit du script (cf. l'historique de `docs/cadrage_2026-07.md` §2.2) : l'espace d'embedding est donc exploitable comme *warm start*, avec une marge confortable en recall@5 mais un top-1 encore insuffisant pour se passer de la vérification/correction du *Navigator* qui suit (§3.4) — voir §8.

## 6. Configuration (variables d'environnement)

| Variable | Usage |
|---|---|
| `NEO4J_URL`, `NEO4J_USERNAME`, `NEO4J_PWD` | Connexion à la base de graphe |
| `OPENAI_BASE_URL`, `OPENAI_API_KEY` | Client LLM compatible OpenAI (génération) |
| `GENERATION_MODEL` | Modèle utilisé par les agents (`BaseAgent`) |
| `MAX_TURNS` | Nombre maximal de tours d'agent (boucle outil → réponse) |
| `EMBEDDING_MODEL`, `URL_EMBEDDING_API`, `MAX_TOKENS` | Modèle et service d'embedding utilisés lors de la construction du graphe |
| `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`, `AWS_ENDPOINT_URL` | Accès S3 (Datalab/Onyxia) pour les données sources et notices |
| `MLFLOW_TRACKING_URI`, `MLFLOW_MODEL_URI` | Chargement du modèle supervisé de production par `SupervisedClassifier` |
| `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_BASE_URL` | Traçage applicatif Langfuse (voir ci-dessous) |

Le traçage applicatif (sessions, coûts, latence, arbre d'appels des agents) est assuré par **Langfuse** (`get_client`, `propagate_attributes`, `@observe` dans `src/main.py`).

### 6.1 Stockage S3 des sorties

`src/utils/storage.py` centralise la construction du client S3 (`get_file_system`, ex-dupliquée dans `notice_manager.py`, `build_eval_set.py` et `convert_to_parquet.py`) et expose `open_path`/`makedirs`/`path_exists`/`remove`, des équivalents de `open`/`os.makedirs`/`os.path.exists`/`os.remove` qui basculent automatiquement sur `s3fs` dès que le chemin commence par `s3://` (sinon, comportement disque local inchangé). `run_eval.py`, `verify_train_labels.py`, `apps/human_review_app.py`, `build_nace_summary.py` et `evaluate_embeddings.py` s'appuient dessus : passer un chemin `s3://projet-ape/graal/data/...` à `--output-dir`/`--output`/`--input` (ou à `OUTPUT_DIR` pour `evaluate_embeddings.py`) écrit/lit directement sur le datalake, sans changer les valeurs par défaut (toujours locales sous `data/`).

Les campagnes d'évaluation, revues humaines et diagnostics d'embeddings déjà produits localement (`data/eval/results*`, `train_verification*`, `embedding_diagnostics/`, `human_review/`, `data/nace_summary.txt`) ont été synchronisés une première fois sous `s3://projet-ape/graal/data/` (même arborescence que `data/`) — ces répertoires restent gitignorés (sorties reproductibles, pas du code) et vivent désormais sur S3 plutôt que seulement sur le poste de développement.

Le checkpoint de prédictions de `run_eval.py` (flush après chaque label, pensé pour limiter la perte en cas de crash) perd sa garantie de durabilité si `--output-dir` est un chemin S3 : `s3fs` bufferise l'écriture et ne pousse l'objet qu'à la fermeture du fichier.

### 6.2 Traçage Langfuse

Audit du 8/07 : le traçage fonctionnait pour `classify_agentic_rag`, `classify_supervised` et `process_batch_file` (`@observe` actif, appels LLM individuels journalisés via `langfuse.openai.AsyncOpenAI` dans `base_agent.py`), mais restait incomplet sur plusieurs points. **Corrigé le 9/07** :

- **`classify_navigator` est maintenant tracé** (`@observe` ré-activé dans `src/main.py`) alors que c'est le chemin agentique principal (cf. cadrage §2.1) — chaque appel LLM était déjà journalisé individuellement mais sans trace/span parent qui les relie en un arbre de raisonnement cohérent.
- **Les échecs de finalisation de `_run_navigator_loop` remontent maintenant comme des erreurs dans Langfuse** : en plus du repli `_fallback_output` (`base_classifier.py`), le span courant est explicitement marqué `level="ERROR"` (`get_client().update_current_span`) avant de retomber sur la dernière position connue, au lieu d'apparaître comme un succès normal (confiance 0.0).
- **`--experiment-name` est maintenant réellement propagé au traçage Langfuse** : la valeur est attachée à la trace courante (nom, tag et métadonnée via `get_client().update_current_trace`) dans les quatre points d'entrée (`classify_navigator`, `classify_agentic_rag`, `classify_supervised`, `process_batch_file`), donc filtrable/groupable par expérience dans l'UI Langfuse.

Reste à faire :

- Pas de `langfuse.flush()`/`shutdown()` explicite avant la sortie du script CLI (repose sur le hook `atexit` du SDK, suffisant en usage normal mais fragile en cas d'arrêt forcé d'un job batch).

**Piège fréquent sur `MLFLOW_MODEL_URI`** : ce n'est pas le lien de la page MLflow ouverte dans le navigateur, mais une URI au schéma `models:` — ex. `models:/FastText-pytorch/9` (pas `https://.../#/models/FastText-pytorch/versions/9`). Et `MLFLOW_TRACKING_URI` doit pointer vers le serveur MLflow où ce modèle est **effectivement enregistré** (le plus souvent l'instance MLflow partagée du projet, ex. `projet-ape-mlflow.user.lab.sspcloud.fr`) — pas nécessairement l'instance MLflow personnelle par défaut sur le Datalab, qui n'a pas accès au registre d'un autre projet. `SupervisedClassifier` lève une erreur explicite si `MLFLOW_MODEL_URI` est un lien `http(s)://` plutôt qu'une URI `models:`.

Un test de connectivité par service externe (Neo4j, LLM de génération, embedding, S3, Langfuse, MLflow) est disponible dans `tests/test_connections.py` — chaque test se saute automatiquement si les variables requises sont absentes, pour rester vert en CI sans secrets Datalab tout en détectant un endpoint/identifiant mal configuré quand ils sont présents.

## 7. Comment étendre GRAAL à une nouvelle nomenclature **[à compléter]**

Cette section documentera, une fois formalisée, le mode opératoire complet pour instancier GRAAL sur une nouvelle nomenclature : format attendu du fichier de notices, exécution du pipeline de construction du graphe, adaptation minimale des prompts si nécessaire.

## 8. Limites connues et dette technique

Recensées ici pour mémoire (suivi détaillé dans le document de cadrage) :

- Les composants branchés le 6/07 (classifieur *Agentic RAG* dans la CLI, chaînage `--verify`, harnais `run_eval`) n'ont pas encore été évaluées.
- **L'espace d'embedding utilisé par l'Agentic RAG comme *warm start*** (`EMBEDDING_MODEL=qwen3-embedding-8b`) a un top-1 correct un peu moins d'une fois sur deux (accuracy@1 = 48,1 % sur l'échantillon complet de 5 181 libellés, cf. §5.1) : suffisant comme point de départ à corriger par le *Navigator*, pas pour lui faire confiance seul. Un diagnostic exploratoire antérieur au commit du script (`evaluate_embeddings.py`) avait rapporté accuracy@1 = recall@5 = 0 % sur un échantillon de 30 libellés (ex. « FOOTBALL FEMININ » classé 373ᵉ/747) ; ce chiffre n'a pas été reproduit une fois le script committé et étendu (§5.1) — le même échantillon de 30 libellés donne désormais 46,7 %/66,7 %, en ligne avec le résultat sur l'échantillon complet. Cause de cet écart non tranchée (script antérieur non versionné, donc non diffable), mais le diagnostic actuel, committé et reproductible, ne corrobore pas le 0 % initial.
- `SupervisedClassifier` (modèle de production via MLflow) est également non validé fonctionnellement : le parsing de la sortie `.predict()` est écrit pour plusieurs formats plausibles mais n'a pas pu être testé contre le modèle réel dans cet environnement (pas d'accès au tracking MLflow).
- La CI couvre lint, syntaxe et tests unitaires purs, mais **pas de tests d'intégration** (agents + graphe + MLflow) : ils nécessiteraient les services correspondants dans le workflow. `tests/test_connections.py` comble partiellement ce manque en local/Datalab (smoke tests skippés si les identifiants sont absents).
---
