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
- l'exécution (`__call__`) via `Runner.run` du SDK `agents`, avec un nombre maximal de tours donné par `get_max_turns` (`MAX_TURNS` par défaut). Cette description vaut pour `CodeChooser`/`MatchVerifier`/`SupervisedClassifier` ; `BaseClassifier` (§3.4) remplace cette boucle par sa propre boucle pas-à-pas ;
- trois points de personnalisation par agent, plutôt que trois réglages globaux : `get_tools` (tous les outils du graphe par défaut, aucun pour les fermeurs — cf. §3.5), `get_max_turns` (un seul tour pour un agent sans outil : il n'a rien à faire d'un second, et un plafond global à 15 ne fait que multiplier par quinze le temps qu'une défaillance met à se voir) et `wrap_output`, qui construit l'objet rendu à l'appelant à partir de ce que le modèle a généré. Ce dernier permet de garder `get_output_type` **exactement** égal au schéma montré au modèle : un champ présent dans ce schéma est un champ que le modèle remplira, y compris quand sa description lui demande de ne pas le faire (`tool_call_count`, `attempt_count`, observés remplis dans les traces). Ce qui se calcule en Python après l'appel est donc ajouté par `wrap_output` dans un type distinct (`MatchVerification` → `MatchVerificationResult`, `CodeSelection` → `CodeChoice`).

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
- **`MatchVerifier`** — audite une correspondance libellé ↔ code : il note le code en place (`match_score`, en %), nomme **à chaque appel** le meilleur code concurrent qu'il trouve (`alternative_code`) et le note aussi (`alternative_score`), écrit une explication globale, puis tranche (`is_match`, plus `is_match_score` : la netteté du verdict) — auxquels s'ajoute `p_match` (voir plus bas). C'est cet agent qui porte le cas d'usage « monitoring du modèle en production » (cf. cadrage §1.3).

Ces deux agents **ferment** une chaîne : ils rendent un jugement, là où les classifieurs explorent. D'où des traits communs, qui les distinguent du reste des agents :

- **presque aucun outil**. Le `CodeChooser` n'en a aucun (`get_tools` → `[]`, `get_max_turns` → 1) : arbitrer entre des codes déjà retenus est un jugement en un coup, et les schémas des cinq outils du graphe lui coûteraient ~500 tokens de prompt par appel pour un détour multi-tours dont il n'a pas l'usage. Le *MatchVerifier*, lui, en a **un seul** (`Graph.get_notice_tools` → `get_notices`, `get_max_turns` → 3) depuis qu'on lui demande de proposer une alternative : il ne peut plus se contenter du code qu'on lui tend. Un seul outil, et en *batch* (une liste de codes, pas un code), parce qu'un modèle qui interroge ses candidats un par un paie un aller-retour LLM complet par candidat — et avec le raisonnement activé, un tour de plus double le temps de la ligne. **Réserve** : sur 34 lignes de mise au point, le modèle n'a appelé cet outil aucune fois, même sur les cas serrés où les instructions le lui demandent ; il se juge assez informé par le résumé et par la notice du code jugé. L'outil reste en place (il coûte un schéma, pas un appel), mais tant que `tool_call_count` vaut zéro dans les runs, c'est le prompt qui fait tout le travail. Dans les deux cas la notice des codes déjà connus reste **dans le prompt** (`Graph.get_notice`), garantie présente, là où un appel d'outil dépend du bon vouloir du modèle ; l'outil ne sert qu'aux candidats que le modèle vient d'imaginer.
- **la nomenclature entière dans les instructions**, pour le *MatchVerifier* (`build_nace_summary.build_summary_text`, niveaux 1 à 5 : 1 059 positions, code et nom, ~21 000 tokens). C'est ce qui rend l'alternative possible — on ne peut pas nommer le meilleur concurrent d'un code sans savoir quels codes existent. Le coût en *prefill* est réel mais amorti : les instructions sont identiques d'un appel à l'autre, donc vLLM les sert depuis son cache de préfixe (mesuré : 1,2 s au premier appel, 0,75 s ensuite pour 20 700 tokens de prompt).
- **une règle de décision explicite**, pour le *MatchVerifier*. « Est-ce que ce code correspond ? » n'en est pas une : les libellés SIRENE sont tronqués, abrégés, souvent plus vagues que la nomenclature, et sans critère le modèle s'en invente un — strict, qui rejetait l'imprécision autant que l'erreur (`'avec chauffeur' -> 4933H` rejeté comme « trop vague »). Le prompt tranche, et la règle est maintenant chiffrée : le code en place garde le bénéfice du doute, il n'est rejeté que si sa notice le rend **incompatible** (`match_score` < 40) ou si l'alternative le dépasse d'au moins 20 points. À scores voisins, `is_match` reste vrai. S'y ajoutent deux règles de lecture d'un libellé, qui règlent le cas fréquent du libellé à plusieurs activités : si elles sont **conflictuelles** (elles relèvent de codes qui s'excluent), c'est **la première citée** qui décide ; si elles ne se contredisent pas — une énumération, un métier décrit par ses tâches — on code l'**impression d'ensemble** et non un élément de la liste.
- **rien sur la provenance du code**. Le même prompt juge un label de référence et la proposition d'un classifieur, sur la paire seule : annoncer une vérité terrain invite à la valider, annoncer une proposition de modèle invite à la corriger. Seule l'explication qui accompagne éventuellement le code est montrée (un raisonnement s'évalue) ; sa confiance, elle, ne l'est pas — c'est une ancre, pas un argument.

Deux mesures d'incertitude, à ne pas confondre, accompagnent le verdict binaire du *MatchVerifier* :

- `is_match_score` (ex-`confidence`), auto-déclarée par le modèle et désormais exprimée en pourcentage entier plutôt qu'en flottant sur [0, 1] — le modèle est nettement moins grégaire sur cette échelle-là (la moitié des flottants auto-déclarés valaient 0,9). Sa description l'ancre maintenant sur l'écart entre `match_score` et `alternative_score` plutôt que sur une assurance générale, mais elle reste mal calibrée par construction : on observait 0,90 en moyenne sur `is_match=true` et 0,50 sur `false`, c'est-à-dire un modèle qui reportait P(le code est bon) plutôt que sa confiance dans son propre verdict. Un seuil sur cette colonne reste peu interprétable.
- `p_match`, calculée en Python : P(`is_match` = true) lue dans les **logprobs** du token `true`/`false` effectivement généré, renormalisée sur ces deux candidats (`p_match_from_logprobs`). Gratuite — les logprobs portent sur des tokens de toute façon décodés, aucun token supplémentaire — et c'est la probabilité du modèle, pas son opinion sur sa probabilité. C'est elle qui permet de choisir *a posteriori* un point de fonctionnement (seuil) au lieu de subir celui que le prompt impose, et donc de tracer une courbe ROC une fois les annotations humaines disponibles plutôt que de mesurer un point unique.

Le raisonnement du modèle (bloc `<think>`) est **activé** pour le *MatchVerifier* (`enable_thinking=True`, paramétrable au constructeur). C'est l'inverse du réglage précédent, et ce qui l'a renversé est le changement de tâche : tant que l'agent ne faisait que valider une paire, le bloc `<think>` refaisait à perte ce que le champ `explanation`, généré avant le verdict, faisait déjà. Chercher le meilleur concurrent parmi un millier de positions, décider s'il faut en lire les notices, puis comparer deux codes chiffres à l'appui, ne tient pas dans une explication de 50 mots destinée à un humain. S'y ajoute une raison moins agréable : **l'endpoint ne respecte pas l'ordre des propriétés du schéma** — sur un même prompt, `explanation` sort tantôt en tête, tantôt en dernier —, donc l'argument « l'explication avant le verdict » ne tient plus ; le bloc `<think>`, lui, précède toujours le JSON.

> **Le prix, mesuré, est lourd** : sur le même échantillon et le même prompt, **13 à 15 s** par ligne avec raisonnement contre **1,5 s** sans (≈2 400 tokens décodés en médiane contre 150, à ~185 tokens/s). Et il n'y a **pas de réglage intermédiaire** — ce qui a été vérifié une intervention après l'autre, toutes sur le même échantillon de 7 lignes :
>
> | Intervention | Résultat |
> |---|---|
> | Consigne de brièveté (« 120 mots », « réflexion courte », « tu es pressé ») | 13-15 s : aucun effet |
> | Consigne dure (« 3 phrases maximum », interdits explicites) | **pire** (16 s) : le modèle se met à compter ses phrases |
> | Recette de raisonnement en 3 temps imposée | pire (16,2 s) |
> | Résumé réduit aux niveaux 1-3 (5 300 tokens de prompt au lieu de 21 800) | 13,7 s : le volume de contexte n'est pas le moteur |
> | Short-list de candidats par similarité d'embedding, pour supprimer la recherche | pire (16,0 s), et 13,6 s même en retirant le résumé |
> | `chat_template_kwargs.thinking_budget`, `reasoning_effort` | ignorés par l'endpoint : réponses identiques au token près |
> | Autre modèle servi (`qwen3-8-27b`, `gemma4-26b-moe`) | 15,8 s ; ou 1,5 s mais sans raisonner du tout |
>
> Autrement dit, la longueur du raisonnement est une propriété du couple (modèle, tâche), pas du prompt. Le plafond `max_tokens` ne sert donc qu'à borner la queue de distribution, pas à raccourcir le cas moyen — et il ne peut pas être serré : le bloc `<think>` précédant le JSON, une troncature en plein raisonnement ne rend **aucun** JSON, donc une ligne rejouée puis abandonnée, et le plafond payé deux fois pour rien. `enable_thinking` est en pratique le seul levier, et il est tout ou rien. C'est aussi pour cela que le budget de tours est à **2** et que `get_notices` prend une liste : un tour de dialogue de plus, c'est un bloc `<think>` de plus, donc le temps de la ligne doublé.

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

### 5.1 Évaluation du *MatchVerifier* (`match_verifier_eval.py` + `apps/match_verifier_eval_app.py`)

Le *MatchVerifier* (§3.5) porte le cas d'usage « monitoring » : il faut donc savoir ce que vaut son verdict, et pas seulement s'en servir. La chaîne tient en deux scripts.

- **`match_verifier_eval.py`** — tire un échantillon du jeu d'entraînement et demande au *MatchVerifier*, sans lui fournir d'explication proposée (il juge donc le label comme une vérité terrain brute, pas comme la proposition d'un modèle), si le code attaché à chaque libellé est correct. Les lignes tournent concurremment (`--concurrency` borne les appels LLM en vol), chaque appel est retenté une fois (`call_with_retries`), chaque ligne terminée est journalisée dans un checkpoint JSONL. Le parquet produit tient en dix colonnes — `libelle`, `current_code`, `match_verifier_verdict`, `match_verifier_match_score`, `match_verifier_alternative_code`, `match_verifier_alternative_score`, `match_verifier_explanation`, `match_verifier_is_match_score`, `match_verifier_p_match` et `match_verifier_duration_s` — et rien d'autre : c'est l'entrée de l'app de revue — le verdict, mais aussi ce contre quoi il a été rendu, puisque le vérificateur note le code en place, nomme à chaque ligne le meilleur code concurrent qu'il ait trouvé et note celui-là aussi (§3.5) —, plus le temps d'inférence de chaque verdict (mesuré autour de l'appel au vérificateur, hors attente derrière `--concurrency` : lookup Neo4j de la notice compris, et retry compris le cas échéant), pour chiffrer le coût en latence du monitoring. Sortie par défaut : `s3://projet-ape/graal/data/eval/match_verifier_eval/<commit>/<modèle>/` (cf. §5.1.1). Comme le checkpoint perd sa durabilité sur S3 (cf. §6.1), `--checkpoint-dir` permet de le garder en local pendant que le résultat part sur le datalake — lui aussi suffixé par `/<commit>/<modèle>`.
- **`apps/match_verifier_eval_app.py`** — app Flask de revue humaine de ce parquet. Pour chaque ligne : le libellé, le code avec sa notice officielle (Neo4j), et le verdict du *MatchVerifier* — accompagné de sa note de correspondance, du meilleur code concurrent trouvé et de la note de celui-ci, pour que l'annotateur voie contre quoi le verdict a été rendu. Les runs antérieurs au passage aux notes en pourcentage restent lisibles : leur `match_verifier_confidence` est relue en repli et les colonnes de score restent vides. L'annotateur répond à une question — le verdict est-il correct ? — plus un champ libre pour le code qu'il aurait mis. Une version antérieure posait en plus, séparément, « le code est-il correct ? », ce qui permettait de recalculer exactitude/précision/rappel de `is_match` sans faire confiance ni au label d'entraînement ni au verdict ; la question a été retirée du formulaire (coût d'annotation jugé supérieur à son apport), mais la colonne `human_code_correct` reste au log et au schéma, et `/metrics` continue d'afficher ces métriques sur les jugements déjà collectés — gelées, avec leur propre dénominateur. Même découpage par annotateur que `multi_method_review_app.py` (pool partagé pour l'accord inter-annotateurs + tranche propre à chacun, déterministe à partir de `--input`/`--reviewers`/`--shared-n`/`--unique-n`/`--seed`, donc revue asynchrone). Un annotateur qui n'était pas prévu au lancement peut s'ajouter lui-même depuis la page d'accueil : il donne son nom, le nombre de libellés qu'il veut relire et la part de sa tranche qui doit recouper celle des autres (défaut `--overlap-pct`, 30 %) — ce recouvrement est tiré en priorité parmi les lignes **déjà jugées** (recouper une ligne que personne n'a encore jugée ne donne de l'accord inter-annotateurs que si son propriétaire y vient), le reste parmi les lignes du parquet que personne n'a. Cette tranche-là dépend de l'état du moment — qui avait déjà quoi — et ne se recalculerait pas à l'identique : elle est écrite dans `<output-dir>/<commit>/<modèle>/match_verifier_assignments.json` puis relue au démarrage, et ses recouvrements rejoignent le pool partagé au calcul de l'accord. `--reviewers` peut donc être laissé vide. Les jugements sont journalisés en JSONL append-only (idempotent par `(reviewer, row_id)`) et rematérialisés après chaque validation en parquet à côté, avec le nom de l'annotateur en colonne `reviewer`. L'app lit `<input>/<commit>/<modèle>/match_verifier_eval.parquet` et écrit dans `<output-dir>/<commit>/<modèle>/` : `--commit` et `--model` valent par défaut le tag de HEAD et `GENERATION_MODEL`, et si aucun run n'existe pour ce couple l'app refuse de démarrer en listant les runs disponibles (cf. §5.1.1). Le run en cours de revue (`<commit>/<modèle>`) est affiché sur chaque page.

```bash
uv run -m src.evaluation.match_verifier_eval --n-samples 500 --concurrency 10
uv run -m src.evaluation.apps.match_verifier_eval_app --reviewers meilame,theo,nathan --port 5052
# comparer deux modèles sur le même prompt :
GENERATION_MODEL=autre-modele uv run -m src.evaluation.match_verifier_eval --n-samples 500
```

`apps/multi_method_review_app.py` reste l'app de revue du jeu d'évaluation multi-méthode (5 candidats, arbitrage `CodeChooser`, un verdict *MatchVerifier* par source) ; celle-ci est sa version resserrée sur le seul vérificateur face aux labels d'entraînement.

Pour une campagne à plusieurs annotateurs, l'app est déployée plutôt que lancée en local : le `Dockerfile` à la racine construit l'image (dépendances installées depuis `uv.lock`) et `.github/workflows/docker.yml` la pousse sur Docker Hub — `meilametayebjee/graal-match-verifier-review`, tags `latest`, `sha-<court>` et semver — à chaque push sur `main`. Les manifests Kubernetes, eux, ne vivent **pas** ici : deployment, service et ingress sont dans le dépôt GitOps [`codif-ape-cd`](https://github.com/InseeFrLab/codif-ape-cd), sous `graal-match-verifier-review/` (avec un README listant les secrets à créer), parce qu'un manifest dupliqué dans deux dépôts est un manifest qui diverge — c'est celui que lit Argo CD qui fait foi. Seule exception, `deploy/argocd/argocd-templates/graal-match-verifier-review.yaml` : l'`Application` Argo CD est le seul manifeste qu'on applique à la main (`kubectl apply -n argocd -f ...`) plutôt qu'Argo CD ne le lise depuis un dépôt, et c'est elle qui lui désigne le dossier à synchroniser. Le run revu y est passé explicitement en `--commit`/`--model` : l'image n'embarque pas `.git`, donc la valeur par défaut — le tag de HEAD — n'a pas de sens en conteneur. Le déploiement tient à un seul réplica (`strategy: Recreate`) parce que l'ajout d'une ligne au JSONL est, sur S3, une réécriture complète de l'objet : deux processus concurrents perdraient des jugements. À l'intérieur d'un processus, Flask servant les requêtes sur des threads, l'app sérialise elle-même l'append et la réécriture du parquet derrière un verrou.

#### 5.1.1 Résultats épinglés au commit et au modèle (`src/utils/run_provenance.py`)

Cette évaluation mesure un prompt **passé dans un modèle**, et les deux bougent sous un script figé : à plat, deux runs s'écrasent l'un l'autre sans que rien dans le résultat ne dise lequel a produit quel chiffre. `match_verifier_eval.py` écrit donc sous `<output>/<commit>/<modèle>/` :

- `<commit>` = `git describe --tags --always HEAD` (le tag s'il y en a un, le SHA court sinon). Le script refuse de démarrer — avant le moindre appel LLM — si l'un des fichiers qui définissent le prompt (`PROMPT_FILES`, aujourd'hui `src/agents/closers/match_verifier.py`, `src/agents/base_agent.py`, `src/neo4j_graph/graph.py` et `src/neo4j_graph/build_nace_summary.py` — le résumé de la nomenclature fait désormais partie du prompt) a des modifications non commitées, y compris indexées ou non suivies : le nom du dossier désignerait alors un commit qui ne contient pas le prompt évalué.
- `<modèle>` = `GENERATION_MODEL` assaini pour un nom de dossier (`Qwen/Qwen3-32B` → `Qwen-Qwen3-32B`). Il n'y a **volontairement pas** de flag `--model` sur le script d'éval : la valeur est lue dans la variable d'environnement que lisent les agents eux-mêmes (`BaseAgent`), donc le dossier nomme le modèle qui a réellement répondu, et non celui qu'un flag aurait revendiqué. Pour comparer deux modèles : `GENERATION_MODEL=... uv run -m ...`. L'app de revue, elle, a bien un `--model`, puisqu'elle ne fait que désigner un run existant.

Le `summary.json` porte en plus `commit`, `commit_sha` et `model`. Le commit vient en premier dans le chemin : une révision de prompt est le changement le plus grossier (elle peut invalider d'un coup les chiffres de tous les modèles), donc grouper par commit garde côte à côte les modèles d'une même révision, la comparaison qu'on veut le plus souvent.

`--allow-dirty` lève le refus, au prix d'un dossier `<commit>-dirty/<modèle>/` : de quoi itérer sur un prompt avant de le commiter sans polluer les résultats de référence, les runs *dirty* successifs s'écrasant délibérément entre eux (ce sont des brouillons, pas des résultats).

Le découpage vaut aussi pour la revue humaine : des jugements portant sur les verdicts de deux prompts — ou de deux modèles — poolés dans un même JSONL fausseraient silencieusement `/metrics`. Le run `ec8bf27/qwen3-6-35b-moe` (500 lignes, 2026-09-15) a été reclassé a posteriori dans ce format ; le modèle y est celui de `GENERATION_MODEL` aujourd'hui, ce run étant antérieur à l'enregistrement du modèle dans le `summary.json`.

**Run courant à relire** : `v0.0.1-1-g2441f54/qwen3-6-35b-moe` (2 000 lignes, 2026-09-16), le premier du *MatchVerifier* qui note et propose une alternative (§3.5). Soit, en chemin complet, `s3://projet-ape/graal/data/eval/match_verifier_eval/v0.0.1-1-g2441f54/qwen3-6-35b-moe/match_verifier_eval.parquet`. Les verdicts de `ec8bf27` portent sur l'ancien prompt : ils ne se poolent pas avec ceux-ci, c'est précisément ce que le découpage par commit empêche. Le déploiement de l'app de revue doit donc voir son `--commit` passer à `v0.0.1-1-g2441f54` — argument qui vit dans le `Deployment` du dépôt GitOps `codif-ape-cd`, pas ici.

> **Limite** : le tag nomme HEAD, donc l'arbre entier, alors que la garde ne couvre que `PROMPT_FILES`. Un run est reproductible depuis son commit pour le prompt, pas nécessairement pour le reste de l'arbre de travail. Côté modèle, le nom ne capte que l'identifiant servi par l'API : deux runs d'un même `GENERATION_MODEL` derrière un endpoint dont les poids ont changé seraient indistinguables.

### 5.2 Diagnostic de l'espace d'embedding (`evaluate_embeddings.py`)

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

`src/utils/storage.py` centralise la construction du client S3 (`get_file_system`, ex-dupliquée dans `notice_manager.py`, `build_eval_set.py` et `convert_to_parquet.py`) et expose `open_path`/`makedirs`/`path_exists`/`remove`/`list_dir`, des équivalents de `open`/`os.makedirs`/`os.path.exists`/`os.remove`/`os.listdir` qui basculent automatiquement sur `s3fs` dès que le chemin commence par `s3://` (sinon, comportement disque local inchangé). `run_eval.py`, `match_verifier_eval.py`, `apps/multi_method_review_app.py`, `apps/match_verifier_eval_app.py`, `build_nace_summary.py` et `evaluate_embeddings.py` s'appuient dessus : passer un chemin `s3://projet-ape/graal/data/...` à `--output-dir`/`--output`/`--input` (ou à `OUTPUT_DIR` pour `evaluate_embeddings.py`) écrit/lit directement sur le datalake, sans changer les valeurs par défaut (toujours locales sous `data/`).

Les campagnes d'évaluation, revues humaines et diagnostics d'embeddings déjà produits localement (`data/eval/results*`, `train_verification*`, `embedding_diagnostics/`, `human_review/`, `data/nace_summary.txt`) ont été synchronisés une première fois sous `s3://projet-ape/graal/data/` (même arborescence que `data/`) — ces répertoires restent gitignorés (sorties reproductibles, pas du code) et vivent désormais sur S3 plutôt que seulement sur le poste de développement.

Le checkpoint de prédictions de `run_eval.py` et de `match_verifier_eval.py` (flush après chaque ligne, pensé pour limiter la perte en cas de crash) perd sa garantie de durabilité si `--output-dir` est un chemin S3 : `s3fs` bufferise l'écriture et ne pousse l'objet qu'à la fermeture du fichier (d'où `--checkpoint-dir` sur `match_verifier_eval.py`, cf. §5.1).

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
- **L'espace d'embedding utilisé par l'Agentic RAG comme *warm start*** (`EMBEDDING_MODEL=qwen3-embedding-8b`) a un top-1 correct un peu moins d'une fois sur deux (accuracy@1 = 48,1 % sur l'échantillon complet de 5 181 libellés, cf. §5.2) : suffisant comme point de départ à corriger par le *Navigator*, pas pour lui faire confiance seul. Un diagnostic exploratoire antérieur au commit du script (`evaluate_embeddings.py`) avait rapporté accuracy@1 = recall@5 = 0 % sur un échantillon de 30 libellés (ex. « FOOTBALL FEMININ » classé 373ᵉ/747) ; ce chiffre n'a pas été reproduit une fois le script committé et étendu (§5.2) — le même échantillon de 30 libellés donne désormais 46,7 %/66,7 %, en ligne avec le résultat sur l'échantillon complet. Cause de cet écart non tranchée (script antérieur non versionné, donc non diffable), mais le diagnostic actuel, committé et reproductible, ne corrobore pas le 0 % initial.
- `SupervisedClassifier` (modèle de production via MLflow) est également non validé fonctionnellement : le parsing de la sortie `.predict()` est écrit pour plusieurs formats plausibles mais n'a pas pu être testé contre le modèle réel dans cet environnement (pas d'accès au tracking MLflow).
- La CI couvre lint, syntaxe et tests unitaires purs, mais **pas de tests d'intégration** (agents + graphe + MLflow) : ils nécessiteraient les services correspondants dans le workflow. `tests/test_connections.py` comble partiellement ce manque en local/Datalab (smoke tests skippés si les identifiants sont absents).
---
