"""Classifieur donnant à l'agent la structure globale de la nomenclature d'emblée,
plutôt qu'une navigation guidée pas à pas (`BaseClassifier`/`NavigatorAgenticClassifier`)
ou un point de départ par similarité (`AgenticRAGClassifier`).

Contrairement à `BaseClassifier`, qui pilote la boucle exploration/finalisation en
Python parce qu'un seul `Runner.run` ne peut pas fiablement à la fois utiliser les
outils et savoir quand s'arrêter, `SummaryAgenticClassifier` laisse le modèle libre :
un unique `Runner.run` (hérité de `BaseAgent`), `tool_choice` non forcé, où le modèle
décide lui-même s'il appelle un outil, lequel, avec quel code, et quand conclure. C'est
un choix de conception assumé pour ce classifieur (donner le plus de liberté possible
à l'agent) : il n'y a donc pas de garde-fou Python équivalent à celui de
`BaseClassifier` empêchant la remontée d'un code non terminal — les instructions
demandent explicitement au modèle de ne conclure que sur un code avec is_final=1.

Utilise les outils *stateless* de `Graph` (`get_code_information`, `get_children`,
`get_descendants`, `get_siblings`, `get_parent`), qui prennent chacun un `code` en
argument, plutôt que les outils du `Navigator` relatifs à une position courante : le
modèle choisit directement quel code interroger, sans notion de position à faire
évoluer pas à pas.

Le résumé est chargé depuis un fichier texte pré-généré par `build_nace_summary.py`
(pas recalculé à chaque appel).
"""

import logging

from src.agents.base_agent import BaseAgent
from src.agents.closers.match_verifier import MatchVerificationInput
from src.neo4j_graph.graph import Graph

logger = logging.getLogger(__name__)

DEFAULT_SUMMARY_PATH = "data/nace_summary.txt"


class SummaryAgenticClassifier(BaseAgent):
    def __init__(self, graph: Graph, summary_path: str = DEFAULT_SUMMARY_PATH):
        with open(summary_path, "r", encoding="utf-8") as f:
            self.summary = f.read()
        super().__init__(graph)

    def get_agent_name(self) -> str:
        return "Summary Agentic Classifier"

    def get_output_type(self):
        return MatchVerificationInput

    def get_instructions(self) -> str:
        return f"""
        Vous êtes un expert en classification NACE.

        Voici un résumé de la nomenclature pour vous orienter :

        {self.summary}

        Ce résumé ne donne que le code et le nom de chaque position : il ne contient pas
        la notice officielle complète (description, inclusions, exclusions, règle
        d'affectation). À vous de décider, avec les outils à votre disposition
        (get_code_information, get_children, get_descendants, get_siblings, get_parent),
        quels codes approfondir pour confirmer ou affiner la position la plus spécifique
        possible. Vous êtes libre d'explorer comme vous le souhaitez : aucune étape n'est
        imposée, et vous pouvez conclure directement si le résumé suffit déjà à trancher
        avec certitude.

        Ne renvoyez comme réponse finale qu'un code terminal (is_final = 1). Justifiez
        votre choix et indiquez votre niveau de confiance entre 0 et 1.
        """

    def build_prompt(self, activity: str) -> str:
        return f"Activité à classifier : {activity}"

    async def __call__(self, activity: str) -> MatchVerificationInput:
        result = await super().__call__(activity)
        # Same rationale as BaseClassifier._run_navigator_loop: the model is not a
        # reliable source for this field, so it's always overridden with the caller's
        # own known value.
        result.activity = activity
        return result
