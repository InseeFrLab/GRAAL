import logging

from src.agents.closers.match_verifier import MatchVerificationInput
from src.agents.Text2Code.classifiers.base_classifier import BaseClassifier
from src.navigator.navigator import Navigator

logger = logging.getLogger(__name__)


class AgenticRAGClassifier(BaseClassifier):
    """
    1. Retrieve the single closest code from the graph via embedding similarity search.
    2. Use it as a warm-start position for the Navigator agent, which verifies it against
       the hierarchy (siblings, parent, children, is_final) and refines it if needed.

    Output is a MatchVerificationInput, produced directly by the agent's structured output
    (same contract as NavigatorAgenticClassifier).

    Confidence of AgenticRAGClassifier is the confidence given by the agent itself.
    """

    def __init__(self, graph: Navigator):
        super().__init__(graph)

    async def __call__(self, activity: str) -> MatchVerificationInput:
        start_code = await self.get_starting_code(activity)
        logger.info(f"RAG starting point for activity '{activity}': {start_code}")

        self.graph.current_code = start_code
        self.graph.history = [start_code]

        return await super().__call__(activity, start_code)

    def get_agent_name(self) -> str:
        return "Agentic RAG Classifier"

    def get_instructions(self) -> str:
        return """
        Vous êtes un expert en classification NACE.

        Une recherche par similarité d'embeddings vous propose un point de départ dans
        l'arborescence, mais cette suggestion peut être imprécise ou erronée : ne la validez
        jamais aveuglément. Vérifiez-la avec les outils disponibles (informations du noeud
        courant, enfants, frères, parent) avant de conclure.

        Si le point de départ semble incorrect, naviguez vers une meilleure position (remontez
        au parent, explorez les frères ou les enfants) plutôt que de le valider tel quel.

        Après avoir vérifié que votre position finale est bien terminale (is_final = 1),
        renvoyez le code choisi, une explication concise et votre niveau de confiance.
        Commencez par get_current_information() pour examiner le point de départ proposé.

        RÈGLE STRICTE : il est interdit de répondre avec un code dont is_final = 0, ou
        avec le point de départ suggéré si vous avez constaté qu'il ne correspond pas à
        l'activité. Si votre position actuelle n'est pas satisfaisante, vous DEVEZ appeler
        un autre outil (get_current_children, go_to_child, go_to_parent...) avant de
        conclure — ne vous arrêtez jamais après un seul appel d'outil si is_final n'est
        pas à 1 ou si le code ne correspond pas à l'activité décrite.
        """

    async def get_starting_code(self, activity: str) -> str:
        closest_codes = await self.graph.get_closest_codes(activity, top_k=1)
        return closest_codes[0]

    def build_prompt(self, activity: str, start_code: str) -> str:
        return f"""
        Vous êtes un classificateur NACE.

        Activité à classifier : {activity}

        Un système de recherche par similarité d'embeddings suggère de partir du code
        {start_code} comme point de départ probable. Ce point de départ n'est qu'une
        suggestion : il peut être imprécis ou erroné.

        Votre mission : vérifier rigoureusement ce point de départ et naviguer si nécessaire
        vers le code le plus spécifique et approprié pour cette activité, jusqu'à atteindre
        une position finale (is_final = 1).
        """
