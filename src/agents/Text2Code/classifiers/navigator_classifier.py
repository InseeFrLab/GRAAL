import logging

from agents import Runner
from src.agents.closers.match_verifier import MatchVerificationInput
from src.agents.Text2Code.classifiers.base_classifier import BaseClassifier

logger = logging.getLogger(__name__)


class NavigatorAgenticClassifier(BaseClassifier):
    def __init__(self, navigator):
        self.navigator = navigator
        super().__init__(navigator)

    def get_agent_name(self) -> str:
        return "Navigator Agentic Classifier"

    def get_output_type(self):
        # No structured output: let the agent freely use tools and return text.
        # The final NACE code is read from navigator.current_code after the run.
        return None

    def build_prompt(self, query: str) -> str:
        return f"""
        Activité à classifier : {query}

        Naviguez dans la hiérarchie NACE pour trouver le code le plus spécifique et approprié.
        Quand vous avez atteint un nœud final (is_final = 1), indiquez le code retenu et expliquez votre choix.
        """

    def get_instructions(self) -> str:
        return """
        Vous êtes un expert en classification NACE. Votre mission est de naviguer
        dans l'arborescence afin d'atteindre le code le plus spécifique caractérisant l'activité indiquée.

        Processus à suivre :
        1. Appelez get_current_children() pour voir les sections disponibles à la racine
        2. Naviguez vers la section la plus pertinente avec go_to_child(code)
        3. Répétez jusqu'à atteindre un nœud avec is_final = 1
        4. Une fois arrivé, expliquez brièvement pourquoi ce code correspond à l'activité

        Si vous n'avez pas réussi à atteindre une position finale, dites-le.
        Soyez méthodique et justifiez chaque choix !
        """

    async def __call__(self, query: str) -> MatchVerificationInput:
        import os
        prompt = self.build_prompt(query)
        result = await Runner.run(
            self.agent,
            prompt,
            max_turns=int(os.environ["MAX_TURNS"]),
        )
        explanation = result.final_output or ""
        logger.info(f"Navigator agent final text: {explanation}")
        logger.info(f"Navigator final position: {self.navigator.current_code}")

        return MatchVerificationInput(
            activity=query,
            code=self.navigator.current_code,
            proposed_explanation=explanation,
            proposed_confidence=1.0 if self.navigator.current_code != "root" else 0.0,
        )
