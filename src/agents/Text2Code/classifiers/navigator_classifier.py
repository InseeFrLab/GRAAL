from src.agents.Text2Code.classifiers.base_classifier import BaseClassifier


class NavigatorAgenticClassifier(BaseClassifier):
    def __init__(self, navigator):
        super().__init__(navigator)

    async def __call__(self, query: str):
        return await self._run_navigator_loop(self.build_prompt(query))

    def get_agent_name(self) -> str:
        return "Navigator Agentic Classifier"

    def build_prompt(self, query: str) -> str:
        return f"""
        Vous êtes un classificateur NACE.

        Activité à classifier : {query}

        Votre mission : Naviguer dans la hiérarchie NACE pour trouver le code le plus spécifique et approprié.
        """

    def get_instructions(self) -> str:
        return """
        Vous êtes un expert en classification NACE. Votre mission est de naviguer
        dans l'arborescence afin d'atteindre le code le plus spécifique caractérisant l'activité indiquée.
        Soyez méthodique et justifiez chaque choix ! Commencez par get_current_children() pour voir les
        options disponibles.
        """
