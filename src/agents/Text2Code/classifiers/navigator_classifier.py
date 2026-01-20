from src.agents.Text2Code.classifiers.base_classifier import BaseClassifier


class NavigatorAgenticClassifier(BaseClassifier):
    def __init__(self, navigator):
        super().__init__(navigator)

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
        Après avoir vérifié que vous êtes au niveau 4 de l'arbre, vous renverrez votre position finale en appelant l'outil submit_classification.
        ## FORMAT DE SORTIE
        ```json
        {
            "best_code": "XX.XX.X",
            "path": ["root", "A", "01", "01.1", "01.11"],
            "confidence": "high|medium|low",
            "reasoning": "Justification détaillée basée sur includes/excludes",
            "alternatives": ["XX.XX.Y", "XX.XX.Z"]
        }
        ```


        Soyez méthodique et justifiez chaque choix !
        """