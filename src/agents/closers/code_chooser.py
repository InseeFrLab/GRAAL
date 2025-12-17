from pydantic import BaseModel, Field

from agents import Runner
from src.agents.base_agent import BaseAgent
from src.neo4j_graph.Graph import Graph


class CodeChoice(BaseModel):
    chosen_code: str = Field(description="Chosen code among the provided options")
    confidence: float = Field(
        description="Confidence level of the choice, between 0 and 1", ge=0, le=1
    )
    explanation: str = Field(description="Concise explanation for the choice made")

    def __str__(self):
        return self.model_dump_json()


class CodeChooser(BaseAgent):
    def __init__(self, graph: Graph, num_choices: int = 2):
        super().__init__(graph)
        self.num_choices = num_choices

    async def __call__(self, activity: str, codes: list[str]) -> str:
        # We rewrite the base __call__ to add the num_choices check
        if self.num_choices != len(codes):
            raise ValueError(f"Expected {self.num_choices} codes, got {len(codes)}")

        prompt = self.build_prompt(activity, codes)
        result = await Runner.run(self.agent, prompt)

        return result

    def get_agent_name(self) -> str:
        return "Code Chooser Agent"

    def get_instructions(self) -> str:
        return """
                Tu es un agent spécialisé dans le choix du code le plus approprié pour une activité donnée parmi plusieurs options.
            """

    def get_output_type(self):
        return CodeChoice

    def build_prompt(self, activity: str, codes: list[str]) -> str:
        """
        Build a prompt for the agent to choose between codes.

        Args:
            activity: The activity description
            codes: List of candidate codes

        Returns:
            str: The formatted prompt
        """
        codes_text = "\n".join([f"- {code}" for code in codes])

        return f"""L'activité à coder est : '{activity}'.

                Les codes candidats sont :
                {codes_text}

                Choisissez le code le plus approprié parmi ces options. Analysez chaque code en utilisant les outils disponibles si nécessaire, puis fournissez :
                1. Le code choisi (exactement comme fourni dans la liste)
                2. Votre niveau de confiance (entre 0 et 1)
                3. Une explication concise de votre choix

                Assurez-vous que le code choisi correspond exactement à l'un des codes fournis."""
