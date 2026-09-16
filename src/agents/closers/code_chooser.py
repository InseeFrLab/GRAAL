from pydantic import BaseModel, Field

from src.agents.base_agent import BaseAgent
from src.neo4j_graph.graph import Graph


class CodeSelection(BaseModel):
    """Exactement ce que le modèle génère (même séparation que MatchVerification :
    un champ de ce schéma est un champ que le modèle remplira, y compris ceux qu'on
    lui demande de laisser vides)."""

    # Field order matches generation order for structured output (cf.
    # base_agent.py): the explanation is asked for before the code/confidence it
    # justifies, so the model's reasoning can actually inform the choice instead of
    # just rationalizing one it already committed to a few tokens earlier.
    explanation: str = Field(description="Concise explanation for the choice made")
    chosen_code: str = Field(description="Chosen code among the provided options")
    confidence: float = Field(
        description="Confidence level of the choice, between 0 and 1", ge=0, le=1
    )


class CodeChoice(CodeSelection):
    """Le choix du modèle, plus ce que Python compte autour de l'appel."""

    tool_call_count: int | None = None
    attempt_count: int | None = None

    def __str__(self):
        return self.model_dump_json()


class CodeChooser(BaseAgent):
    def __init__(self, graph: Graph, num_choices: int = 2):
        super().__init__(graph)
        self.num_choices = num_choices

    async def __call__(self, activity: str, codes: list[str]) -> CodeChoice:
        # We extend the base __call__ with the num_choices check
        if self.num_choices != len(codes):
            raise ValueError(f"Expected {self.num_choices} codes, got {len(codes)}")
        return await super().__call__(activity, codes)

    def get_agent_name(self) -> str:
        return "Code Chooser Agent"

    def get_tools(self):
        """Aucun outil : arbitrer entre des codes déjà retenus est un jugement en un
        coup, pas une exploration (même raison que MatchVerifier.get_tools)."""
        return []

    def get_max_turns(self) -> int:
        """Un seul tour : sans outil, il n'y a rien à faire d'un second."""
        return 1

    def wrap_output(self, model_output, result) -> CodeChoice:
        return CodeChoice(**model_output.model_dump())

    def get_instructions(self) -> str:
        return """
                Tu es un agent spécialisé dans le choix du code le plus approprié pour une activité donnée parmi plusieurs options.
            """

    def get_output_type(self):
        return CodeSelection

    def build_prompt(self, activity: str, codes: list[str]) -> str:
        """
        Build a prompt for the agent to choose between codes.

        Args:
            activity: The activity description
            codes: List of candidate codes

        Returns:
            str: The formatted prompt
        """
        # Notice complète de chaque candidat, plutôt que les outils du graphe : sans
        # outil (cf. get_tools), c'est la seule façon pour le modèle de comparer deux
        # codes sur autre chose que ce qu'il en mémorise — et elle est garantie présente,
        # là où un appel d'outil dépendait du bon vouloir du modèle.
        codes_text = "\n\n".join(
            f"- {code}\n{self.graph.get_notice(code) or 'Aucune notice trouvée dans la base.'}"
            for code in codes
        )

        return f"""L'activité à coder est : '{activity}'.

                Les codes candidats, avec leur notice officielle, sont :
                {codes_text}

                Choisissez le code le plus approprié parmi ces options en vous appuyant sur les notices ci-dessus, puis fournissez, dans cet ordre :
                1. Une explication concise qui raisonne sur votre choix avant de le formuler
                2. Le code choisi (exactement comme fourni dans la liste)
                3. Votre niveau de confiance (entre 0 et 1)

                Assurez-vous que le code choisi correspond exactement à l'un des codes fournis."""
