from pydantic import BaseModel, Field

from src.agents.base_agent import BaseAgent
from src.neo4j_graph.graph import Graph


class Code2TextAgentOutput(BaseModel):
    code: str = Field(description="The code for which the activity has been generated")
    generated_description: str = Field(
        description="A typical activity description that would correspond to the code"
    )


class Code2TextAgent(BaseAgent):
    def __init__(self, graph: Graph):
        super().__init__(graph)

    def get_agent_name(self) -> str:
        return "Code2Text Agent - Synthetic Data Generator"

    def get_output_type(self):
        return Code2TextAgentOutput

    def get_instructions(self) -> str:
        return """Tu es un expert en génération de descriptions textuelles à partir de codes NAF.

        Tu te mets dans la peau d'un chef d'entreprise décrivant son activité qui corrrepondrait au code NAF demandé.

        INSTRUCTIONS :
        1. La description doit être typique de celle déclarée par un chef d'entreprise au Guichet Unique. Ne sois pas verbeux.
        2. Fais particulièrement attention à ce que le texte corresponde exactement au code, et n'empiète pas sur d'autres codes
        3. Tu ne dois pas simplement répéter la description du code, mais paraître naturel.

        IMPORTANT :
        - Vérifie que ta description correspond exactement au périmètre du code donné
        - Consulte la nomenclature si nécessaire pour t'assurer de la précision
        """

    def build_prompt(self, code):
        prompt = f"""CODE À DÉCRIRE : {code}

        Génère maintenant la description textuelle pour le code fourni"""

        return prompt
