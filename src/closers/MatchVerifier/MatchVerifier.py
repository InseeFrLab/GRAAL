import os
from src.closers import Closer
from pydantic import BaseModel, Field
from agents import Agent, Runner
from src.tools import graph, tools

class MatchVerificationResult(BaseModel):
    is_match: bool = Field(description="Indicates whether the match is valid or not")
    confidence: float = Field(description="Confidence level of the verification, between 0 and 1", ge=0, le=1)
    explanation: str = Field(description="Concise explanation for the verification result")

    def __str__(self):
        return self.model_dump_json()

class MatchVerifier(Closer):
    def __init__(self, graph):

        self.graph = graph
        self.agent = Agent(
            name = "Code Chooser Agent",
            instructions="""
                Tu es un agent spécialisé dans la vérification de la validité d'une correspondance entre un libellé textuel et le code qui lui a été associé.
            """,
            tools=self.graph.get_tools(),
            model=os.environ["GENERATION_MODEL"],
            model_settings={
                "temperature": 0,
            },
            output_type=MatchVerificationResult
        )
    
    def __call__(self, activity: str, code: str) -> MatchVerificationResult:
        # Build the prompt
        prompt = self.build_prompt(activity, code)
        
        # Run the agent with the prompt
        runner = Runner(agent=self.agent)
        result = runner.run(prompt)
        
        return result.output
    
    def build_prompt(self, activity: str, code: str) -> str:
        """
        Construire le prompt pour l'agent de vérification de correspondance.
        """
        prompt = f"""
        Vérifie si le code suivant correspond bien à l'activité décrite.

        Activité : {activity}

        Code proposé : {code}

        Réponds en fournissant :
        1. Un booléen indiquant si la correspondance est valide.
        2. Un niveau de confiance entre 0 et 1.
        3. Une explication concise de ta décision.
        """
        return prompt