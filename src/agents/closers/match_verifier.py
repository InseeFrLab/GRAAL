import os
from pydantic import BaseModel, Field

from src.agents.base_agent import BaseAgent
from src.neo4j_graph.Graph import Graph

class MatchVerificationResult(BaseModel):
    is_match: bool = Field(description="Indicates whether the match is valid or not")
    confidence: float = Field(description="Confidence level of the verification, between 0 and 1", ge=0, le=1)
    explanation: str = Field(description="Concise explanation for the verification result")

    def __str__(self):
        return self.model_dump_json()

class MatchVerificationInput(BaseModel):
    activity: str = Field(description="The textual label of the activity to verify")
    proposed_code: str = Field(description="The code that has been associated with the activity")
    proposed_explanation: str = Field(description="The explanation provided for the proposed code")
    proposed_confidence: float = Field(description="The confidence level of the proposed match, between 0 and 1", ge=0, le=1)

class MatchVerifier(BaseAgent):
    def __init__(self, graph: Graph):
        super().__init__(graph)
    
    def get_agent_name(self) -> str:
        return "MatchVerifier Agent"
    
    def get_instructions(self) -> str:
        return """
                Tu es un agent spécialisé dans la vérification de la validité d'une correspondance entre un libellé textuel et le code qui lui a été associé.
            """

    def get_output_type(self):
        return MatchVerificationResult
    
    def build_prompt(self, match_verification_input: MatchVerificationInput) -> str:
        """
        Construire le prompt pour l'agent de vérification de correspondance.
        """
        prompt = f"""
        Vérifie si le code suivant correspond bien à l'activité décrite.

        Activité : {match_verification_input.activity}

        Code proposé : {match_verification_input.proposed_code}
        Explication proposée : {match_verification_input.proposed_explanation}

        Réponds en fournissant :
        1. Un booléen indiquant si la correspondance est valide.
        2. Un niveau de confiance entre 0 et 1.
        3. Une explication concise de ta décision.
        """
        return prompt