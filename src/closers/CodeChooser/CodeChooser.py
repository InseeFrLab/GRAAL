import os

from dotenv import load_dotenv
from agents import Agent, Runner
from pydantic import BaseModel, Field
from typing import Literal

from judge import judge_no_agentic_prompt
from llm.client import sync_get_llm_client
from tools import graph, tools


client = sync_get_llm_client()
load_dotenv()

class CodeChoice(BaseModel):
    chosen_code: str = Field(description="Chosen code among the provided options")
    confidence: float = Field(description="Confidence level of the choice, between 0 and 1", ge=0, le=1)
    explanation: str = Field(description="Concise explanation for the choice made")

    def __str__(self):
        return self.model_dump_json()

class CodeChooser(Closer):
    def __init__(self, tools, num_choices: int = 2):
        
        self.agent = Agent(
            name = "Code Chooser Agent",
            instructions="""
You are an expert in choosing the most appropriate code for a given activity based on provided options.
            """,
            tools=tools,
            model=os.environ["GENERATION_MODEL"],
            model_settings={
                "temperature": 0,
            },
            output_type=CodeChoice
        )
        self.num_choices = num_choices
    
    def __call__(self, activity: str, codes:list[str]) -> str:
        # --- Create agent ---

        if self.num_choices != len(codes):
            raise ValueError(f"Expected {self.num_choices} codes, got {len(codes)}")
        
        # Build the prompt
        prompt = self.build_prompt(activity, codes)
        
        # Run the agent with the prompt
        runner = Runner(agent=self.agent)
        result = runner.run(prompt)
        
        return result.output
        
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
