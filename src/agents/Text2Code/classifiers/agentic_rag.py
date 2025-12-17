from src.agents.Text2Code.classifiers.base_classifier import BaseClassifier
from src.agents.base_agent import BaseAgent
from src.agents.closers.code_chooser import CodeChooser
from src.agents.closers.match_verifier import MatchVerificationInput
import logging

logger = logging.getLogger(__name__)

class AgenticRAGClassifier(BaseClassifier):

    """
    1. Retrive the top_k closest codes from the graph
    2. A CodeChooser agent to select the most appropriate code for the given activity.
    
    Output is a CodeChoice, which is converted in a MatchVerificationInput as per requirements for classifiers.

    Confidence of AgenticRAGClassifier is the confidence given by the CodeChooser.

    """

    def __init__(self, graph, top_k):
        super().__init__(graph)
        self.top_k = top_k
        self.code_chooser = CodeChooser(graph, num_choices=top_k)
    
    async def __call__(self, activity: str) -> str:
        closest_codes = await self.get_closest_codes(activity)
        logger.info(f"Closest codes for activity '{activity}': {closest_codes}")
        code_choice_result = await self.code_chooser(activity=activity, codes=closest_codes)

        # We need to convert the CodeChoice into MatchVerificationInput
        # because all classifiers must return that type
        code_choice_result = code_choice_result.final_output
        result = MatchVerificationInput(
            activity=activity,
            proposed_code=code_choice_result.chosen_code,
            proposed_explanation=code_choice_result.explanation,
            proposed_confidence=code_choice_result.confidence  # confidence from CodeChoice
        )
        
        return result
    
    def get_agent_name(self) -> str:
        return "Agentic RAG Classifier"
    
    def get_instructions(self) -> str:
        return None
    
    async def get_closest_codes(self, activity):
        return await self.graph.get_closest_codes(activity, top_k=self.top_k)
    
    def build_prompt(self):
        return None