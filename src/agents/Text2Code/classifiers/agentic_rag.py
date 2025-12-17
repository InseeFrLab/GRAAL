from src.agents.Text2Code.classifiers.base_classifier import BaseClassifier
from src.agents.base_agent import BaseAgent
from src.agents.closers.code_chooser import CodeChooser
from src.agents.closers.match_verifier import MatchVerificationInput

class AgenticRAGClassifier(BaseClassifier):

    """
    1. Retrive the top_k closest codes from the graph
    2. A CodeChooser agent to select the most appropriate code for the given activity.
    
    Output is a CodeChoice, which is converted in a MatchVerificationInput as per requirements for classifiers.

    """

    def __init__(self, graph, top_k):
        super().__init__(graph)
        self.top_k = top_k
        self.code_chooser = CodeChooser(graph, num_choices=top_k)
    
    def __call__(self, activity: str) -> str:
        closest_codes = self.get_closest_codes(activity)
        code_choice_result = self.code_chooser(activity=activity, codes=closest_codes)

        # We need to convert the CodeChoice into MatchVerificationInput
        # because all classifiers must return that type
        result = MatchVerificationInput(
            activity=activity,
            proposed_code=code_choice_result.chosen_code,
            proposed_explanation=code_choice_result.explanation
        )
        
        return result
    
    def get_agent_name(self) -> str:
        return "Agentic RAG Classifier"
    
    def get_instructions(self) -> str:
        return None
    
    def get_closest_codes(self, activity):
        return self.graph.get_closest_codes(activity, top_k=self.top_k)
    
    def build_prompt(self):
        return None