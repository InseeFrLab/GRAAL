from src.agents.base_agent import BaseAgent
from src.agents.closers.match_verifier import MatchVerificationInput


class BaseClassifier(BaseAgent):
    def get_output_type(self):
        return MatchVerificationInput
