from agents.model_settings import ModelSettings
from src.agents.base_agent import BaseAgent
from src.agents.closers.match_verifier import MatchVerificationInput


class BaseClassifier(BaseAgent):
    def get_output_type(self):
        return MatchVerificationInput

    def get_model_settings(self) -> ModelSettings:
        return ModelSettings(temperature=0, tool_choice="required")
