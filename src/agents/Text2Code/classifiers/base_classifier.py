from abc import ABC, abstractmethod

from src.agents.base_agent import BaseAgent
from src.agents.closers.match_verifier import MatchVerificationInput
from src.neo4j_graph.Graph import Graph


class BaseClassifier(BaseAgent):
    def get_output_type(self):
        return MatchVerificationInput