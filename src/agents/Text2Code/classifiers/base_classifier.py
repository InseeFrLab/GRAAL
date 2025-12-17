from src.agents.base_agent import BaseAgent
from src.agents.closers.match_verifier import MatchVerificationInput
from src.neo4j_graph.Graph import Graph


class BaseClassifier(BaseAgent):
    def __init__(self, graph: Graph):
        super().__init__(graph)

    def get_output_type(self):
        return MatchVerificationInput