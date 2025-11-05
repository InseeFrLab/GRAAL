import logging

from langchain_neo4j import Neo4jGraph

logger = logging.getLogger(__name__)


class HierarchicalNavigator:
    def __init__(self, graph: Neo4jGraph, root: str = "root"):
        self.graph = graph
        self.node_code = root

    def go_up(self, levels=1) -> dict:
        """Navigate up the hierarchy"""
        if levels == 1:
            query = """
            MATCH (node {code: $node_code})<-[:HAS_CHILD]-(parent)
            RETURN parent
            """
        else:
            query = """
            MATCH (node {{code: $node_code}})<-[:HAS_HILD*{levels}]-(ancestor)
            RETURN ancestor
            """.format(levels=levels)

        result = self.graph.query(query, params={"node_code": self.node_code})

        if result and levels == 1:
            self.node_code = result[0]["parent"]["code"]

        return result

    def get_children(self, levels=1) -> dict:
        """Navigate down the hierarchy"""
        if levels == 1:
            query = """
            MATCH (node {code: $node_code})-[:HAS_CHILD]->(child)
            RETURN child
            """
        else:
            query = """
            MATCH (node {{code: $node_code}})-[:HAS_CHILD*{levels}]->(descendant)
            RETURN descendant
            """.format(levels=levels)

        return self.graph.query(query, params={"node_code": self.node_code})

    def get_current_node(self) -> dict:
        """Get information about the current node"""
        query = """
        MATCH (node {code: $node_code})
        RETURN node
        """
        return self.graph.query(query, params={"node_code": self.node_code})

    def go_down(self, node: str) -> dict:
        """Go down to the target node"""
        query = """
        MATCH (node {code: $node_code})
        RETURN node
        """
        result = self.graph.query(query, params={"node_code": self.node_code})

        if result:
            self.node_code = result[0]["code"]

        return result
