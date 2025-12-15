import logging
from typing import Dict, Any, List, Optional
from langchain_neo4j import Neo4jGraph
from langchain.tools import tool


from neo4j.CachedNeo4j import CachedNeo4jQuery

logger = logging.getLogger(__name__)

# TODO: Manage the object graph, setup graph necessary
# TODO: Create Node Class

class Graph:
    def __init__(self, graph: Neo4jGraph, cache_size: int = 1000):
        self.cached_graph = CachedNeo4jQuery(graph, cache_size)

    @tool
    def get_node_information(node: Node) -> str:
        """
        Get comprehensive information about a node.

        Returns detailed information including:
        - Code, level, and name
        - Full description and classification rules
        - What this code includes and excludes
        - Parent code and children count
        - Navigation history

        Use this tool first to understand where you are in the hierarchy.
        """
        query = """
        MATCH (node {CODE: $node_code})
        OPTIONAL MATCH (node)<-[:HAS_CHILD]-(parent)
        OPTIONAL MATCH (node)-[:HAS_CHILD]->(child)
        WITH node, parent, collect(child.CODE) as children_codes
        RETURN node.CODE as code,
               node.LEVEL as level,
               node.NAME as name,
               node.text as description,
               node.Includes as includes,
               node.IncludesAlso as includes_also,
               node.Excludes as excludes,
               node.Implementation_rule as implementation_rule,
               parent.CODE as parent_code,
               children_codes,
               size(children_codes) as children_count
        """
        result = self.cached_graph.query(query)
        node_info = result[0] if result else None
        
        if node_info:
            return {
                "code": node_info["code"],
                "level": node_info["level"],
                "name": node_info["name"],
                "description": node_info["description"],
                "includes": node_info["includes"],
                "includes_also": node_info["includes_also"],
                "excludes": node_info["excludes"],
                "implementation_rule": node_info["implementation_rule"],
                "parent_code": node_info["parent_code"],
                "children_codes": node_info["children_codes"],
                "children_count": node_info["children_count"],
                "navigation_history": self.history[-5:]  # Last 5 steps
            }
        
        return {"error": f"Node {self.node_code} not found"}
    
        return json.dumps(result, ensure_ascii=False, indent=2)


    @tool
    def get_children() -> str:
        """
        Get all direct children of the current node with their detailed information.

        Returns:
        - List of all child nodes with codes, names, and descriptions
        - What each child includes/excludes
        - Whether each child is a final classification code

        Use this tool to explore what options are available at the next level down.
        """
        result = navigator.get_children()
        return json.dumps(result, ensure_ascii=False, indent=2)


    @tool
    def get_siblings() -> str:
        """
        Get all sibling nodes (other children of the current node's parent).

        This helps you explore alternatives at the same level of the hierarchy.
        Useful when the current branch doesn't seem to fit the activity being classified.

        Returns:
        - List of sibling nodes with codes, names, and descriptions
        - What each sibling includes/excludes
        """
        result = navigator.get_siblings()
        return json.dumps(result, ensure_ascii=False, indent=2)


    @tool
    def get_context_summary() -> str:
        """
        Get a comprehensive overview of your current position in the hierarchy.

        This is the most useful tool for understanding the full context:
        - Where you are (current node details)
        - Where you can go (children count)
        - Alternative options (siblings count)
        - Navigation path taken so far
        - Whether you can go deeper or if this is a final node

        Use this tool when you need to make a decision about navigation direction.
        """
        result = navigator.get_context_summary()
        return json.dumps(result, ensure_ascii=False, indent=2)


    @tool
    def go_down(node_code: str) -> str:
        """
        Navigate down to a specific child node.

        Args:
            node_code: The CODE of the child node to navigate to (e.g., "10.71", "C")

        Returns:
            Success/failure information and details about the new current node.

        Only call this after using get_children() to see available options.
        The node_code must be a direct child of the current node.
        """
        result = navigator.go_down(node_code)
        return json.dumps(result, ensure_ascii=False, indent=2)


    @tool
    def go_up() -> str:
        """
        Navigate up one level to the parent node.

        Use this tool when:
        - You want to backtrack and explore a different branch
        - The current branch doesn't match the activity
        - You want to reconsider your navigation choices

        Returns:
            Information about the parent node and new current position.
        """
        result = navigator.go_up()
        return json.dumps(result, ensure_ascii=False, indent=2)
