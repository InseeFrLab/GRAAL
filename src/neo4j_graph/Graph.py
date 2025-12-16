import logging
import os
from typing import Dict, Any, Optional, Tuple
from functools import lru_cache
from langchain_neo4j import Neo4jGraph
from agents import function_tool

logger = logging.getLogger(__name__)
CACHE_SIZE = os.getenv("GRAPH_CACHE_SIZE", 1000)


class Graph:
    def __init__(self, graph: Neo4jGraph):
        self.graph = graph

    @function_tool
    @lru_cache(maxsize=CACHE_SIZE)
    def get_code_information(self, code: str) -> Dict[str, Any]:
        """
        Get comprehensive information about a code node.

        Args:
            code: The code to look up (e.g., "10.71", "C")

        Returns:
            Dictionary with:
            - code, level, name
            - description, includes, excludes
            - parent_code, children_codes, children_count
        """
        query = """
        MATCH (node {CODE: $code})
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
        
        result = self.graph.query(query, params={"code": code})
        
        if result:
            return result[0]
        
        return {"error": f"Code {code} not found"}

    @function_tool
    @lru_cache(maxsize=CACHE_SIZE)
    def get_children(self, code: str) -> Tuple[Dict[str, Any], ...]:
        """
        Get all direct children of a code.

        Args:
            code: The parent code (e.g., "10", "C")

        Returns:
            Tuple of child nodes with code, level, name, description, includes, excludes
        """
        query = """
        MATCH (node {CODE: $code})-[:HAS_CHILD]->(child)
        RETURN child.CODE as code,
               child.LEVEL as level,
               child.NAME as name,
               child.text as description,
               child.Includes as includes,
               child.Excludes as excludes
        ORDER BY child.CODE
        """
        
        result = self.graph.query(query, params={"code": code})
        # Convert to tuple for hashability
        return tuple(result)

    @function_tool
    @lru_cache(maxsize=CACHE_SIZE)
    def get_descendants(self, code: str, levels: int = 2) -> Tuple[Dict[str, Any], ...]:
        """
        Get descendants of a code at a specific depth.

        Args:
            code: The ancestor code
            levels: How many levels down to traverse (default: 2)

        Returns:
            Tuple of descendant nodes with their information
        """
        query = f"""
        MATCH (node {{CODE: $code}})-[:HAS_CHILD*{levels}]->(descendant)
        RETURN descendant.CODE as code,
               descendant.LEVEL as level,
               descendant.NAME as name,
               descendant.text as description,
               descendant.Includes as includes,
               descendant.Excludes as excludes
        ORDER BY descendant.CODE
        """
        
        result = self.graph.query(query, params={"code": code})
        # Convert to tuple for hashability
        return tuple(result)

    @function_tool
    @lru_cache(maxsize=CACHE_SIZE)
    def get_siblings(self, code: str) -> Tuple[Dict[str, Any], ...]:
        """
        Get all sibling nodes (other children of this code's parent).

        Args:
            code: The code whose siblings to find

        Returns:
            Tuple of sibling nodes with their information
        """
        query = """
        MATCH (node {CODE: $code})<-[:HAS_CHILD]-(parent)
        MATCH (parent)-[:HAS_CHILD]->(sibling)
        WHERE sibling.CODE <> $code
        RETURN sibling.CODE as code,
               sibling.LEVEL as level,
               sibling.NAME as name,
               sibling.text as description,
               sibling.Includes as includes,
               sibling.Excludes as excludes
        ORDER BY sibling.CODE
        """
        
        result = self.graph.query(query, params={"code": code})
        # Convert to tuple for hashability
        return tuple(result)

    @function_tool
    @lru_cache(maxsize=CACHE_SIZE)
    def get_parent(self, code: str) -> Optional[Dict[str, Any]]:
        """
        Get the parent of a code.

        Args:
            code: The child code

        Returns:
            Parent node information or None if no parent exists
        """
        query = """
        MATCH (node {CODE: $code})<-[:HAS_CHILD]-(parent)
        RETURN parent.CODE as code,
               parent.LEVEL as level,
               parent.NAME as name,
               parent.text as description
        """
        
        result = self.graph.query(query, params={"code": code})
        
        return result[0] if result else None

    @function_tool
    @lru_cache(maxsize=CACHE_SIZE)
    def search_codes(self, search_term: str) -> Tuple[Dict[str, Any], ...]:
        """
        Search for codes by name or description.

        Args:
            search_term: Text to search for

        Returns:
            Tuple of matching codes with their information
        """
        query = """
        MATCH (node)
        WHERE toLower(node.NAME) CONTAINS toLower($search_term)
           OR toLower(node.text) CONTAINS toLower($search_term)
        RETURN node.CODE as code,
               node.LEVEL as level,
               node.NAME as name,
               node.text as description
        ORDER BY node.LEVEL, node.CODE
        LIMIT 20
        """
        
        result = self.graph.query(query, params={"search_term": search_term})
        # Convert to tuple for hashability
        return tuple(result)
    
    def clear_cache(self):
        """Clear all cached query results"""
        self.get_code_information.cache_clear()
        self.get_children.cache_clear()
        self.get_descendants.cache_clear()
        self.get_siblings.cache_clear()
        self.get_parent.cache_clear()
        self.search_codes.cache_clear()
        logger.info("Cache cleared")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get cache statistics for all cached methods"""
        return {
            "get_code_info": self.get_code_information.cache_info()._asdict(),
            "get_children": self.get_children.cache_info()._asdict(),
            "get_descendants": self.get_descendants.cache_info()._asdict(),
            "get_siblings": self.get_siblings.cache_info()._asdict(),
            "get_parent": self.get_parent.cache_info()._asdict(),
            "search_codes": self.search_codes.cache_info()._asdict(),
        }