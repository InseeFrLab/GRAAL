import logging
from typing import Dict, Any, List, Optional
from langchain_neo4j import Neo4jGraph

from neo4j.CachedNeo4j import CachedNeo4jQuery

logger = logging.getLogger(__name__)

# TODO: Use the tools already configured 


class HierarchicalNavigator:
    def __init__(self, graph: Neo4jGraph, root: str = "root", cache_size: int = 1000):
        self.node_code = root
        self.history = [root]  # Track navigation history
        self.cached_graph = CachedNeo4jQuery(graph)

    def _get_children_information(self, node_code: str, levels: int = 1) -> List[Dict]:
        if levels == 1:
            query = """
            MATCH (node {CODE: $node_code})-[:HAS_CHILD]->(child)
            RETURN child.CODE as code,
                   child.LEVEL as level,
                   child.NAME as name,
                   child.text as description,
                   child.Includes as includes,
                   child.Excludes as excludes
            ORDER BY child.CODE
            """
        else:
            query = """
            MATCH (node {CODE: $node_code})-[:HAS_CHILD*{levels}]->(descendant)
            RETURN descendant.CODE as code,
                   descendant.LEVEL as level,
                   descendant.NAME as name,
                   descendant.text as description,
                   descendant.Includes as includes,
                   descendant.Excludes as excludes
            ORDER BY descendant.CODE
            """.format(levels=levels)
        
        return self.cached_graph.query(query, params={"node_code": node_code})
        
    def _get_parent_information(self, node_code: str, levels: int = 1) -> Optional[Dict]:
        if levels == 1:
            query = """
            MATCH (node {CODE: $node_code})<-[:HAS_CHILD]-(parent)
            RETURN parent.CODE as code, 
                   parent.LEVEL as level, 
                   parent.NAME as name,
                   parent.text as description,
            """
        else:
            query = """
            MATCH (node {CODE: $node_code})<-[:HAS_CHILD*{levels}]-(ancestor)
            RETURN ancestor.CODE as code,
                   ancestor.LEVEL as level,
                   ancestor.NAME as name,
                   ancestor.text as description,
            """.format(levels=levels)
        
        result = self.cached_graph.query(query, params={"node_code": node_code})
        return result[0] if result and len(result) > 0 else None

    def _generate_position_summary(self, current: Dict, children: Dict, siblings: Dict) -> str:
        """Generate a human-readable summary of the current position"""
        summary_parts = []
        
        if current.get("code"):
            summary_parts.append(f"Currently at node '{current['code']}' (Level {current.get('level')})")
        elif children["has_children"]:
            summary_parts.append(f"Has {children['children_count']} child node(s) - can go deeper")
        else:
            summary_parts.append("No children - at a leaf node")
        
        if siblings["siblings_count"] > 0:
            summary_parts.append(f"Has {siblings['siblings_count']} sibling(s) at the same level")
        
        return ". ".join(summary_parts)
    
    def go_up(self, levels: int = 1) -> Dict[str, Any]:
        """Navigate up the hierarchy and return parent information with context"""
        parent_info = self._get_parent_information(self.node_code, levels)
        
        if parent_info and levels == 1:
            parent_code = parent_info["code"]
            self.node_code = parent_code
            self.history.append(parent_code)
            logger.info(f"📤 Moved up to: {parent_code}")
        
        return {
            "success": parent_info is not None,
            "parent": parent_info,
            "current_position": self.node_code,
            "navigation_depth": len(self.history)
        }
    
    def get_children(self, levels: int = 1) -> Dict[str, Any]:
        """Get children of the current node with detailed information"""
        children = self._get_children_information(self.node_code, levels)
        
        return {
            "current_node": self.node_code,
            "children_count": len(children),
            "children": children,
            "has_children": len(children) > 0
        }
    
    def get_current_node(self) -> Dict[str, Any]:
        """Get comprehensive information about the current node"""
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
        result = self.cached_graph.query(query, params={"node_code": node_code})
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
    
    def go_down(self, node_code: str) -> Dict[str, Any]:
        """Navigate down to a specific child node with validation"""
        # Use cached children to validate
        children = self._get_children_information(self.node_code)
        child_codes = [child["code"] for child in children]
        
        if node_code in child_codes:
            # Get detailed info about the target node
            target_info = next((child for child in children if child["code"] == node_code), None)
            
            self.node_code = node_code
            self.history.append(node_code)
            logger.info(f"📥 Moved down to: {node_code}")
            
            return {
                "success": True,
                "node": target_info,
                "current_position": self.node_code,
                "navigation_depth": len(self.history)
            }
        else:
            return {
                "success": False,
                "error": f"Node {node_code} is not a direct child of {self.node_code}",
                "current_position": self.node_code,
                "available_children": child_codes
            }
    
    def get_siblings(self) -> Dict[str, Any]:
        """
        Get siblings of the current node (other children of the parent).
        This helps explore alternatives at the same level.
        """
        query = """
            MATCH (current {CODE: $node_code})<-[:HAS_CHILD]-(parent)-[:HAS_CHILD]->(sibling)
            WHERE sibling.CODE <> $node_code
            RETURN sibling.CODE as code,
                sibling.LEVEL as level,
                sibling.NAME as name,
                sibling.text as description,
                sibling.Includes as includes,
                sibling.Excludes as excludes
            ORDER BY sibling.CODE
        """
        result = self.cached_graph.query(query, params={"node_code": self.node_code})
        siblings = result

        return {
            "current_node": self.node_code,
            "siblings_count": len(siblings),
            "siblings": siblings,
            "message": f"Found {len(siblings)} sibling(s) of {self.node_code}"
        }
    
    def get_context_summary(self) -> Dict[str, Any]:
        """
        Get a comprehensive summary of the current position in the hierarchy.
        Includes parent, current node, children, and siblings.
        """
        current = self.get_current_node()
        children = self.get_children()
        siblings = self.get_siblings()
        
        return {
            "current_node": {
                "code": current.get("code"),
                "name": current.get("name"),
                "level": current.get("level"),
                "description": current.get("description")[:200] + "..." if current.get("description") and len(current.get("description", "")) > 200 else current.get("description")
            },
            "parent_code": current.get("parent_code"),
            "children_count": children["children_count"],
            "siblings_count": siblings["siblings_count"],
            "navigation_path": " → ".join(self.history[-5:]),
            "can_go_deeper": children["has_children"],
            "summary": self._generate_position_summary(current, children, siblings)
        }
    
    def reset_to_root(self) -> Dict[str, Any]:
        """Reset navigation to root"""
        self.node_code = self.history[0]
        self.history = [self.history[0]]
        logger.info("🔄 Reset to root")
        return {
            "message": "Navigation reset to root",
            "current_position": self.node_code
        }

    def get_path_to_root(self) -> Dict[str, Any]:
        """Get the full path from current node to root"""
        query = """MATCH path = (node {CODE: $node_code})<-[:HAS_CHILD*]-(root {CODE: 'root'})
            RETURN [n in nodes(path) | {code: n.CODE, name: n.NAME, level: n.LEVEL}] as path"""
        result = self.cached_graph.query(query, params={"node_code": self.node_code})
        path = result[0]["path"] if result else None
        
        if path:
            return {
                "current_node": self.node_code,
                "path_to_root": path,
                "depth": len(path)
            }
        
        return {"error": "Could not find path to root"}