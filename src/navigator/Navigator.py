import logging
from typing import Dict, Any, List, Optional
from langchain_neo4j import Neo4jGraph
from agents import function_tool

from neo4j_graph.Graph import Graph

logger = logging.getLogger(__name__)


class Navigator:
    """
    Classe de navigation dans la hiérarchie NACE avec état persistant.
    Utilise Graph pour les requêtes et maintient la position courante.
    """
    
    def __init__(self, graph: Neo4jGraph, root: str = "root"):
        self.current_code = root
        self.history = [root]
        self.graph = Graph(graph)

    # ------------------------------------------------------------------
    # Information methods
    # ------------------------------------------------------------------

    @function_tool
    def get_current_information(self) -> Dict[str, Any]:
        """
        Retourne les informations du noeud actuel.
        
        Returns:
            Informations complètes du noeud courant avec historique de navigation
        """
        info = self.graph.get_code_information(self.current_code)
        if "error" not in info:
            info["navigation_history"] = self.history[-5:]
        return info

    @function_tool
    def get_code_information(self, code: str) -> Dict[str, Any]:
        """
        Retourne les informations d'un code spécifique sans changer la position.
        
        Args:
            code: Code NACE à consulter
        
        Returns:
            Informations complètes du code
        """
        return self.graph.get_code_information(code)

    @function_tool
    def get_current_children(self) -> List[Dict[str, Any]]:
        """
        Retourne les enfants directs du noeud actuel.
        
        Returns:
            Liste des codes enfants du noeud courant
        """
        return self.graph.get_children(self.current_code)

    @function_tool
    def get_current_siblings(self) -> List[Dict[str, Any]]:
        """
        Retourne les codes au même niveau que le noeud actuel.
        
        Returns:
            Liste des siblings du noeud courant
        """
        return self.graph.get_siblings(self.current_code)

    @function_tool
    def get_descendants(self, levels: int = 2) -> List[Dict[str, Any]]:
        """
        Retourne les descendants du noeud actuel jusqu'à N niveaux.
        
        Args:
            levels: Nombre de niveaux à descendre (défaut: 2)
        
        Returns:
            Liste de tous les descendants
        """
        return self.graph.get_descendants(self.current_code, levels)

    # ------------------------------------------------------------------
    # Navigation methods
    # ------------------------------------------------------------------

    @function_tool
    def navigate_to(self, code: str) -> Dict[str, Any]:
        """
        Se déplace vers un code spécifique.
        
        Args:
            code: Code NACE de destination
        
        Returns:
            Résultat de la navigation avec informations du nouveau noeud
        """
        info = self.graph.get_code_information(code)
        
        if "error" in info:
            return {
                "success": False,
                "error": f"Code {code} not found",
                "current_position": self.current_code
            }
        
        self.current_code = code
        self.history.append(code)
        logger.info(f"Navigated to: {code}")
        
        return {
            "success": True,
            "node": info,
            "current_position": self.current_code,
            "navigation_depth": len(self.history)
        }

    @function_tool
    def go_to_parent(self) -> Dict[str, Any]:
        """
        Remonte au parent du noeud actuel.
        
        Returns:
            Résultat de la navigation avec informations du parent
        """
        parent_info = self.graph.get_parent(self.current_code)
        
        if parent_info is None:
            return {
                "success": False,
                "error": "No parent found (already at root level)",
                "current_position": self.current_code
            }
        
        parent_code = parent_info["code"]
        self.current_code = parent_code
        self.history.append(parent_code)
        logger.info(f"Moved up to: {parent_code}")
        
        return {
            "success": True,
            "parent": parent_info,
            "current_position": self.current_code,
            "navigation_depth": len(self.history)
        }

    @function_tool
    def go_to_child(self, child_code: str) -> Dict[str, Any]:
        """
        Descend vers un enfant spécifique du noeud actuel.
        
        Args:
            child_code: Code de l'enfant vers lequel naviguer
        
        Returns:
            Résultat de la navigation avec validation
        """
        children = self.graph.get_children(self.current_code)
        child_codes = [child["code"] for child in children]
        
        if child_code not in child_codes:
            return {
                "success": False,
                "error": f"{child_code} is not a direct child of {self.current_code}",
                "current_position": self.current_code,
                "available_children": child_codes
            }
        
        target_info = next((c for c in children if c["code"] == child_code), None)
        self.current_code = child_code
        self.history.append(child_code)
        logger.info(f"📥 Moved down to: {child_code}")
        
        return {
            "success": True,
            "node": target_info,
            "current_position": self.current_code,
            "navigation_depth": len(self.history)
        }

    @function_tool
    def reset_to_root(self) -> Dict[str, Any]:
        """
        Réinitialise la navigation à la racine.
        
        Returns:
            Confirmation de la réinitialisation
        """
        root = self.history[0]
        self.current_code = root
        self.history = [root]
        logger.info("🔄 Reset to root")
        
        return {
            "success": True,
            "message": "Navigation reset to root",
            "current_position": self.current_code
        }

    # ------------------------------------------------------------------
    # Search methods (delegation to Graph)
    # ------------------------------------------------------------------

    @function_tool
    def search_codes(self, search_term: str) -> List[Dict[str, Any]]:
        """
        Recherche des codes par mots-clés dans name et description.
        
        Args:
            search_term: Terme à rechercher
        
        Returns:
            Maximum 20 résultats
        """
        return self.graph.search_codes(search_term)

    # ------------------------------------------------------------------
    # Context methods
    # ------------------------------------------------------------------

    @function_tool
    def get_context_summary(self) -> Dict[str, Any]:
        """
        Retourne un résumé complet de la position actuelle dans la hiérarchie.
        
        Returns:
            Résumé avec noeud actuel, parent, enfants, siblings et chemin
        """
        current = self.graph.get_code_information(self.current_code)
        children = self.graph.get_children(self.current_code)
        siblings = self.graph.get_siblings(self.current_code)
        parent = self.graph.get_parent(self.current_code)
        
        description = current.get("description", "")
        truncated_desc = description[:200] + "..." if len(description) > 200 else description
        
        return {
            "current_node": {
                "code": current.get("code"),
                "name": current.get("name"),
                "level": current.get("level"),
                "description": truncated_desc
            },
            "parent_code": parent.get("code") if parent else None,
            "children_count": len(children),
            "siblings_count": len(siblings),
            "navigation_path": " → ".join(self.history[-5:]),
            "can_go_deeper": len(children) > 0
        }

    @function_tool
    def get_navigation_history(self) -> Dict[str, Any]:
        """
        Retourne l'historique complet de navigation.
        
        Returns:
            Historique et position courante
        """
        return {
            "current_position": self.current_code,
            "full_history": self.history,
            "recent_history": self.history[-10:],
            "navigation_depth": len(self.history)
        }

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def clear_caches(self) -> None:
        """Vide tous les caches (à appeler lors du rechargement des données)"""
        self.graph.clear_caches()
    
    # ------------------------------------------------------------------
    # Get tools
    # ------------------------------------------------------------------

    def get_tools(self) -> List[BaseTool]:
        """
        Retourne la liste de tous les tools disponibles pour l'agent.
        
        Returns:
            Liste des tools de navigation et d'information
        """
        return [
            # Information methods
            self.get_current_information,
            self.get_code_information,
            self.get_current_children,
            self.get_current_siblings,
            self.get_descendants,
            
            # Navigation methods
            self.navigate_to,
            self.go_to_parent,
            self.go_to_child,
            self.reset_to_root,
            
            # Search methods
            self.search_codes,
            
            # Context methods
            self.get_context_summary,
            self.get_navigation_history,
        ]