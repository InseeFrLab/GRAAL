import logging
from typing import Any, Dict, List, Optional
import json 

from agents import function_tool
from src.neo4j_graph.graph import Graph, Neo4JConfig, _unfreeze_dict, _unfreeze_list_of_dicts
from src.agents.closers.match_verifier import MatchVerificationInput

logger = logging.getLogger(__name__)


def make_tools(navigator):
    # ------------------------------------------------------------------
    # Information methods
    # ------------------------------------------------------------------

    @function_tool
    def get_current_information() -> Dict[str, Any]:
        """
        Retourne les informations du noeud actuel.
        Fournis les codes et les noms des noeuds enfants.  
        Pour avoir l'information détaillée sur un enfant, utilise get_code_information(code)
        
        Returns:
            Informations complètes du noeud courant avec historique de navigation
        """
        logger.info(f"Navigator: get_current_information called at the position: {navigator.current_code}")
        
        # Cas spécial pour root
        if navigator.current_code == "root":
            children = get_current_children()
            result = {
                "success": True,
                "code": "root",
                "level": 0,
                "is_final": False,
                "description": "Nœud racine de la nomenclature NACE. Utilisez get_current_children() pour voir les sections disponibles.",
                "current_position": "root",
            }
            logger.info(f"get_current_information (root): Data sent to the llm: {result}")
            return result
        
        # Cas normal (pas root)
        data = _unfreeze_dict(navigator._cached_get_code_information(navigator.current_code))
        if not data:
            logger.warning(f"No data for code: {navigator.current_code}")
            return {
                "success": False,
                "error": f"Code {navigator.current_code} not found",
                "current_position": navigator.current_code
            }
        
        result = {
            "success": True, 
            **data, 
            "current_position": navigator.current_code,
        }
        logger.info(f"get_current_information: Data sent to the llm: {result}")
        return result

    @function_tool
    def get_code_information(code: str) -> Dict[str, Any]:
        """
        Retourne les informations d'un code spécifique sans changer la position.

        Args:
            code: Code NACE à consulter

        Returns:
            Informations complètes du code
        """
        logger.info(f"Navigator: get_code_information called with code {code}")

        data = navigator._cached_get_code_information(code)

        if not data:
            logger.info(f"No data for the code: {code}")
            return {
                "success": False,
                "error": f"Code {code} not found",
                "current_position": navigator.current_code,
            }

        data = _unfreeze_dict(data)

        filtered_data = {
            "success": True,
            "code": data.get("code"),
            "name": data.get("name"),
            "level": data.get("level"),
            "is_final": data.get("is_final"),
            "description": data.get("description", ""),
            "current_position": navigator.current_code,
        }

        logger.info(f"get_code_information with code {code}: Data sent to the llm: {filtered_data}")
        return filtered_data
    

    @function_tool
    def get_current_children() -> List[Dict[str, Any]]:
        """
        Retourne les codes et les noms des enfants directs du noeud actuel.
        Pour avoir l'information détaillée sur un enfant, utilise get_code_information(code)

        Returns:
            Liste des codes enfants du noeud courant, contient le code et son nom
        """
        logger.info(f"Navigator: get_current_children called at the position {navigator.current_code}")
        children_found = _unfreeze_list_of_dicts(navigator._cached_get_children(navigator.current_code))
        keys_to_keep = ["code", "name", "is_final"]
        filtered_children_found = [
            {k: d[k] for k in keys_to_keep}
            for d in children_found
        ]
        logger.info(f"get_current_children: Data sent to the llm {filtered_children_found}")
        return filtered_children_found

    @function_tool
    def get_current_siblings() -> List[Dict[str, Any]]:
        """
        Retourne les codes au même niveau que le noeud actuel.

        Returns:
            Liste des siblings du noeud courant
        """
        logger.info("Navigator: get_current_siblings called")
        result = _unfreeze_list_of_dicts(navigator._cached_get_siblings(navigator.current_code))
        logger.info(f"get_current_siblings: Data sent to the llm: {result}")
        return result

    @function_tool
    def get_current_parent() -> Optional[Dict[str, Any]]:
        """
        Retourne le parent direct du noeud actuel.

        Returns:
            Dictionnaire du parent ou None si pas de parent
        """
        logger.info("Navigator: get_current_parent called")
        data = navigator._cached_get_parent(navigator.current_code)
        result = _unfreeze_dict(data) if data else None
        logger.info(f"get_current_parent: Data sent to the llm: {result}")
        return result

    # ------------------------------------------------------------------
    # Navigation methods
    # ------------------------------------------------------------------

    @function_tool
    def navigate_to(code: str) -> Dict[str, Any]:
        """
        Se déplace vers un code spécifique.

        Args:
            code: Code NACE de destination

        Returns:
            Résultat de la navigation avec informations du nouveau noeud
        """
        logger.info(f"Navigator: navigate_to called with node: {code}")
        info = get_code_information(code)

        if "error" in info:
            logger.info("Error in navigate_to")
            return {
                "success": False,
                "error": f"Code {code} not found",
                "current_position": navigator.current_code,
            }

        navigator.current_code = code
        navigator.history.append(code)
        logger.info(f"Navigated to: {code}")
        
        result = {
            "success": True,
            "node": info,
            "current_position": navigator.current_code,
        }
        logger.info(f"navigate_to: Data sent to the llm: {result}")
        return result

    @function_tool
    def go_to_parent() -> Dict[str, Any]:
        """
        Remonte au parent du noeud actuel.

        Returns:
            Résultat de la navigation avec informations du parent
        """
        logger.info("Navigator: go_to_parent called")
        parent_info = _unfreeze_dict(navigator._cached_get_parent(navigator.current_code))
        if parent_info is None:
            logger.warning("parent_info is None, go_to_parent failed")
            return {
                "success": False,
                "error": "No parent found (already at root level)",
                "current_position": navigator.current_code,
            }

        parent_code = parent_info["code"]
        navigator.current_code = parent_code
        navigator.history.append(parent_code)
        logger.info(f"Move up to: {parent_code}")
        
        parent_info_filtered = {
            'code': parent_info['code'], 
            'level': parent_info['level'], 
            'name': parent_info['name'],
        }

        result = {
            "success": True,
            "parent": parent_info_filtered,
            "current_position": navigator.current_code,
        }

        logger.info(f"go_to_parent: data sent to the llm: {result}")
        return result
    
    @function_tool
    def go_to_child(child_code: str) -> Dict[str, Any]:
        """
        Descend vers un enfant spécifique du noeud actuel.

        Args:
            child_code: Code de l'enfant vers lequel naviguer

        Returns:
            Résultat de la navigation avec validation
        """
        logger.info(f"Navigator: go_to_child called with child_code: {child_code}")

        children = _unfreeze_list_of_dicts(navigator._cached_get_children(navigator.current_code))
        child_codes = [child["code"] for child in children]

        if child_code not in child_codes:
            logger.warning(f"child_code {child_code} is not in child_codes")
            return {
                "success": False,
                "error": f"{child_code} is not a direct child of {navigator.current_code}",
                "current_position": navigator.current_code,
                "available_children": child_codes,
            }

        target_info = next((c for c in children if c["code"] == child_code), None)
        new_node_information = {
            "code": target_info["code"],
            "level": target_info["level"],
            "is_final": target_info['is_final'],
            "name": target_info["name"], 
            }
        navigator.current_code = child_code
        navigator.history.append(child_code)
        logger.info(f"Move down to: {child_code}")
        logger.info(f"Navigator.current_code is {navigator.current_code} and navigator.history is {navigator.history}")

        result = {
            "success": True,
            "node": new_node_information,
            "current_position": navigator.current_code,
        }
        logger.info(f"go_to_child: Data sent to the llm: {result}")
        return result

    @function_tool
    def reset_to_root() -> Dict[str, Any]:
        """
        Réinitialise la navigation à la racine.

        Returns:
            Confirmation de la réinitialisation
        """
        logger.info("Navigator: reset_to_root called")

        root = navigator.history[0]
        navigator.current_code = root
        navigator.history = [root]
        logger.info("Reset to root")

        return {
            "success": True,
            "message": "Navigation reset to root",
            "current_position": navigator.current_code,
        }

    logger.info("Navigator tools created")

    return [
        get_current_information,
        get_code_information,
        get_current_children,
        get_current_siblings,
        go_to_parent,
        go_to_child,
    ]

class Navigator(Graph):
    """
    Classe de navigation dans la hiérarchie NACE avec état persistant.
    Utilise Graph pour les requêtes et maintient la position courante.
    """

    def __init__(self, neo4j_config: Neo4JConfig, root: str = "root"):
        super().__init__(neo4j_config)
        self.current_code = root
        self.history = [root]

    def get_tools(self):
        """
        Retourne les tools de navigation (override de Graph.get_tools).
        """
        tools = make_tools(self)
        tool_names = [tool.name for tool in tools]
        logger.info(f"Tools accessible to the navigator agent: {tool_names}")
        return tools

    def reset_to_root(self):
        self.current_code = "root"
        self.history = ["root"]
        logger.info("Navigator: Reset to root")