import logging
from typing import Any, Dict, List, Optional

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
        logger.info("Navigator: get_current_information called")
        data = navigator._cached_get_code_information(navigator.current_code)
        if not data:
            logger.info("No data to sent")
            return {"error": f"Code {navigator.current_code} not found"}
        logger.info(f"Data sent to the llm: {data}")
        return _unfreeze_dict(data)

    @function_tool
    def get_code_information(code: str) -> Dict[str, Any]:
        """
        Retourne les informations d'un code spécifique sans changer la position.

        Args:
            code: Code NACE à consulter

        Returns:
            Informations complètes du code
        """
        logger.info("Navigator: get_code_information called")

        data = navigator._cached_get_code_information(code)

        if not data:
            return {"error": f"Code {code} not found"}

        info = _unfreeze_dict(data)

        logger.info(f"Information available: {data}")

        filtered_information = {
            "code": info.get("code"),
            "name": info.get("name"),
            "level": info.get("level"),
            "description": info.get("description", "")[:500],  # Limiter la taille
        }

        logger.info(f"Filtered information: {filtered_information}")

        return filtered_information

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
        keys_to_keep = ["code", "name"]
        print(f"Keys_to_keep: {keys_to_keep}")
        filtered_children_found = [
            {k:d[k] for k in keys_to_keep}
            for d in children_found
        ]
        logger.info(f"Navigator children found: {filtered_children_found}")
        return filtered_children_found

    @function_tool
    def get_current_siblings() -> List[Dict[str, Any]]:
        """
        Retourne les codes au même niveau que le noeud actuel.

        Returns:
            Liste des siblings du noeud courant
        """
        logger.info("Navigator: get_current_siblings called")
        return _unfreeze_list_of_dicts(navigator._cached_get_siblings(navigator.current_code))

    @function_tool
    def get_current_descendants(levels: int = 2) -> List[Dict[str, Any]]:
        """
        Retourne les descendants du noeud actuel jusqu'à N niveaux.

        Args:
            levels: Nombre de niveaux à descendre (défaut: 2)

        Returns:
            Liste de tous les descendants
        """
        logger.info("Navigator: get_current_descendants called")
        return _unfreeze_list_of_dicts(
            navigator._cached_get_descendants(navigator.current_code, levels)
        )

    @function_tool
    def get_current_parent() -> Optional[Dict[str, Any]]:
        """
        Retourne le parent direct du noeud actuel.

        Returns:
            Dictionnaire du parent ou None si pas de parent
        """
        logger.info("Navigator: get_current_parent called")
        data = navigator._cached_get_parent(navigator.current_code)
        return _unfreeze_dict(data) if data else None

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
            return {
                "success": False,
                "error": f"Code {code} not found",
                "current_position": navigator.current_code,
            }

        navigator.current_code = code
        navigator.history.append(code)
        logger.info(f"Navigated to: {code}")

        return {
            "success": True,
            "node": info,
            "current_position": navigator.current_code,
            "navigation_depth": len(navigator.history),
        }

    @function_tool
    def go_to_parent() -> Dict[str, Any]:
        """
        Remonte au parent du noeud actuel.

        Returns:
            Résultat de la navigation avec informations du parent
        """
        logger.info("Navigator: go_to_parent called")

        parent_info = get_current_parent()

        if parent_info is None:
            return {
                "success": False,
                "error": "No parent found (already at root level)",
                "current_position": navigator.current_code,
            }

        parent_code = parent_info["code"]
        navigator.current_code = parent_code
        navigator.history.append(parent_code)
        logger.info(f"Moved up to: {parent_code}")

        return {
            "success": True,
            "parent": parent_info,
            "current_position": navigator.current_code,
            "navigation_depth": len(navigator.history),
        }

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
        navigator.current_code = child_code
        navigator.history.append(child_code)
        logger.info(f"Moved down to: {child_code}")
        logger.info(f"Navigator.current_code is {navigator.current_code} and navigator.history is {navigator.history}")

        return {
            "success": True,
            "node": target_info,
            "current_position": navigator.current_code,
            "navigation_depth": len(navigator.history),
        }

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

    # ------------------------------------------------------------------
    # Context methods
    # ------------------------------------------------------------------

    @function_tool
    def get_context_summary() -> Dict[str, Any]:
        """
        Retourne un résumé complet de la position actuelle dans la hiérarchie.

        Returns:
            Résumé avec noeud actuel, parent, enfants, siblings et chemin
        """
        logger.info("Navigator: get_context_summary called")

        current = get_current_information()
        if "error" in current:
            return current
        children = get_current_children()
        siblings = get_current_siblings()
        parent = get_current_parent()

        description = current.get("description", "")
        truncated_desc = description[:200] + "..." if len(description) > 200 else description

        result = {
            "current_node": {
                "code": current.get("code"),
                "name": current.get("name"),
                "level": current.get("level"),
                "description": truncated_desc,
            },
            "parent_code": parent.get("code") if parent else None,
            "children_count": len(children),
            "siblings_count": len(siblings),
            "navigation_path": " → ".join(navigator.history[-5:]),
            "can_go_deeper": len(children) > 0,
        }
        logger.info("Navigator result for get_context_summary: {result}")
        return result

    @function_tool
    def get_navigation_history() -> Dict[str, Any]:
        """
        Retourne l'historique complet de navigation.

        Returns:
            Historique et position courante
        """
        logger.info("Navigator: get_navigation_history called")

        return {
            "current_position": navigator.current_code,
            "full_history": navigator.history,
            "recent_history": navigator.history[-10:],
            "navigation_depth": len(navigator.history),
        }

    @function_tool
    def submit_classification(
        query: str,
        confidence: str,
        reasoning: str, 
        # TODO: rajouter "alternatives: str"
    ) -> MatchVerificationInput:
        """
        Retourne le résultat final de la classification. 
        
        Args:
         - query: Libellé à classer
         - confidence: Score de confiance (compris entre 0 et 1)
         - reasoning: Justification détaillée du choix de code
         
        Returns:
            Dictionnaire comprenant la position finale, l'historique de navigation,
            le niveau de confiance, la justification et les alternatives.
        """

        result = {
            "activity": query,
            "code": navigator.current_code,
            "proposed_explanation": reasoning,
            "proposed_confidence": float(confidence),
        }

        return result

    logger.info("Navigator tools created")

    return [
        get_current_information,
        get_code_information,
        get_current_parent,
        get_current_children,
        get_current_siblings,
        go_to_parent,
        go_to_child,
        get_context_summary,
        submit_classification
    ]

    """ return [
        get_current_information,
        get_code_information,
        get_current_parent,
        get_current_children,
        get_current_siblings,
        get_current_descendants,
        navigate_to,
        go_to_parent,
        go_to_child,
        reset_to_root,
        get_context_summary,
        get_navigation_history,
        submit_classification  
    ] """


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
