import logging
from typing import Dict, Any, List, Optional, Tuple
from functools import lru_cache

from langchain_neo4j import Neo4jGraph
from agents import function_tool

logger = logging.getLogger(__name__)


def _freeze_dict(d: Dict[str, Any]) -> Tuple[Tuple[str, Any], ...]:
    """Convert dict to immutable tuple for caching."""
    return tuple(d.items())


def _freeze_list_of_dicts(lst: List[Dict[str, Any]]) -> Tuple[Tuple[Tuple[str, Any], ...], ...]:
    """Convert list[dict] to immutable structure for caching."""
    return tuple(_freeze_dict(d) for d in lst)


def _unfreeze_dict(t: Tuple[Tuple[str, Any], ...]) -> Dict[str, Any]:
    return dict(t)


def _unfreeze_list_of_dicts(
    t: Tuple[Tuple[Tuple[str, Any], ...], ...]
) -> List[Dict[str, Any]]:
    return [dict(d) for d in t]


class Graph:
    def __init__(self):
        self.graph = Neo4jGraph(
            url=NEO4J_URL,
            username=NEO4J_USERNAME,
            password=NEO4J_PWD,
            enhanced_schema=True,
            )


    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def clear_caches(self) -> None:
        """Clear all internal caches (call on data reload)."""
        self._cached_get_code_information.cache_clear()
        self._cached_get_children.cache_clear()
        self._cached_get_descendants.cache_clear()
        self._cached_get_siblings.cache_clear()
        self._cached_get_parent.cache_clear()
        self._cached_search_codes.cache_clear()

    # ------------------------------------------------------------------
    # get_code_information
    # ------------------------------------------------------------------

    @lru_cache(maxsize=10_000)
    def _cached_get_code_information(self, code: str) -> Tuple[Tuple[str, Any], ...]:
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
        if not result:
            return ()
        return _freeze_dict(result[0])

    @function_tool
    def get_code_information(self, code: str) -> Dict[str, Any]:
        """
        Retourne les informations complètes d'un code NACE.
        
        Args:
            code: Code NACE (ex: "62.01", "J", "62")
        
        Returns:
            Dictionnaire avec code, level, name, description, includes, includes_also,
            excludes, implementation_rule, parent_code, children_codes, children_count
        """
        data = self._cached_get_code_information(code)
        return _unfreeze_dict(data) if data else {"error": f"Code {code} not found"}

    # ------------------------------------------------------------------
    # get_children
    # ------------------------------------------------------------------

    @lru_cache(maxsize=10_000)
    def _cached_get_children(
        self, code: str
    ) -> Tuple[Tuple[Tuple[str, Any], ...], ...]:
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
        return _freeze_list_of_dicts(result)

    @function_tool
    def get_children(self, code: str) -> List[Dict[str, Any]]:
        """
        Retourne les enfants directs d'un code (niveau N+1).
        
        Args:
            code: Code parent
        
        Returns:
            Liste des codes enfants avec code, level, name, description, includes, excludes
        """
        return _unfreeze_list_of_dicts(self._cached_get_children(code))

    # ------------------------------------------------------------------
    # get_descendants
    # ------------------------------------------------------------------

    @lru_cache(maxsize=10_000)
    def _cached_get_descendants(
        self, code: str, levels: int
    ) -> Tuple[Tuple[Tuple[str, Any], ...], ...]:
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
        return _freeze_list_of_dicts(result)

    @function_tool
    def get_descendants(self, code: str, levels: int = 2) -> List[Dict[str, Any]]:
        """
        Retourne les descendants d'un code jusqu'à N niveaux de profondeur.
        
        Args:
            code: Code de départ
            levels: Nombre de niveaux à descendre (défaut: 2, recommandé: ≤3)
        
        Returns:
            Liste de tous les descendants jusqu'au niveau spécifié
        """
        return _unfreeze_list_of_dicts(
            self._cached_get_descendants(code, levels)
        )

    # ------------------------------------------------------------------
    # get_siblings
    # ------------------------------------------------------------------

    @lru_cache(maxsize=10_000)
    def _cached_get_siblings(
        self, code: str
    ) -> Tuple[Tuple[Tuple[str, Any], ...], ...]:
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
        return _freeze_list_of_dicts(result)

    @function_tool
    def get_siblings(self, code: str) -> List[Dict[str, Any]]:
        """
        Retourne les codes au même niveau hiérarchique (même parent).
        
        Args:
            code: Code dont on cherche les siblings
        
        Returns:
            Liste des codes siblings (excluant le code d'origine)
        """
        return _unfreeze_list_of_dicts(self._cached_get_siblings(code))

    # ------------------------------------------------------------------
    # get_parent
    # ------------------------------------------------------------------

    @lru_cache(maxsize=10_000)
    def _cached_get_parent(
        self, code: str
    ) -> Tuple[Tuple[str, Any], ...]:
        query = """
        MATCH (node {CODE: $code})<-[:HAS_CHILD]-(parent)
        RETURN parent.CODE as code,
               parent.LEVEL as level,
               parent.NAME as name,
               parent.text as description
        """
        result = self.graph.query(query, params={"code": code})
        if not result:
            return ()
        return _freeze_dict(result[0])

    @function_tool
    def get_parent(self, code: str) -> Optional[Dict[str, Any]]:
        """
        Retourne le parent direct d'un code (niveau N-1).
        
        Args:
            code: Code dont on cherche le parent
        
        Returns:
            Dictionnaire du parent ou None si pas de parent
        """
        data = self._cached_get_parent(code)
        return _unfreeze_dict(data) if data else None

    # ------------------------------------------------------------------
    # search_codes
    # ------------------------------------------------------------------

    @lru_cache(maxsize=5_000)
    def _cached_search_codes(
        self, search_term: str
    ) -> Tuple[Tuple[Tuple[str, Any], ...], ...]:
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
        return _freeze_list_of_dicts(result)

    @function_tool
    def search_codes(self, search_term: str) -> List[Dict[str, Any]]:
        """
        Recherche des codes par mots-clés dans name et description.
        
        Args:
            search_term: Terme à rechercher (insensible à la casse)
        
        Returns:
            Maximum 20 résultats triés par level puis code
        """
        return _unfreeze_list_of_dicts(
            self._cached_search_codes(search_term)
        )

    def get_tools(self):
        """Return the tools associated with the graph."""
        return [
            self.get_code_information,
            self.get_children,
            self.get_descendants,
            self.get_siblings,
            self.get_parent,
            self.search_codes,
        ]
