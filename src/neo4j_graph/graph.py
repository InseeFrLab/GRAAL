import logging
import os
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph, Neo4jVector
from langchain_openai import OpenAIEmbeddings
from pydantic import BaseModel

from agents import function_tool

logger = logging.getLogger(__name__)
load_dotenv(override=True)


def _freeze_dict(d: Dict[str, Any]) -> Tuple[Tuple[str, Any], ...]:
    """Convert dict to immutable tuple for caching."""
    return tuple(d.items())


def _freeze_list_of_dicts(lst: List[Dict[str, Any]]) -> Tuple[Tuple[Tuple[str, Any], ...], ...]:
    """Convert list[dict] to immutable structure for caching."""
    return tuple(_freeze_dict(d) for d in lst)


def _unfreeze_dict(t: Tuple[Tuple[str, Any], ...]) -> Dict[str, Any]:
    return dict(t)


def _unfreeze_list_of_dicts(t: Tuple[Tuple[Tuple[str, Any], ...], ...]) -> List[Dict[str, Any]]:
    return [dict(d) for d in t]


def make_tools(graph):
    @function_tool
    def get_code_information(code: str) -> Dict[str, Any]:
        """
        Retourne les informations complètes d'un code NACE.

        Args:
            code: Code NACE (ex: "62.01", "J", "62")

        Returns:
            Dictionnaire avec code, level, name, description, includes, includes_also,
            excludes, implementation_rule, parent_code, children_codes, children_count
        """
        data = graph._cached_get_code_information(code)
        return _unfreeze_dict(data) if data else {"error": f"Code {code} not found"}

    @function_tool
    def get_children(code: str) -> List[Dict[str, Any]]:
        """
        Retourne les enfants directs d'un code (niveau N+1).

        Args:
            code: Code parent

        Returns:
            Liste des codes enfants avec code, level, name, description, includes, excludes
        """
        return _unfreeze_list_of_dicts(graph._cached_get_children(code))

    @function_tool
    def get_descendants(code: str, levels: int = 2) -> List[Dict[str, Any]]:
        """
        Retourne les descendants d'un code jusqu'à N niveaux de profondeur.

        Args:
            code: Code de départ
            levels: Nombre de niveaux à descendre (défaut: 2, recommandé: ≤3)

        Returns:
            Liste de tous les descendants jusqu'au niveau spécifié
        """
        return _unfreeze_list_of_dicts(graph._cached_get_descendants(code, levels))

    @function_tool
    def get_siblings(code: str) -> List[Dict[str, Any]]:
        """
        Retourne les codes au même niveau hiérarchique (même parent).

        Args:
            code: Code dont on cherche les siblings

        Returns:
            Liste des codes siblings (excluant le code d'origine)
        """
        return _unfreeze_list_of_dicts(graph._cached_get_siblings(code))

    @function_tool
    def get_parent(code: str) -> Optional[Dict[str, Any]]:
        """
        Retourne le parent direct d'un code (niveau N-1).

        Args:
            code: Code dont on cherche le parent

        Returns:
            Dictionnaire du parent ou None si pas de parent
        """
        data = graph._cached_get_parent(code)
        return _unfreeze_dict(data) if data else None

    return [get_code_information, get_children, get_descendants, get_siblings, get_parent]


class Neo4JConfig(BaseModel):
    url: str
    username: str
    password: str


class Graph:
    def __init__(self, neo4j_config: Neo4JConfig) -> None:
        self.graph = Neo4jGraph(
            url=neo4j_config.url,
            username=neo4j_config.username,
            password=neo4j_config.password,
            enhanced_schema=True,
        )

        self.emb_model = OpenAIEmbeddings(
            model=os.environ["EMBEDDING_MODEL"],
            openai_api_base=os.environ["URL_EMBEDDING_API"],
            openai_api_key=os.environ["OPENAI_API_KEY"],
        )

        self.db = Neo4jVector.from_existing_graph(
            graph=self.graph,
            embedding=self.emb_model,
            index_name="id",
            node_label="Chunk",
            text_node_properties=["text"],
            keyword_index_name="text",
            embedding_node_property="embedding",
            search_type="vector",
        )

    # ------------------------------------------------------------------
    # Get tools
    # ------------------------------------------------------------------

    def get_tools(self):
        """
        Retourne les tools de navigation spécifiques au Navigator.

        Returns:
            Tuple des tools de navigation avec état
        """
        return make_tools(self)

    async def get_closest_codes(self, activity: str, top_k: int = 5) -> List[str]:
        retrieval = await self.db.asimilarity_search(
            f"query : {activity}", k=top_k, filter={"FINAL": 1}
        )
        return [item.metadata["CODE"] for item in retrieval]

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

    @lru_cache(maxsize=0)
    def _cached_get_code_information(self, code: str) -> Tuple[Tuple[str, Any], ...]:
        query = """
        MATCH (node {CODE: $code})
        OPTIONAL MATCH (node)<-[:HAS_CHILD]-(parent)
        OPTIONAL MATCH (node)-[:HAS_CHILD]->(child)
        WITH node, parent, collect(child.CODE), collect(child.NAME) as children_codes
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
        logger.info(f"_cached_get_code_information called with code {code}")
        result = self.graph.query(query, params={"code": code})
        logger.info("_cached_get_code_information result: {result}")
        if not result:
            logger.info("No result in _cached_get_code_information")
            return ()

        return _freeze_dict(result[0])

    # ------------------------------------------------------------------
    # get_children
    # ------------------------------------------------------------------

    @lru_cache(maxsize=0)
    def _cached_get_children(self, code: str) -> Tuple[Tuple[Tuple[str, Any], ...], ...]:
        query = """
        MATCH (node {CODE: $code})-[:HAS_CHILD]->(child)
        RETURN child.CODE as code,
               child.LEVEL as level,
               child.FINAL as final
               child.NAME as name,
               child.text as description,
               child.Includes as includes,
               child.Excludes as excludes
        ORDER BY child.CODE
        """
        result = self.graph.query(query, params={"code": code})
        return _freeze_list_of_dicts(result)

    # ------------------------------------------------------------------
    # get_descendants
    # ------------------------------------------------------------------

    @lru_cache(maxsize=0)
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

    # ------------------------------------------------------------------
    # get_siblings
    # ------------------------------------------------------------------

    @lru_cache(maxsize=0)
    def _cached_get_siblings(self, code: str) -> Tuple[Tuple[Tuple[str, Any], ...], ...]:
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

    # ------------------------------------------------------------------
    # get_parent
    # ------------------------------------------------------------------

    @lru_cache(maxsize=0)
    def _cached_get_parent(self, code: str) -> Tuple[Tuple[str, Any], ...]:
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

    # ------------------------------------------------------------------
    # search_codes
    # ------------------------------------------------------------------

    @lru_cache(maxsize=0)
    def _cached_search_codes(self, search_term: str) -> Tuple[Tuple[Tuple[str, Any], ...], ...]:
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
