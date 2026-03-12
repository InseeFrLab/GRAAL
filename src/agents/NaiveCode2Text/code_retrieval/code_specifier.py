import logging

from langchain_neo4j import Neo4jGraph

logger = logging.getLogger(__name__)


def get_all_codes(
        graph: Neo4jGraph,
        n_samples_per_code: int
        ) -> list:
    """
    Sample codes with replacement using lazyframes from Polars.

    Args:
        fs (S3FileSystem): The filesystem for importation.
        population_path (str): The path of the parquet file of the population.
        code_column (str): The name of the column for codes.
        n_codes (int): The number of codes to sample.

    Returns:
        numpy.ndarray: An array of n_codes codes sampled with replacement.
    """

    query = """
        MATCH (n)
        WHERE n.FINAL = 1
        RETURN DISTINCT n.CODE;
        """

    response = graph.query(query)

    if not response:
        logger.warn("No response in get_all_codes")
        return []

    result = []

    for r in response:
        result += [r["n.CODE"]] * n_samples_per_code

    return result


def get_code_information(
        graph: Neo4jGraph,
        code: str
        ) -> dict:
    """
    Retrieve code specifications from a Neo4j graph

    Args:
        graph (Graph from local library): The Neo4j graph.
        code (str): The code to specify.

    Returns:
        dict: Every accessible information of the code in the graph.
    """

    query = """
    MATCH (node {CODE: $code})
    OPTIONAL MATCH (node)<-[:HAS_CHILD]-(parent)
    OPTIONAL MATCH (node)-[:HAS_CHILD]->(child)
    WITH node, parent, collect({code: child.CODE, name: child.NAME}) as children
    RETURN node.CODE as code,
           node.LEVEL as level,
           node.NAME as name,
           node.text as description,
           node.Includes as includes,
           node.IncludesAlso as includes_also,
           node.Excludes as excludes,
           node.Implementation_rule as implementation_rule,
           parent.CODE as parent_code,
           children,
           size(children) as children_count
    """
    result = graph.query(query, params={"code": code})

    if not result:
        logger.warn("No result in get_code_information")
        return []

    return result[0]


def get_code_list_information(
        graph: Neo4jGraph,
        code_list: list[str]
        ) -> dict:
    """
    Retrieve code specifications from a Neo4j graph for multiple codes.

    Args:
        graph (Neo4jGraph): The Neo4j graph.
        code_list (list[str]): List of codes to retrieve.

    Returns:
        dict: Dictionary of code information keyed by code.
    """

    if not code_list:
        return {}

    query = """
    MATCH (node)
    WHERE node.CODE IN $code_list
    OPTIONAL MATCH (node)<-[:HAS_CHILD]-(parent)
    OPTIONAL MATCH (node)-[:HAS_CHILD]->(child)
    WITH node, parent, collect({code: child.CODE, name: child.NAME}) AS children
    RETURN node.CODE AS code,
           node.LEVEL AS level,
           node.NAME AS name,
           node.text AS description,
           node.Includes AS includes,
           node.IncludesAlso AS includes_also,
           node.Excludes AS excludes,
           node.Implementation_rule AS implementation_rule,
           parent.CODE AS parent_code,
           children,
           size(children) AS children_count
    """

    result = graph.query(query, params={"code_list": code_list})

    if not result:
        logger.warn("No result in get_code_list_information")
        return {}

    # Transformer la liste de résultats en dictionnaire de dictionnaires
    code_dict = {}
    for record in result:
        code_key = record["code"]
        code_dict[code_key] = {
            "level": record["level"],
            "name": record["name"],
            "description": record["description"],
            "includes": record["includes"],
            "includes_also": record["includes_also"],
            "excludes": record["excludes"],
            "implementation_rule": record["implementation_rule"]
        }

    return code_dict
