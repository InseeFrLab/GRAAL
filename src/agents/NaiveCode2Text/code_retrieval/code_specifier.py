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


def NAF_to_NACE(
        code: str
        ) -> str:
    """
    For the case of NAF code format (DDDDL), transform it into NACE (DD.DD).

    Args:
        code (str): The code in NAF format to transform.

    Returns:
        str: The code in NACE format.
    """
    return code[0:2] + '.' + code[2:4]


def to_proper_NAF(
        code: str
        ) -> str:
    """
    For the case of bad NAF code format (DDDDL), transform it into proper NAF (DD.DDL).

    Args:
        code (str): The code in bad NAF format to transform.

    Returns:
        str: The code in proper NAF format.
    """
    return code[0:2] + '.' + code[2:]


def to_bad_NAF(
        code: str
        ) -> str:
    """
    For the case of proper NAF code format (DD.DDL), transform it into bad NAF (DD.DDL).

    Args:
        code (str): The code in proper NAF format to transform.

    Returns:
        str: The code in bad NAF format.
    """
    return code[0:2] + code[3:]
