import logging

from langchain_neo4j import Neo4jGraph, Neo4jVector
from neo4j import GraphDatabase

from neo4j_graph.graph_builder.config import NEO4J_PWD, NEO4J_URL, NEO4J_USERNAME

logger = logging.getLogger(__name__)


def create_vector_db(docs, embedding_model, clean_previous: bool = True) -> Neo4jVector:
    logger.info("Creating Neo4j vector DB with embeddings")

    if clean_previous:
        command = "DROP INDEX vector IF EXISTS"
        logger.info("🧹 Cleaning previous vector DB. Running command " + command)
        execute_cypher_command(command)

        command = "MATCH (n) DETACH DELETE n"
        logger.info("🧹 Cleaning previous vector DB. Running command " + command)
        execute_cypher_command(command)

    return Neo4jVector.from_documents(
        docs,
        embedding_model,
        url=NEO4J_URL,
        username=NEO4J_USERNAME,
        password=NEO4J_PWD,
        ids=[f"{i}" for i in range(len(docs))],
    )


def create_root_node():
    logger.info("Creating a root node")
    command = """
        MERGE (root {CODE: 'root', LEVEL: 0})
        WITH root
        MATCH (n WHERE n.LEVEL = 1)
        MERGE (root)-[:HAS_CHILD]->(n)
    """
    execute_cypher_command(command)


def create_parent_child_relationships(graph: Neo4jGraph):
    logger.info("🔁 Creating HAS_CHILD relationships")
    graph.query(
        """
    MATCH (child)
    WHERE child.PARENT_ID IS NOT NULL
    MATCH (parent {ID: child.PARENT_ID})
    MERGE (parent)-[:HAS_CHILD]->(child)
    """
    )
    logger.info("✅ Relationships created")


def setup_graph() -> Neo4jGraph:
    logger.info("🔗 Connecting to Neo4j graph DB")
    return Neo4jGraph(
        url=NEO4J_URL,
        username=NEO4J_USERNAME,
        password=NEO4J_PWD,
        enhanced_schema=True,
    )


def execute_cypher_command(query, parameters=None):
    driver = GraphDatabase.driver(NEO4J_URL, auth=(NEO4J_USERNAME, NEO4J_PWD))
    try:
        with driver.session() as session:
            result = session.run(query, parameters or {})
            return result.data()  # Returns list of records
    except Exception as e:
        logging.error(f"Error executing Cypher query: {e}")
        raise
    finally:
        driver.close()
