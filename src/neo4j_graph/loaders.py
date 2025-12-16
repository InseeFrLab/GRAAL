import logging
import os

from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph, Neo4jVector
from langchain_openai import OpenAIEmbeddings
from neo4j import GraphDatabase

NEO4J_URL = "neo4j://neo4j-959343.projet-ape:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PWD = os.environ["NEO4J_API_KEY"]

load_dotenv()

URL_EMBEDDING_API = os.environ.get("URL_EMBEDDING_API", None)
if URL_EMBEDDING_API is None:
    raise ValueError("URL_EMBEDDING_API environment variable must be set.")

EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", None)
if EMBEDDING_MODEL is None:
    raise ValueError("EMBEDDING_MODEL environment variable must be set.")

logger = logging.getLogger(__name__)


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


def create_vector_db(docs, embedding_model, clean_previous: bool = True) -> Neo4jVector:
    logger.info("🧠 Creating Neo4j vector DB with embeddings")

    if clean_previous:
        command = "DROP INDEX vector IF EXISTS"
        logger.info("🧹 Cleaning previous vector DB. Running command " + command)
        execute_cypher_command(command)

        command = "MATCH (n) DETACH DELETE n"
        logger.info("🧹 Cleaning previous vector DB. Running command " + command)
        execute_cypher_command(command)

    return Neo4jVector.from_documents(
        docs, embedding_model, url=NEO4J_URL, username=NEO4J_USERNAME, password=NEO4J_PWD, ids=[f"{i}" for i in range(len(docs))]
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


def setup_graph() -> Neo4jGraph:
    logger.info("🔗 Connecting to Neo4j graph DB")
    return Neo4jGraph(
        url=NEO4J_URL,
        username=NEO4J_USERNAME,
        password=NEO4J_PWD,
        enhanced_schema=True,
    )


def get_embedding_model(model_name: str) -> OpenAIEmbeddings:
    """Initialize the embedding model."""
    return OpenAIEmbeddings(
        model=model_name,
        openai_api_base=URL_EMBEDDING_API,
        openai_api_key="EMPTY",
        tiktoken_enabled=False,
    )


async def get_vector_db() -> Neo4jVector:
    """Initialize the Neo4jVector Store from existing graph."""
    emb_model = get_embedding_model(EMBEDDING_MODEL)
    graph = setup_graph()
    return Neo4jVector.from_existing_graph(
        graph=graph,
        embedding=emb_model,
        index_name="id",
        node_label="Chunk",
        text_node_properties=["text"],
        keyword_index_name="text",
        embedding_node_property="embedding",
        search_type="vector",
    )