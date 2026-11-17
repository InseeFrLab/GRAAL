from dotenv import load_dotenv

from src.neo4j_graph.graph import Neo4JConfig
from src.nomenclature import nomenclature_env

load_dotenv(override=True)

NEO4J_URL = nomenclature_env("NEO4J_URL")
NEO4J_USERNAME = nomenclature_env("NEO4J_USERNAME")
NEO4J_PWD = nomenclature_env("NEO4J_PWD")

neo4j_config = Neo4JConfig(url=NEO4J_URL, username=NEO4J_USERNAME, password=NEO4J_PWD)
