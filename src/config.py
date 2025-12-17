from dotenv import load_dotenv
import os 

from src.neo4j_graph.Graph import Neo4JConfig

load_dotenv()

NEO4J_URL = os.environ["NEO4J_URL"]  
NEO4J_USERNAME = os.environ["NEO4J_USERNAME"]
NEO4J_PWD = os.environ["NEO4J_PWD"]

neo4j_config = Neo4JConfig(url=NEO4J_URL, username=NEO4J_USERNAME, password=NEO4J_PWD)

