import json

from agents import Agent, tool
from langchain_neo4j import Neo4jVector
from pydantic import BaseModel

from llm.client import setup_langfuse
from vector_db.utils import retrieve_docs_for_code

setup_langfuse()


class HierarchicalNACEOutput(BaseModel):
    code: str


agent = Agent(
    name="Expert en nomenclature NAF",
    instructions="""Tu es un expert en classification NAF (Nomenclature d'activités française, Insee).
     Etant donné une description d'activité, tu dois identifier le code NAF le plus approprié en utilisant les outils à ta disposition,
     en descendant dans la hiérarchie des codes si nécessaire.
     Tu dois toujours justifier tes choix et expliquer pourquoi tu choisis un code plutôt qu'un autre.
     Si tu n'es pas sûr, tu peux explorer plusieurs options avant de faire un choix final.""",
)


# --- Tool definitions ---
@tool
def search_nodes(db: Neo4jVector, level: int, query: str, k: int = 10) -> str:
    """
    Retrieve top-k NACE nodes at a given level by semantic similarity.
    Returns JSON list of {code, level, title, notice, score}.
    """
    docs = db.asimilarity_search(f"query : {query}", k=k, filter={"LEVEL": level})
    return json.dumps(
        [
            {
                "code": d.metadata["CODE"],
                "level": d.metadata["LEVEL"],
                "title": d.metadata.get("TITLE", ""),
                "notice": d.page_content,
                "score": float(d.score),
            }
            for d in docs
        ],
        ensure_ascii=False,
    )


@tool
def get_children(db: Neo4jVector, parent_code: str, query: str, k: int = 10) -> str:
    """
    Retrieve children of a parent code prioritized for the query.
    Returns JSON list of {code, level, title, notice, score}.
    """
    docs = await retrieve_docs_for_code(parent_code, query, db, k=k)
    return json.dumps(
        [
            {
                "code": d.metadata["CODE"],
                "level": d.metadata["LEVEL"],
                "title": d.metadata.get("TITLE", ""),
                "notice": d.page_content,
                "score": float(d.score),
            }
            for d in docs
        ],
        ensure_ascii=False,
    )


# # --- Agent definition ---
# nace_agent = Agent(
#     client=client,
#     model="gpt-4.1",  # or gpt-4.1-mini if cost/latency matters
#     name="NACEClassifier",
#     instructions="""Vous êtes un expert en classification NACE.
# Utilisez les outils disponibles pour :

# Explorer les principaux candidats au Niveau 1 de la classification NACE.
# Affiner la recherche avec la méthode get_children jusqu'à obtenir le code le plus spécifique.
# À chaque étape, justifiez votre choix.
# Retournez votre réponse finale sous forme de JSON : {"best_code": ..., "path": [...], "alternatives": [...]}
# """,
#     tools=[search_nodes, get_children],
# )

# # --- Run classification ---
# query = "Boulangerie"
# result = nace_agent.run(f"Classe cette activité : {query}")

# print(result.output_text)  # should be the JSON with final code, path, alternatives
