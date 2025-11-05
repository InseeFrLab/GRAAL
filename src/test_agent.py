import os

from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate
from langchain.tools import tool
from langchain_openai import ChatOpenAI

from build_graph_db import setup_graph
from hierarchical_navigator import HierarchicalNavigator
from llm.client import setup_langfuse

load_dotenv()
setup_langfuse()

graph = setup_graph()
navigator = HierarchicalNavigator(graph)


# --- Tool definitions ---
@tool
def go_up() -> dict:
    """
    Retrieve node Returns JSON list of {code, level, title, notice, score}.
    """
    return navigator.go_up()


@tool
def go_down(node: str) -> dict:
    """
    Retrieve children of a parent code.
    Returns JSON list of {code, level, title, notice, score}.
    """
    return navigator.go_down(node)


@tool
def get_current_node() -> dict:
    """
    Get information about the current node
    """
    return navigator.get_current_node()


@tool
def get_children() -> dict:
    """
    Get information about the children
    """
    return navigator.get_children()


llm = ChatOpenAI(
    model="mistralai/Mistral-Small-24B-Instruct-2501",
    temperature=0,
    openai_api_base=os.environ["URL_LLM_API"],
    openai_api_key=os.environ["OPENAI_API_KEY"],
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Vous êtes un expert en classification NACE.
Utilisez les outils disponibles pour :
- Explorer les principaux candidats au Niveau 1 de la classification NACE.
- Affiner la recherche avec get_children jusqu'à obtenir le code le plus spécifique.
- À chaque étape, justifiez votre choix.
Retournez votre réponse finale sous forme de JSON : {{"best_code": ..., "path": [...], "alternatives": [...]}}""",
        ),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)


tools = [go_up, go_down, get_current_node, get_children]
agent = create_openai_functions_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


# --- Run classification ---
query = "Boulangerie"
result = executor.invoke({"input": f"Classe cette activité : {query}"})
