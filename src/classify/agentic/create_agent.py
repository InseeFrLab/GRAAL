import os

from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.prompts import ChatPromptTemplate
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from typing import Dict, Any

from build_graph_db import setup_graph
from hierarchical_navigator import HierarchicalNavigator
from llm.client import setup_langfuse, get_llm_client
from llm.responses import get_llm_choice


class CreateAgent():
    def __init__:
        load_dotenv()
        setup_langfuse()
        self.graph = setup_graph()
        self.navigator = HierarchicalNavigator(graph)
        self.client = get_llm_client()

  

    # --- Tool definitions ---
    @tool
    def go_up() -> Dict[str, Any]:
        """
        Retrieve node Returns JSON list of {code, level, title, notice, score}.
        """
        return navigator.go_up()


    @tool
    def go_down(node: str) -> Dict[str, Any]:
        """
        Retrieve children of a parent code.
        Returns JSON list of {code, level, title, notice, score}.
        """
        return navigator.go_down(node)


    @tool
    def get_current_node() -> Dict[str, Any]:
        """
        Get information about the current node
        """
        return navigator.get_current_node()


    @tool
    def get_children() -> Dict[str, Any]:
        """
        Get information about the children
        """
        return navigator.get_children()




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
agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


# --- Run classification ---
query = "Boulangerie"
result = executor.invoke({"input": f"Classe cette activité : {query}"})
