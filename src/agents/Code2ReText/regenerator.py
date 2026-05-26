import asyncio
from typing import List, Type, Literal
from typing_extensions import TypedDict
from pydantic import BaseModel, Field, create_model

import httpx
from qdrant_client import QdrantClient
from openai import OpenAI

# Imports LangChain & LangGraph
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from langfuse.langchain import CallbackHandler


# Définition de l'état local au module
class AgentState(TypedDict):
    messages: list
    contexte_generation: dict
    code: str
    code_name: str


class ReGenerator:
    """Service utilitaire de génération de données synthétiques basé sur LangGraph."""

    def __init__(
        self,
        gen_client: ChatOpenAI,
        qdrant_client: QdrantClient,
        qdrant_collection: str,
        embed_client: OpenAI,
        embed_model: str,
        discrim_url: str,
        nb_labels: int
    ):
        self.langfuse_handler = CallbackHandler()
        self.gen_client = gen_client
        self.qdrant_client = qdrant_client
        self.qdrant_collection = qdrant_collection
        self.embed_client = embed_client
        self.embed_model = embed_model
        self.discrim_url = discrim_url

        self.discrimination_threshold = 0.7

        self.LabelGeneration = create_model(
            "LabelGeneration",
            labels=(List[str], Field(..., min_items=nb_labels, max_items=nb_labels))
        )

        # Initialisation interne des composants du graphe
        self.tools = self._init_tools()
        self.tool_node = ToolNode(self.tools)
        # On lie les outils au LLM injecté
        self.gen_with_tools = self.gen_client.bind_tools(self.tools)
        self.graph = self._compile_graph()

    def _init_tools(self) -> list:
        """Définit les outils en accédant aux clients de l'instance."""

        @tool
        async def discriminate_labels(texts: List[str]) -> List[str]:
            """Évalue si les libellés ressemblent à de l'IA."""
            # Sécurité au cas où le LLM envoie une chaîne imbriquée dans une liste de listes
            if texts and isinstance(texts[0], list):
                texts = [item for sublist in texts for item in sublist]

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.discrim_url,
                    json={"texts": texts},
                    timeout=10.0
                )

                # Vérification du statut de la réponse avant de parser
                if response.status_code != 200:
                    return [f"Erreur API ({response.status_code}): Impossible d'analyser ces textes."]

                res_json = response.json()

                # Sécurité : Si l'API a renvoyé un dictionnaire d'erreur plutôt qu'une liste de probabilités
                if isinstance(res_json, dict):
                    return [f"Erreur format API: {res_json.get('detail', 'Réponse invalide')}" ]

                try:
                    # On force la conversion en float pour éviter le conflit str/float
                    probs = [float(p) for p in res_json]
                except (ValueError, TypeError):
                    return ["Erreur: L'API de discrimination n'a pas renvoyé des scores numériques valides."]

                # Construction des labels de validité
                validity = ["détecté comme IA → à regénérer" if x >= self.discrimination_threshold else "détecté comme humain → à conserver" for x in probs]

                return [text + ": " + valid for text, valid in zip(texts, validity)]

        @tool
        async def get_human_examples(query: str, limit: int = 5) -> List[str]:
            """Récupère des exemples réels de libellés humains depuis la base de données pour t'informer 
            sur le style d'écriture à reproduire. À utiliser si tes propositions sont rejetées."""
            query_vector = self.embed_client.embeddings.create(
                model=self.embed_model, input=query
            ).data[0].embedding

            points = self.qdrant_client.query_points(
                collection_name=self.qdrant_collection,
                query=query_vector,
                with_payload=True,
                limit=limit
            ).points
            return [f"Exemple humain : {p.payload['label']}" for p in points]

        return [discriminate_labels, get_human_examples]

    # --- NŒUDS DE COMPORTEMENT DU GRAPHE ---
    async def _call_model(self, state: AgentState):
        """Nœud utilisant le LLM configuré avec ses outils."""
        messages = state["messages"]
        response = await self.gen_with_tools.ainvoke(messages)
        return {"messages": [response]}

    async def _post_tool_guidance(self, state: AgentState) -> AgentState:
        """Injecte un rappel de mission après chaque retour d'outil."""
        last_messages = state["messages"]

        # Vérifie si le dernier message est un tool result de discriminate
        last = last_messages[-1]
        is_discrimination_result = (
            hasattr(last, "name") and last.name == "discriminate_labels"
        )

        if is_discrimination_result:
            relance = (
                f"Certains libellés sont détectés comme IA. "
                f"Rappel de ta mission : tu génères des libellés pour le code "
                f"{state['code']} ({state['code_name']}). "
                f"Appelle get_human_examples pour t'inspirer du style humain, "
                f"puis régénère UNIQUEMENT les libellés invalides en imitant ce style."
            )
            return {
                "messages": last_messages + [{"role": "user", "content": relance}]
            }

        return state

    async def _format_output(self, state: AgentState):
        """Nœud forçant la structuration Pydantic finale."""
        llm_structured = self.gen_client.with_structured_output(self.LabelGeneration)
        res = await llm_structured.ainvoke(state["messages"])
        return {"messages": [res]}

    @staticmethod
    def _should_continue(state: AgentState) -> Literal["tools", "final_format"]:
        """Routeur déterminant s'il faut appeler un outil ou clore la boucle."""
        last_message = state["messages"][-1]
        if last_message.tool_calls:
            return "tools"
        return "final_format"

    def _compile_graph(self) -> CompiledStateGraph:
        """Assemble et compile le workflow LangGraph."""
        workflow = StateGraph(AgentState)

        # Association des méthodes de l'instance aux nœuds
        workflow.add_node("agent", self._call_model)
        workflow.add_node("tools", self.tool_node)
        workflow.add_node("post_tool_guidance", self._post_tool_guidance)
        workflow.add_node("final_format", self._format_output)

        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges("agent", self._should_continue)
        workflow.add_edge("tools", "post_tool_guidance")
        workflow.add_edge("post_tool_guidance", "agent")
        workflow.add_edge("final_format", END)

        return workflow.compile()

    # --- LOGIQUE D'EXÉCUTION PUBLIQUE ---
    @staticmethod
    def build_label_generation_model(nb_labels: int) -> Type[BaseModel]:
        return create_model(
            "LabelGeneration",
            labels=(List[str], Field(..., min_items=nb_labels, max_items=nb_labels))
        )

    async def run_single_agent(
        self,
        system_prompt: str,
        user_prompt: str,
        code: str,
        code_name: str
    ):
        inputs = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "contexte_generation": {"model_pydantic": self.LabelGeneration},
            "code": code,
            "code_name": code_name
        }

        final_state = await self.graph.ainvoke(
            inputs,
            config={
                "callbacks": [self.langfuse_handler],
                "recursion_limit": 20
            }
        )

        return final_state["messages"][-1]

    async def run_multiple_agents(
        self,
        system_prompt: str,
        user_prompts: list,
        codes: list,
        code_names: list,
        max_concurrency: int = 15
    ):
        semaphore = asyncio.Semaphore(max_concurrency)

        async def safe_run(prompt, code, code_name):
            async with semaphore:
                return await self.run_single_agent(system_prompt, prompt, code, code_name)

        tasks = [safe_run(p, c, cn) for p, c, cn in zip(user_prompts, codes, code_names)]
        return await asyncio.gather(*tasks)
