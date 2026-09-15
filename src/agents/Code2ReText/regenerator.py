import asyncio
from typing import List, Literal, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel, Field, create_model

import httpx
from qdrant_client import QdrantClient
from openai import OpenAI

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langfuse.langchain import CallbackHandler


# ==============================================================================
# ÉTAT PARTAGÉ ENTRE LES DEUX AGENTS
# ==============================================================================

class AgentState(TypedDict):
    # Contexte de la tâche (immuable)
    code: str
    code_name: str
    system_prompt_generator: str
    user_prompt: str

    # Messages du générateur
    generator_messages: list

    # Résultats intermédiaires
    current_labels: List[str]           # dernière génération brute
    discrimination_results: List[str]   # résultats bruts de discriminate_labels
    invalid_indices: List[int]          # indices des libellés invalides
    exact_match_indices: List[int]      # indices des libellés qui sont des copies exactes
    human_examples: List[str]           # exemples humains récupérés depuis Qdrant

    # Consignes dynamiques produites par le superviseur
    supervisor_instructions: Optional[str]

    # Contrôle de boucle
    iteration: int
    all_valid: bool


# ==============================================================================
# MODÈLES PYDANTIC
# ==============================================================================

class LabelList(BaseModel):
    labels: List[str]


# ==============================================================================
# REGENERATOR
# ==============================================================================

class ReGenerator:
    """
    Architecture deux agents :
    - Générateur  : génère des libellés selon le system prompt + consignes dynamiques.
    - Superviseur : évalue, recherche des exemples humains, produit de nouvelles consignes.
    La boucle s'arrête quand tous les libellés sont valides ou après max_iterations tours.
    """

    def __init__(
        self,
        gen_client: ChatOpenAI,
        qdrant_client: QdrantClient,
        qdrant_collection: str,
        embed_client: OpenAI,
        embed_model: str,
        discrim_url: str,
        nb_labels: int,
        gen_prompt: str,
        sup_prompt: str,
        max_iterations: int = 5,
        discrimination_threshold: float = 0.7,
    ):
        self.langfuse_handler = CallbackHandler()
        self.gen_client = gen_client
        self.qdrant_client = qdrant_client
        self.qdrant_collection = qdrant_collection
        self.embed_client = embed_client
        self.embed_model = embed_model
        self.discrim_url = discrim_url
        self.nb_labels = nb_labels
        self.gen_prompt = gen_prompt
        self.sup_prompt = sup_prompt
        self.max_iterations = max_iterations
        self.discrimination_threshold = discrimination_threshold

        self.LabelGeneration = create_model(
            "LabelGeneration",
            labels=(List[str], Field(..., min_items=nb_labels, max_items=nb_labels))
        )

        self.graph = self._compile_graph()

    # ==========================================================================
    # NŒUDS DU GÉNÉRATEUR
    # ==========================================================================

    async def _generate(self, state: AgentState) -> dict:
        """
        Génère nb_labels libellés via structured output.
        Si le superviseur a fourni des consignes, elles sont injectées
        comme dernier message utilisateur avant la génération.
        """
        messages = list(state["generator_messages"])

        if state["supervisor_instructions"]:
            messages.append({
                "role": "user",
                "content": state["supervisor_instructions"]
            })

        llm_structured = self.gen_client.with_structured_output(self.LabelGeneration)
        result = await llm_structured.ainvoke(messages)

        # On mémorise les messages enrichis pour les tours suivants
        updated_messages = messages + [
            {"role": "assistant", "content": str(result.labels)}
        ]

        return {
            "generator_messages": updated_messages,
            "current_labels": result.labels,
            "supervisor_instructions": None,   # reset pour le prochain tour
        }

    # ==========================================================================
    # NŒUDS DU SUPERVISEUR
    # ==========================================================================

    async def _discriminate(self, state: AgentState) -> dict:
        """Appelle le discriminateur sur tous les libellés courants."""
        labels = state["current_labels"]
        iteration = state["iteration"]

        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.discrim_url,
                json={"texts": labels},
                timeout=60.0
            )

        if response.status_code != 200:
            # En cas d'erreur API, on considère tout comme valide pour ne pas bloquer
            return {
                "discrimination_results": [],
                "invalid_indices": [],
                "exact_match_indices": [],
                "all_valid": True,
            }

        probs = [float(p) for p in response.json()]
        invalid_indices = [i for i, p in enumerate(probs) if p >= self.discrimination_threshold]
        nb_invalides = len(invalid_indices)

        # --- ENVOI DES SCORES STRUCTURÉS À LANGFUSE ---
        if hasattr(self, "langfuse_handler"):
            # Demande au handler l'ID de la trace parente active pour cette coroutine
            client = self.langfuse_handler.client
            trace_id = client.get_current_trace_id()

            if trace_id:
                # 1. Enregistrement du score spécifique au tour
                client.score_current_trace(
                    name=f"erreurs_tour_{iteration}",
                    value=nb_invalides,
                    data_type="NUMERIC"
                )

                client.score_current_trace(
                    name=f"error_rate_lap_{iteration}",
                    value=nb_invalides/len(labels),
                    data_type="NUMERIC"
                )

        discrimination_results = [
            f"{labels[i]} [score IA: {probs[i]:.3f}]" for i in range(len(labels))
        ]

        return {
            "discrimination_results": discrimination_results,
            "invalid_indices": invalid_indices,
            "exact_match_indices": [],          # sera rempli par _find_exact_matches
            "all_valid": len(invalid_indices) == 0,
        }

    async def _find_exact_matches(self, state: AgentState) -> dict:
        """
        Parmi les libellés invalides, cherche ceux qui sont des copies exactes
        de libellés existants dans Qdrant (FPR du discriminateur).
        Effectue une recherche groupée : 3 à 5 exemples par libellé invalide,
        retourne une liste désordonnée de libellés originaux.
        """
        invalid_indices = state["invalid_indices"]
        if not invalid_indices:
            return {"exact_match_indices": [], "human_examples": []}

        labels = state["current_labels"]
        invalid_labels = [labels[i] for i in invalid_indices]

        # Recherche groupée dans Qdrant
        all_human_examples = []
        exact_match_indices = []

        for idx, label in zip(invalid_indices, invalid_labels):
            query_vector = self.embed_client.embeddings.create(
                model=self.embed_model, input=label
            ).data[0].embedding

            points = self.qdrant_client.query_points(
                collection_name=self.qdrant_collection,
                query=query_vector,
                with_payload=True,
                limit=5
            ).points

            retrieved = [p.payload["label"] for p in points]
            all_human_examples.extend(retrieved)

            # Correspondance exacte (normalisation basique)
            if any(label.strip().lower() == r.strip().lower() for r in retrieved):
                exact_match_indices.append(idx)

        # Dédoublonnage tout en préservant le désordre (ordre d'insertion)
        seen = set()
        deduped_examples = []
        for ex in all_human_examples:
            key = ex.strip().lower()
            if key not in seen:
                seen.add(key)
                deduped_examples.append(ex)

        return {
            "exact_match_indices": exact_match_indices,
            "human_examples": deduped_examples,
        }

    async def _build_supervisor_instructions(self, state: AgentState) -> dict:
        """
        Nœud Superviseur (LLM) : Analyse de manière critique les défauts des libellés,
        croise avec les exemples humains et génère des consignes textuelles intelligentes
        sans copier les originaux.
        """
        invalid_indices = state["invalid_indices"]
        exact_match_indices = state["exact_match_indices"]
        labels = state["current_labels"]
        human_examples = state["human_examples"]
        iteration = state["iteration"]

        # Préparation du rapport d'anomalies textuel pour le LLM Superviseur
        evaluation_report = []
        evaluation_report.append(f"--- RAPPORTS DES ANOMALIES (Tour {iteration + 1}) ---")
        
        for i, label in enumerate(labels):
            if i in exact_match_indices:
                evaluation_report.append(f"Libellé : '{label}' -> ERREUR : Copie exacte d'un vrai libellé de la base de données. REJETÉ.")
            elif i in invalid_indices:
                evaluation_report.append(f"Libellé : '{label}' -> ERREUR : Détecté comme 'Style IA' par le discriminateur. REJETÉ.")
            else:
                evaluation_report.append(f"Libellé : '{label}' -> VALIDE (À conserver tel quel).")

        if human_examples:
            evaluation_report.append("\n--- EXEMPLES HUMAINS RÉELS POUR INSPIRATION STYLE ---")
            for ex in human_examples:
                evaluation_report.append(f"- {ex}")

        supervisor_user_content = (
            f"Voici le rapport d'évaluation de la génération précédente :\n"
            f"{chr(10).join(evaluation_report)}\n\n"
            f"Génère les consignes de correction précises pour le Générateur pour le tour suivant.\n"
            f"Rappelle-lui de conserver les libellés VALIDES intacts et de ne renvoyer que la liste finale mise à jour de {self.nb_labels} libellés."
        )

        # Appel au LLM (On utilise le même client, mais sans structured output pour avoir une critique textuelle riche)
        supervisor_response = await self.gen_client.ainvoke([
            {"role": "system", "content": self.sup_prompt},
            {"role": "user", "content": supervisor_user_content}
        ])

        # Extraction des consignes rédigées par le LLM
        instructions = supervisor_response.content

        return {
            "supervisor_instructions": instructions,
            "iteration": state["iteration"] + 1,
        }

    # ==========================================================================
    # ROUTEURS
    # ==========================================================================

    @staticmethod
    def _should_stop(state: AgentState) -> Literal["supervisor_discriminate", "end"]:
        """Après génération : discrimine toujours sauf si max_iterations atteint."""
        if state["iteration"] >= state.get("_max_iterations", 5):
            return "end"
        return "supervisor_discriminate"

    @staticmethod
    def _after_discrimination(state: AgentState) -> Literal["end", "supervisor_find_matches"]:
        """Après discrimination : arrêt si tout est valide, sinon cherche les copies exactes."""
        if state["all_valid"]:
            return "end"
        return "supervisor_find_matches"

    # ==========================================================================
    # COMPILATION DU GRAPHE
    # ==========================================================================

    def _compile_graph(self) -> CompiledStateGraph:
        workflow = StateGraph(AgentState)

        # Nœuds
        workflow.add_node("generate", self._generate)
        workflow.add_node("supervisor_discriminate", self._discriminate)
        workflow.add_node("supervisor_find_matches", self._find_exact_matches)
        workflow.add_node("supervisor_build_instructions", self._build_supervisor_instructions)

        # Flux principal
        workflow.add_edge(START, "generate")

        workflow.add_conditional_edges(
            "generate",
            self._should_stop,
            {
                "supervisor_discriminate": "supervisor_discriminate",
                "end": END,
            }
        )

        workflow.add_conditional_edges(
            "supervisor_discriminate",
            self._after_discrimination,
            {
                "end": END,
                "supervisor_find_matches": "supervisor_find_matches",
            }
        )

        workflow.add_edge("supervisor_find_matches", "supervisor_build_instructions")
        workflow.add_edge("supervisor_build_instructions", "generate")

        return workflow.compile()

    # ==========================================================================
    # API PUBLIQUE
    # ==========================================================================

    async def run_single_agent(
        self,
        user_prompt: str,
        code: str = "",
        code_name: str = "",
        session_id: str = None
    ) -> LabelList:

        initial_state: AgentState = {
            "code": code,
            "code_name": code_name,
            "system_prompt_generator": self.gen_prompt,
            "user_prompt": user_prompt,
            "generator_messages": [
                {"role": "system", "content": self.gen_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "current_labels": [],
            "discrimination_results": [],
            "invalid_indices": [],
            "exact_match_indices": [],
            "human_examples": [],
            "supervisor_instructions": None,
            "iteration": 0,
            "all_valid": False,
            "_max_iterations": self.max_iterations,
        }

        run_config = {
            "callbacks": [self.langfuse_handler],
            "recursion_limit": self.max_iterations * 4 + 10,
        }

        if session_id:
            run_config["metadata"] = {"langfuse_session_id": session_id}

        final_state = await self.graph.ainvoke(
            initial_state,
            config=run_config
        )

        return self.LabelGeneration(labels=final_state["current_labels"])

    async def run_multiple_agents(
        self,
        user_prompts: list,
        codes: Optional[List[str]] = None,
        code_names: Optional[List[str]] = None,
        max_concurrency: int = 15,
        session_id: str = None
    ) -> List[LabelList]:

        if codes is None:
            codes = [""] * len(user_prompts)
        if code_names is None:
            code_names = [""] * len(user_prompts)

        semaphore = asyncio.Semaphore(max_concurrency)

        async def safe_run(prompt, code, code_name):
            async with semaphore:
                return await self.run_single_agent(prompt, code, code_name, session_id)

        tasks = [
            safe_run(p, c, cn)
            for p, c, cn in zip(user_prompts, codes, code_names)
        ]
        return await asyncio.gather(*tasks)