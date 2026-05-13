import logging
import json
import asyncio
import os

from dotenv import load_dotenv
import requests
import httpx
from openai import OpenAI, AsyncOpenAI
from pydantic import BaseModel, Field, create_model
from typing import List, Type, Optional
from qdrant_client import QdrantClient

load_dotenv(override=True)

DISCRIM_URL = "https://labelguard.lab.sspcloud.fr"
DISCRIMINATION_THRESHOLD = 0.7
LLM_API_URL = os.environ["LLM_URL"]
LLM_API_KEY = os.environ["LLM_API_KEY"]
QDRANT_CLIENT = QdrantClient(
    url="http://qdrant:6333",
    api_key=os.environ["QDRANT_API_KEY"],
    timeout=120
)
EMBED_CLIENT = OpenAI(base_url=LLM_API_URL, api_key=LLM_API_KEY)
EMBED_MODEL = "qwen3-embedding-8b"


def discriminate(texts):
    probs = requests.post(DISCRIM_URL + "/predict_proba", json={"texts": texts})
    return [prob >= DISCRIMINATION_THRESHOLD for prob in probs]


async def discriminate_async(texts: list[str]) -> list[bool]:
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{DISCRIM_URL}/predict_proba",
            json={"texts": texts},
            timeout=10.0
        )
        probs = response.json()
        return [prob >= DISCRIMINATION_THRESHOLD for prob in probs]


logger = logging.getLogger(__name__)


async def search_human_labels(query: str, limit: int = 5) -> List[str]:
    """Recherche des libellés réels dans la base Qdrant pour servir d'inspiration."""
    query_vector = EMBED_CLIENT.embeddings.create(
                model=EMBED_MODEL,
                input=query
            ).data[0].embedding

    points = QDRANT_CLIENT.query_points(
        collection_name="original_cleaned",
        query=query_vector,
        with_payload=True,
        with_vectors=False,
        limit=limit
    ).points

    examples = [point.payload["label"] for point in points]

    # Simulation de retour
    return [f"Exemple humain {i+1} pour {query} : {example}" for i, example in enumerate(examples)]


def build_label_generation_model(nb_labels: int) -> Type[BaseModel]:
    """
    Dynamically build a Pydantic model enforcing exactly n_labels.

    Args:
        nb_labels (int): The number of labels.

    Returns:
        Type[BaseModel]: The expected format for output.
    """

    return create_model(
        "LabelGeneration",
        labels=(
            List[str],
            Field(
                ...,
                min_items=nb_labels,
                max_items=nb_labels,
                description=f"Exactly {nb_labels} generated labels"
            )
        )
    )


# Modèle pour les arguments de l'outil
class DiscriminateArgs(BaseModel):
    texts: List[str] = Field(..., description="Liste des libellés à évaluer pour détecter un style IA.")


class QdrantSearchArgs(BaseModel):
    query: str = Field(..., description="Le concept ou terme pour lequel chercher des exemples de libellés humains réels.")
    limit: int = Field(default=5, description="Nombre d'exemples à récupérer.")


async def ask_model_agentic_single(
    system_prompt: str,
    user_prompt: str,
    llm_client: AsyncOpenAI,
    model: str,
    temperature: float,
    LabelGeneration: Type[BaseModel],
    max_iterations: int = 8
) -> Optional[BaseModel]:

    tools = [
        {
            "type": "function",
            "function": {
                "name": "discriminate_labels",
                "description": "Évalue si les libellés ressemblent à de l'IA. Retourne une liste de booléens (True=IA/Rejeté, False=Humain/Valide).",
                "parameters": DiscriminateArgs.model_json_schema()
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_human_examples",
                "description": "Récupère des exemples réels de libellés humains depuis la base de données pour t'informer sur le style d'écriture à reproduire. À utiliser si tes propositions sont rejetées.",
                "parameters": QdrantSearchArgs.model_json_schema()
            }
        }
    ]

    messages = [
        {
            "role": "system",
            "content": (
                f"{system_prompt}\n"
                "CONSIGNES AGENTIQUES :\n"
                "1. Génère tes libellés.\n"
                "2. Utilise 'discriminate_labels' pour vérifier ton travail.\n"
                "3. Si tes libellés sont marqués comme IA (True), utilise 'get_human_examples' pour comprendre le style attendu.\n"
                "4. Ne copie jamais mot pour mot les exemples de la base de données, inspire-toi de leur structure et de leur ton."
            )
        },
        {"role": "user", "content": user_prompt}
    ]

    for _ in range(max_iterations):
        response = await llm_client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            temperature=temperature,
            tool_choice="auto"
        )

        response_message = response.choices[0].message
        messages.append(response_message)

        if not response_message.tool_calls:
            # Si le modèle décide de finir, on valide le format final
            try:
                final_check = await llm_client.chat.completions.parse(
                    model=model,
                    messages=messages,
                    response_format=LabelGeneration,
                    temperature=temperature
                )
                return final_check.choices[0].message.parsed
            except Exception as e:
                logger.error(f"Erreur de parsing final : {e}")
                return None

        # Gestion des appels d'outils
        for tool_call in response_message.tool_calls:
            function_name = tool_call.function.name
            func_args = json.loads(tool_call.function.arguments)

            if function_name == "discriminate_labels":
                content = await discriminate_async(func_args.get("texts", []))
            elif function_name == "get_human_examples":
                content = await search_human_labels(
                    query=func_args.get("query"), 
                    limit=func_args.get("limit", 5)
                )
            else:
                content = "Erreur: Outil inconnu."

            messages.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": function_name,
                "content": json.dumps(content)
            })

    return None


async def ask_model_multiple_agentic(
    system_prompt: str,
    user_prompts: list,
    llm_client: AsyncOpenAI,
    model: str,
    temperature: float,
    LabelGeneration: Type[BaseModel],
    max_concurrency: int = 15,
    max_iterations: int = 5
) -> list:
    """
    Exécute plusieurs agents en parallèle avec une limite de concurrence.
    """
    semaphore = asyncio.Semaphore(max_concurrency)

    async def limited_call(prompt):
        async with semaphore:
            return await ask_model_agentic_single(
                system_prompt=system_prompt,
                user_prompt=prompt,
                llm_client=llm_client,
                model=model,
                temperature=temperature,
                LabelGeneration=LabelGeneration,
                max_iterations=max_iterations
            )

    tasks = [limited_call(p) for p in user_prompts]
    return await asyncio.gather(*tasks)
