import logging
import asyncio

from openai import OpenAI, AsyncOpenAI
from pydantic import BaseModel, Field, create_model
from typing import List, Type

logger = logging.getLogger(__name__)


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


def ask_model(
        system_prompt: str,
        user_prompt: str,
        llm_client: OpenAI,
        model: str,
        temperature: float,
        LabelGeneration: Type[BaseModel]
        ) -> str:
    """
    Dialogue with the model.
    The model and the temperature are to be configured in the config.py file.

    Args:
        system_prompt (str): The system prompt (orders).
        user_prompts (str): The user prompt (adapted to a specific case).
        llm_client (OpenAI): The client to connect to (initialized with your logins)
        model (str): The model to talk to.
        temperatur (float): Temperature for the answer, between 0 and 2.
        LabelGeneration (Type[BaseModel]): The expected format for output.

    Returns:
        str: The answer returned by the model.
    """
    response = llm_client.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=temperature,
        response_format=LabelGeneration
    )

    if response.choices:
        return response.choices[0].message.parsed

    else:
        logger.warn("The LLM did not return an answer.")
        return None


async def ask_model_async(
        system_prompt: str,
        user_prompt: str,
        llm_client: AsyncOpenAI,
        model: str,
        temperature: float,
        LabelGeneration: Type[BaseModel]
        ) -> str:
    """
    Dialogue with the model.
    The model and the temperature are to be configured in the config.py file.

    Args:
        system_prompt (str): The system prompt (orders).
        user_prompts (str): The user prompt (adapted to a specific case).
        llm_client (OpenAI): The client to connect to (initialized with your logins)
        model (str): The model to talk to.
        temperatur (float): Temperature for the answer, between 0 and 2.
        LabelGeneration (Type[BaseModel]): The expected format for output.

    Returns:
        str: The answer returned by the model.
    """
    response = await llm_client.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=temperature,
        response_format=LabelGeneration
    )

    if response.choices:
        return response.choices[0].message.parsed

    else:
        logger.warn("The LLM did not return an answer.")
        return None


async def ask_model_multiple(
        system_prompt: str,
        user_prompts: list,
        llm_client: AsyncOpenAI,
        model: str,
        temperature: float,
        LabelGeneration: Type[BaseModel],
        max_concurrency: int = 15
        ) -> list:

    semaphore = asyncio.Semaphore(max_concurrency)

    async def limited_call(prompt):
        async with semaphore:
            return await ask_model_async(
                system_prompt=system_prompt,
                user_prompt=prompt,
                llm_client=llm_client,
                model=model,
                temperature=temperature,
                LabelGeneration=LabelGeneration
                )

    tasks = [limited_call(p) for p in user_prompts]
    return await asyncio.gather(*tasks)
