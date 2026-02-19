import logging
import re
import asyncio

import s3fs
import pandas as pd
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


def retrieve_label(
        label_listing: str,
        delimiter: str = r"\n\s+\d+.\s+",
        nb_labels: int = 10
        ) -> list:
    """
    Split the message which should correspond to a label listing into a list of labels.

    Args:
        label_listing (str): The listing.
        delimiter (str): What should separate each label.
        nb_label (int): The expected number of labels.
            Send a warning if it differs from the splitting length.

    Returns:
        str: The list of labels.
    """
    # Search and split
    label_list = re.split(r"\n\s+\d+.\s+", label_listing)
    label_list[0] = label_list[0].replace("1. ", "")

    # Quick check
    if len(label_list) != nb_labels:
        logger.warn("The answer might not be a wording.")

    return label_list


def export_to_txt(
        codes: list,
        names: list,
        labels: list,
        file_path: str,
        generation_time: float
        ) -> bool:
    """
    Save results to file .txt
    The number of codes used for generation should not exceed 100.
    Else, upload in .parquet format

    Args:
        codes (list): List of codes generated.
        names (list): List of names for the codes generated
        labels (list of lists): List of labels for each code
        file_path (str): The .txt file path.
        generation_time (float): The time it took to generate.

    Returns:
        bool: True if the file has been correctly saved.
    """
    if len(labels) == 0 or len(labels) > 100:
        return False

    nb_labels = len(labels) * len(labels[0])

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(f"{nb_labels} wordings have been generated in {generation_time:.2f} sec.\n\n")
        f.write("=" * 36 + "\n")

        for code, name, generated_labels in zip(codes, names, labels):
            f.write(f"Code: {code}\n")
            f.write(f"Name: {name}\n")
            f.write("Result:\n")

            for j, label in enumerate(generated_labels):
                f.write(f"{j}. {label}\n")

            f.write("\n" + "=" * 36 + "\n")

    return True


def export_to_parquet(
        codes: list,
        names: list,
        labels: list,
        file_path: str,
        fs: s3fs.S3FileSystem
        ) -> bool:
    """
    Save results to file .txt
    The number of codes used for generation should not exceed 100.
    Else, upload in .parquet format

    Args:
        codes (list): List of codes used for the generation.
        names (list): List of names for the codes used for the generation.
        labels (list of lists): List of generated labels for each code.
        file_path (str): The .parquet file path.
        fs (float): The filesystem for exportation.

    Returns:
        bool: True if the file has been correctly saved.
    """

    # Basic validation
    if not (len(codes) == len(names) == len(labels)):
        raise "Codes, names, and labels must have the same length."
        return False

    if len(codes) == 0:
        raise "Empty input: nothing to export."
        return False

    # Flatten structure
    rows = []

    for code, name, generated_labels in zip(codes, names, labels):

        if not isinstance(generated_labels, list):
            raise "Labels must be stored in a list."
            return False

        for label in generated_labels:
            rows.append({
                "code": code,
                "name": name,
                "label": label
            })

    # Save to parquet
    df = pd.DataFrame(rows)

    if fs is None:
        try:
            existing_df = pd.read_parquet(file_path)
            df = pd.concat([existing_df, df], ignore_index=True)
        except FileNotFoundError:
            logger.info(f"No file found at location{file_path}, creating...")
    else:
        try:
            with fs.open(file_path, 'rb') as f:
                existing_df = pd.read_parquet(f)
            df = pd.concat([existing_df, df], ignore_index=True)
        except FileNotFoundError:
            logger.info(f"No file found at location{file_path}, creating...")

    df.to_parquet(
        file_path,
        engine="pyarrow",
        index=False,
        filesystem=fs
    )

    return True
