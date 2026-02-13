import os
import logging
import re

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)
logger = logging.getLogger(__name__)

LLM_API_KEY = os.environ["LLM_API_KEY"]
LLM_URL = os.environ["LLM_URL"]
MODEL = "gpt-oss:20b"

client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_URL)


def ask_model(system_prompt, model, user_prompt, temperature):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": user_prompt}
        ],
        temperature=temperature,
    )

    return response.choices[0].message.content


def retrieve_wording(message, delimiter="", nb_labels=10):
    labels = re.split(r"\n\s+\d+.\s+", message)
    if len(labels) != nb_labels:
        logger.warn("The answer might not be a wording.")
    return labels
