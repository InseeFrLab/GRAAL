import asyncio
import logging
import os

from openai import OpenAI

from constants.prompts import SYS_PROMPT
from llm.schema import Response

logger = logging.getLogger(__name__)

missing = [var for var in ["GENERATION_MODEL", "OPENAI_API_KEY"] if not os.environ.get(var)]
if missing:
    raise EnvironmentError(f"Missing required env vars: {', '.join(missing)}")

GENERATION_MODEL = os.environ.get("GENERATION_MODEL")


async def get_llm_choice(prompt: str, client: OpenAI, retries: int = 3, delay: float = 2.0) -> str:
    for attempt in range(1, retries + 1):
        try:
            response = await client.beta.chat.completions.parse(
                model=GENERATION_MODEL,
                messages=[
                    {"role": "system", "content": SYS_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                response_format=Response,
                # extra_body={"guided_decoding_backend": "guidance"}, Guidance doesn't work with mistral from 0.8.4 vllm
            )
            return response.choices[0].message.parsed.code

        except Exception as e:
            logger.warning("⚠️ LLM erreur tentative %d : %s", attempt, str(e))
            if attempt == retries:
                raise
            await asyncio.sleep(delay)
