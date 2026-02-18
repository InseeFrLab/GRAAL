import openai
import os 
import asyncio
import logging
from dotenv import load_dotenv

load_dotenv(override=True)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def chat(client)->None:
    logger.info(os.environ["GENERATION_MODEL"])
    response = await client.chat.completions.create(
        model=os.environ["GENERATION_MODEL"],
        messages=[
            {"role": "system", "content": "Talk like a pirate."},
            {
                "role": "user",
                "content": "How do I check if a Python object is an instance of a class?",
            },
        ],
    )
    logger.info(f"Chat: {response}")
    return 


async def response(client)->None:
    logger.info(os.environ["GENERATION_MODEL"])
    response = await client.responses.create(
        model=os.environ["GENERATION_MODEL"],
        instructions="You are a coding assistant that talks like a pirate.",
        input="How do I check if a Python object is an instance of a class?",
    )
    logger.info(f"Response: {response}")
    return 


if __name__ == "__main__": 
    logger.info(f"URL: {os.environ["OPENAI_BASE_URL"]}")
    logger.info(f"KEY: {os.environ["OPENAI_API_KEY"]}")
    client = openai.AsyncOpenAI(
        base_url=os.environ["OPENAI_BASE_URL"],
        api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
    )
    logger.info("Client connected")
    try:
        logger.info("Chat") 
        asyncio.run(chat(client=client))
    except Exception as e: 
        logger.info(f"Chat doesnt work: {e}")
    try: 
        logger.info("Response")
        asyncio.run(response(client=client))
    except Exception as e:
        logger.info(f"Response does not work: {e}")