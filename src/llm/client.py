import base64
import os
from contextlib import asynccontextmanager

import httpx
from dotenv import load_dotenv
from langfuse.openai import AsyncOpenAI

load_dotenv()

if os.environ.get("URL_LLM_API") is None:
    raise ValueError("URL_LLM_API environment variable must be set.")
URL_LLM_API = os.environ.get("URL_LLM_API")


def setup_langfuse():
    # Validate required environment variables
    required_vars = [
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
        "LANGFUSE_HOST",
        "OPENAI_API_KEY",
    ]
    missing = [var for var in required_vars if not os.environ.get(var)]
    if missing:
        raise EnvironmentError(f"Missing required env vars: {', '.join(missing)}")

    # Build Langfuse Basic Auth header
    LANGFUSE_AUTH = base64.b64encode(f"{os.environ['LANGFUSE_PUBLIC_KEY']}:{os.environ['LANGFUSE_SECRET_KEY']}".encode()).decode()

    # Set OTEL exporter environment variables
    os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = os.environ["LANGFUSE_HOST"] + "/api/public/otel"
    os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"Authorization=Basic {LANGFUSE_AUTH}"


# Load environment variables from .env file if it exists
load_dotenv()


@asynccontextmanager
async def get_llm_client():
    setup_langfuse()
    client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"), base_url=URL_LLM_API, timeout=httpx.Timeout(1 * 60 * 60))
    try:
        yield client
    finally:
        await client.close()


def sync_get_llm_client():
    setup_langfuse()
    client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"), base_url=URL_LLM_API, timeout=httpx.Timeout(1 * 60 * 60))
    return client
