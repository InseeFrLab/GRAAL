"""Connectivity smoke tests for the external services the project depends on.

Each test performs a minimal, real network call against a service (Neo4j,
the generation LLM API, the embedding API, S3, Langfuse) using the environment
variables documented in docs/framework.md. A test is skipped automatically
when its required variables aren't set, so the suite stays green on a
fresh clone or in CI without Datalab/Onyxia secrets, while still catching
misconfigured endpoints/credentials when they are.

Run explicitly with:
    uv run pytest tests/test_connections.py -v
"""

import os

import pytest
from dotenv import load_dotenv

load_dotenv(override=True)


def _require_env(*names):
    missing = [name for name in names if not os.environ.get(name)]
    return pytest.mark.skipif(
        bool(missing),
        reason=f"missing env var(s): {', '.join(missing)}",
    )


@_require_env("NEO4J_URL", "NEO4J_USERNAME", "NEO4J_PWD")
def test_neo4j_connection():
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(
        os.environ["NEO4J_URL"],
        auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PWD"]),
    )
    try:
        driver.verify_connectivity()
    finally:
        driver.close()


@_require_env("OPENAI_BASE_URL", "OPENAI_API_KEY")
def test_generation_model_api_connection():
    from openai import OpenAI

    client = OpenAI(base_url=os.environ["OPENAI_BASE_URL"], api_key=os.environ["OPENAI_API_KEY"])
    assert list(client.models.list())


@_require_env("URL_EMBEDDING_API", "EMBEDDING_MODEL")
def test_embedding_api_connection():
    from openai import OpenAI

    client = OpenAI(
        base_url=os.environ.get("URL_EMBEDDING_API"),
        api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
    )
    assert list(client.models.list())


_S3_PROBE_PATH = "projet-ape/notices/Notices-NAF2025-FR.parquet"


@_require_env("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY")
def test_s3_connection():
    import s3fs

    fs = s3fs.S3FileSystem(
        client_kwargs={"endpoint_url": os.environ.get("AWS_ENDPOINT_URL")},
        key=os.environ["AWS_ACCESS_KEY_ID"],
        secret=os.environ["AWS_SECRET_ACCESS_KEY"],
        token=os.environ.get("AWS_SESSION_TOKEN"),
        # Fail fast rather than hang: this endpoint has been observed to time
        # out (instead of erroring) on requests it won't serve.
        config_kwargs={"connect_timeout": 5, "read_timeout": 8, "retries": {"max_attempts": 1}},
    )
    assert fs.exists(_S3_PROBE_PATH)


@_require_env("LLM_URL", "LLM_API_KEY")
def test_naive_code2text_llm_connection():
    """NaiveCode2Text uses its own LLM_URL/LLM_API_KEY pair (see naive_code2text.py)."""
    from openai import OpenAI

    client = OpenAI(base_url=os.environ["LLM_URL"], api_key=os.environ["LLM_API_KEY"])
    assert list(client.models.list())


@_require_env("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY")
def test_langfuse_connection():
    """Langfuse backs the application tracing used in src/main.py."""
    from langfuse import Langfuse

    client = Langfuse(
        public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
        secret_key=os.environ["LANGFUSE_SECRET_KEY"],
        base_url=os.environ.get("LANGFUSE_BASE_URL"),
    )
    assert client.auth_check()
