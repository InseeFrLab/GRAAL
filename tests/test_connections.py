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

import asyncio
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


@_require_env("MLFLOW_TRACKING_URI", "MLFLOW_MODEL_URI")
def test_mlflow_connection():
    """Confirms the tracking server + artifact store are reachable.

    Deliberately does NOT call `mlflow.pyfunc.load_model`: the registered
    pyfunc model pickles a custom wrapper class from the training repo
    (`src.api_wrapper`), which doesn't exist in this repo's `src` package, so
    loading it here raises `ModuleNotFoundError`. `get_model_info` only reads
    the model's metadata (MLmodel file), which is enough to check
    connectivity without triggering that unpickling.
    """
    import mlflow

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    assert mlflow.models.get_model_info(os.environ["MLFLOW_MODEL_URI"]) is not None


@_require_env("CODIF_APE_API_USERNAME", "CODIF_APE_API_PASSWORD")
def test_supervised_model_api_connection():
    """Backs SupervisedClassifier (src/agents/Text2Code/classifiers/supervised_classifier.py).

    Exercises the real HTTP path SupervisedClassifier uses in production
    (codif-ape-API), rather than loading the model locally via MLflow.
    """
    from src.agents.Text2Code.classifiers.supervised_classifier import SupervisedClassifier

    async def _predict():
        classifier = SupervisedClassifier()
        try:
            return await classifier._predict("vente de vêtements de sport")
        finally:
            await classifier.client.aclose()

    code, confidence, libelle = asyncio.run(_predict())
    assert code
