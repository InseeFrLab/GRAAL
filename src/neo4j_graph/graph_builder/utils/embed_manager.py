import logging
import os
import requests
from typing import List

from langchain_text_splitters import TokenTextSplitter
from langchain_core.embeddings import Embeddings

from src.neo4j_graph.graph_builder.config import URL_EMBEDDING_API, OPENAI_API_KEY, EMBEDDING_MODEL

logger = logging.getLogger(__name__)


class CustomAPIEmbeddings(Embeddings):
    """Custom embeddings class that directly calls the API without langchain dependencies."""

    def __init__(self, model_name: str, base_url: str, api_key: str):
        self.model_name = model_name
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        embeddings = []
        for text in texts:
            embedding = self._get_single_embedding(text)
            embeddings.append(embedding)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        return self._get_single_embedding(text)

    def _get_single_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text."""
        payload = {
            "input": text,
            "model": self.model_name
        }

        try:
            response = requests.post(
                f"{self.base_url}/embeddings",
                json=payload,
                headers=self.headers,
                timeout=30
            )
            response.raise_for_status()
            data = response.json()

            if "data" in data and len(data["data"]) > 0:
                return data["data"][0]["embedding"]
            else:
                raise ValueError(f"Unexpected API response format: {data}")

        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling embedding API: {e}")
            raise


def truncate_docs_to_max_tokens(docs, max_tokens):
    splitter = TokenTextSplitter(chunk_size=max_tokens, chunk_overlap=0)
    truncated_docs = []

    for doc in docs:
        original_text = doc.page_content
        chunks = splitter.split_text(original_text)

        if len(chunks) > 1:
            logger.warning(f"Document truncated to {max_tokens} tokens. Metadata: {doc.metadata}")

        doc.page_content = chunks[0]
        truncated_docs.append(doc)

    return truncated_docs


# TODO: Remove langchain depency
# TODO: factorize embedder manager outside from the graph builder ?
def get_embedding_model(model_name: str = EMBEDDING_MODEL) -> CustomAPIEmbeddings:
    """Initialize the custom embedding model that directly calls the API."""
    logger.info(f"URL EMBEDDING API: {URL_EMBEDDING_API}")
    logger.info(f"Using custom embedding model: {model_name}")

    return CustomAPIEmbeddings(
        model_name=model_name,
        base_url=URL_EMBEDDING_API,
        api_key=OPENAI_API_KEY,
    )
