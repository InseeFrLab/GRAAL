import logging
import os

from langchain_text_splitters import TokenTextSplitter
from langchain_openai import OpenAIEmbeddings

from src.neo4j_graph.graph_builder.config import URL_EMBEDDING_API

logger = logging.getLogger(__name__)


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
def get_embedding_model(model_name: str) -> OpenAIEmbeddings:
    """Initialize the embedding model."""
    print(f"URL EMBEDDING API: {URL_EMBEDDING_API}")
    return OpenAIEmbeddings(
        model=model_name,
        openai_api_base=URL_EMBEDDING_API,
        # L'API LLM du Datalab refuse désormais les appels sans clé (401 avec "EMPTY").
        openai_api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
        tiktoken_enabled=False,
        # Sans tiktoken, langchain chargerait le tokenizer Hugging Face du nom de modèle
        # pour redécouper les textes — or un nom servi par l'API (« qwen3-embedding-8b »)
        # n'est pas un identifiant du Hub. Le découpage est déjà fait en amont, cf.
        # truncate_docs_to_max_tokens.
        check_embedding_ctx_length=False,
    )
