import logging

from langchain.text_splitter import TokenTextSplitter
from langchain_openai import OpenAIEmbeddings
from neo4j_graph.graph_builder.config import URL_EMBEDDING_API

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
    return OpenAIEmbeddings(
        model=model_name,
        openai_api_base=URL_EMBEDDING_API,
        openai_api_key="EMPTY",
        tiktoken_enabled=False,
    )