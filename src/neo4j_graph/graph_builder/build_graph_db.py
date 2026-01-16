import logging

from langchain_community.document_loaders import DataFrameLoader

from src.neo4j_graph.graph_builder.config import (
    COLUMNS_TO_KEEP,
    EMBEDDING_MODEL,
    MAX_TOKENS,
    NOTICES_PATH,
)

from src.neo4j_graph.graph_builder.utils.db_manager import (
    create_parent_child_relationships,
    create_root_node,
    create_vector_db,
    setup_graph,
)

from src.neo4j_graph.graph_builder.utils.embed_manager import (
    get_embedding_model,
    truncate_docs_to_max_tokens,
)
from src.neo4j_graph.graph_builder.utils.notice_manager import load_notices
from src.utils.logging import configure_logging

configure_logging()
logger = logging.getLogger(__name__)


if EMBEDDING_MODEL is None:
    raise ValueError("EMBEDDING_MODEL environment variable must be set.")


def run_pipeline():
    df = load_notices(NOTICES_PATH, COLUMNS_TO_KEEP)

    df["text_to_embed"] = (
        df["NAME"]
        + "\n"
        + df["Implementation_rule"].fillna("")
        + "\n"
        + df["Includes"].fillna("")
        + "\n"
        + df["IncludesAlso"].fillna("")
    )

    docs = DataFrameLoader(df, page_content_column="text_to_embed").load()

    docs = truncate_docs_to_max_tokens(docs, MAX_TOKENS)

    emb_model = get_embedding_model(EMBEDDING_MODEL)
    create_vector_db(docs, emb_model)

    create_root_node()

    graph = setup_graph()
    create_parent_child_relationships(graph)


if __name__ == "__main__":
    run_pipeline()
