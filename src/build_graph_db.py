import logging
import os

from dotenv import load_dotenv
from langchain_community.document_loaders import DataFrameLoader

from constants.paths import NOTICES_PATH
from utils.cypher import create_parent_child_relationships
from utils.data import load_notices
from utils.logging import configure_logging
from vector_db.loaders import create_root_node, create_vector_db, get_embedding_model, setup_graph
from vector_db.utils import truncate_docs_to_max_tokens

configure_logging()
load_dotenv()

logger = logging.getLogger(__name__)

COLUMNS_TO_KEEP = [
    "ID",
    "CODE",
    "NAME",
    "PARENT_ID",
    "PARENT_CODE",
    "LEVEL",
    "FINAL",
    "Implementation_rule",
    "Includes",
    "IncludesAlso",
    "Excludes",
    "text_content",
]

EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", None)
if EMBEDDING_MODEL is None:
    raise ValueError("EMBEDDING_MODEL environment variable must be set.")

MAX_TOKENS = int(os.environ.get("MAX_TOKENS", 32000))


def run_pipeline():
    df = load_notices(NOTICES_PATH, COLUMNS_TO_KEEP)

    docs = DataFrameLoader(df, page_content_column="text_content").load()

    docs = truncate_docs_to_max_tokens(docs, MAX_TOKENS)

    emb_model = get_embedding_model(EMBEDDING_MODEL)
    _ = create_vector_db(docs, emb_model)

    _ = create_root_node()

    graph = setup_graph()
    create_parent_child_relationships(graph)


if __name__ == "__main__":
    run_pipeline()
