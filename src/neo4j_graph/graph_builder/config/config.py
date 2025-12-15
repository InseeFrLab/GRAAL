import os
from dotenv import load_dotenv

load_dotenv()

# EMBED MANAGER 
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", None)
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", 32000))
URL_EMBEDDING_API = "EMPTY"

# NEO4J DB MANAGER
NEO4J_URL = "EMPTY"
NEO4J_USERNAME = "EMPTY"
NEO4J_PWD = "EMPTY"

# NOTICE MANAGER
NOTICES_PATH = "projet-ape/notices/Notices-NAF2025-FR.parquet"
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
