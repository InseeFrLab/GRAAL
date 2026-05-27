import os

from dotenv import load_dotenv

load_dotenv(override=True)

# EMBED MANAGER
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", None)
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", 32000))
URL_EMBEDDING_API = os.environ.get("URL_EMBEDDING_API", 'EMPTY')

# NEO4J DB MANAGER
# NEO4J_URL = "EMPTY"
# NEO4J_USERNAME = "EMPTY"
# NEO4J_PWD = "EMPTY"
NEO4J_URL = os.environ["NEO4J_URL"]
NEO4J_USERNAME = os.environ["NEO4J_USERNAME"]
NEO4J_PWD = os.environ["NEO4J_PWD"]

# NOTICE MANAGER: all notices (except NAF) have been created using
# script from ./utils/convert_to_parquet.py
# Notice FR NAF
NOTICES_PATH = "projet-ape/notices/Notices-NAF2025-FR.parquet"
# Notice EN NACE
# NOTICES_PATH = "projet-ape/notices/NACE_Rev2.1_Structure_Explanatory_Notes_EN.parquet"
# Notice COICOP EN
# NOTICES_PATH = "projet-ape/notices/coicop-2018_envoi_rmes_20251022_en.parquet"
# Notice COICOP FR
# NOTICES_PATH = "projet-ape/notices/coicop-2018_envoi_rmes_20251022_fr.parquet"

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", None)
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", None)

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
