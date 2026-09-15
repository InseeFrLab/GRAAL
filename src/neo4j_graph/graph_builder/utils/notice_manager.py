import logging

import pandas as pd

from src.utils.storage import get_file_system

logger = logging.getLogger(__name__)


def load_notices(parquet_path: str, columns: list) -> pd.DataFrame:
    logger.info("Loading Parquet data from: %s", parquet_path)
    fs = get_file_system()
    df = pd.read_parquet(parquet_path, filesystem=fs)
    return df[columns]
