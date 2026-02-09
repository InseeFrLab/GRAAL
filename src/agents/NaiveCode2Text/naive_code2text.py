import os

import s3fs
from dotenv import load_dotenv
import pandas as pd

load_dotenv(override=True)

# Import original data to sample

fs = s3fs.S3FileSystem(
    client_kwargs={'endpoint_url': 'https://'+'minio.lab.sspcloud.fr'},
    key=os.environ["AWS_ACCESS_KEY_ID"],
    secret=os.environ["AWS_SECRET_ACCESS_KEY"],
    token=os.environ["AWS_SESSION_TOKEN"])

ORIGINAL_DATA_PATH = "projet-ape/data/08112022_27102024/naf2025/split/df_train.parquet"

COLUMNS_TO_KEEP = [
    "nace2025",
    "libelle",
]

df_original_data = pd.read_parquet(ORIGINAL_DATA_PATH, filesystem=fs)[COLUMNS_TO_KEEP]
