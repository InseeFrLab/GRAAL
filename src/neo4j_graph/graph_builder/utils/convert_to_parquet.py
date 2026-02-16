"""
Used to convert notices into correct parquet format.
- XLSL format used for NACE (found online - Eurostat)
- CSV format used for COICOP (internal, same as on the Insee website)
"""

# openpyxl==3.1.5 and pyarrow==23.0.0 have been used

import os

from dotenv import load_dotenv
import s3fs
import pandas as pd
import numpy as np

load_dotenv(override=True)

fs = s3fs.S3FileSystem(
    client_kwargs={'endpoint_url': 'https://'+'minio.lab.sspcloud.fr'},
    key=os.environ["AWS_ACCESS_KEY_ID"],
    secret=os.environ["AWS_SECRET_ACCESS_KEY"],
    token=os.environ["AWS_SESSION_TOKEN"])

# NACE
# XLSX_PATH = "projet-ape/notices/NACE_Rev2.1_Structure_Explanatory_Notes_EN.xlsx"
# PARQUET_PATH = "projet-ape/notices/NACE_Rev2.1_Structure_Explanatory_Notes_EN.parquet"

# COICOP
CSV_PATH = "projet-ape/notices/coicop-2018_envoi_rmes_20251022.csv"
# PARQUET_PATH = "projet-ape/notices/coicop-2018_envoi_rmes_20251022_en.parquet"
PARQUET_PATH = "projet-ape/notices/coicop-2018_envoi_rmes_20251022_fr.parquet"

ORDERED_COLUMNS = [
    "ID",
    "CODE",
    "NAME",
    "PARENT_ID",
    "PARENT_CODE",
    "LEVEL",
    "FINAL",
    "text_content",
    "Implementation_rule",
    "Includes",
    "IncludesAlso",
    "Excludes",
]


def convert_xlsx_to_parquet(xlsx_path, parquet_path):
    with fs.open(xlsx_path, 'rb') as f:
        df = pd.read_excel(f)
    df.rename(columns={"HEADING": "NAME"}, inplace=True)
    df["FINAL"] = (df["ID"].str.len() == 4).astype(int)
    df["text_content"] = (
        df["NAME"].fillna("") + " " +
        df["Implementation_rule"].fillna("") + " " +
        df["Includes"].fillna("") + " " +
        df["IncludesAlso"].fillna("") + " " +
        df["Excludes"].fillna("")
    )
    df = df[ORDERED_COLUMNS]

    df.to_parquet(parquet_path, filesystem=fs, engine="pyarrow")


def convert_csv_to_parquet(csv_path, parquet_path, language_suffix="_en"):
    with fs.open(csv_path, 'rb') as f:
        df = pd.read_csv(f, sep=";")

    # Select columns
    columns_to_keep = [
        "type",
        "parent",
        "code",
        "label" + language_suffix,
        "note_generale" + language_suffix,
        "contenu_central" + language_suffix,
        "contenu_additionnel" + language_suffix,
        "note_exclusion" + language_suffix
    ]
    df = df[columns_to_keep].copy()

    # Rename columns
    new_columns_names = {
        "parent": "PARENT_CODE",
        "code": "CODE",
        "label" + language_suffix: "NAME",
        "note_generale" + language_suffix: "text_content",
        "contenu_central" + language_suffix: "Includes",
        "contenu_additionnel" + language_suffix: "IncludesAlso",
        "note_exclusion" + language_suffix: "Excludes"
    }
    df.rename(columns=new_columns_names, inplace=True)

    # Level mapping
    def level_mapping(x: str):
        if x == "Division":
            return 1
        if x == "Groupe":
            return 2
        if x == "Classe":
            return 3
        if x == "Sous-classe":
            return 4
        if x == "Poste":
            return 5
    df["LEVEL"] = df["type"].map(level_mapping)

    # ID mapping
    df["ID"] = df["CODE"].str.replace(".", "")
    df["PARENT_ID"] = df["PARENT_CODE"].str.replace(".", "")
    df["FINAL"] = (df["type"] == "Poste").astype(int)

    # Adding empty implementation rule for format
    df["Implementation_rule"] = np.nan

    # Ordering columns
    df = df[ORDERED_COLUMNS]

    # Exporting
    df.to_parquet(parquet_path, filesystem=fs, engine="pyarrow")


if __name__ == "__main__":
    # convert_xlsx_to_parquet(XLSX_PATH, PARQUET_PATH)
    convert_csv_to_parquet(
        CSV_PATH,
        PARQUET_PATH,
        language_suffix="_fr"
        # language_suffix="_en"
    )
