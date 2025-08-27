"""
This run can be run directly using:
        uv run evaluate.py

Arguments can be passed using the command line, see --help and parse_args() for more details.

"""

import asyncio
import datetime
import logging
import os
import time
from math import ceil
from typing import List

import httpx
import humanize
import mlflow
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from evaluation.args_parser import parse_args
from evaluation.data import get_all_levels, load_test_data, process_response, save_predictions
from utils.data import get_df_naf
from utils.logging import configure_logging

load_dotenv()
configure_logging()
logger = logging.getLogger(__name__)

API_URL = "http://localhost:5000"
TIMEOUT = 3600


async def evaluate_method(
    client: httpx.AsyncClient,
    method: str,
    queries: List[str],
    df_naf: pd.DataFrame,
    ground_truth: pd.DataFrame,
    batch_size: int = 10,  # Add batch_size parameter
) -> pd.DataFrame:
    try:
        logger.info(f"🚀 Starting evaluation for '{method}'")

        start_time = time.time()
        num_batches = ceil(len(queries) / batch_size)
        all_preds = pd.DataFrame(columns=["code_ape"])  # Initialize an empty DataFrame

        for i in tqdm(range(num_batches)):
            batch_queries = queries[i * batch_size : (i + 1) * batch_size]
            response = await client.post(
                f"{API_URL}/{method}/batch",
                json={"queries": batch_queries},
            )
            response.raise_for_status()
            batch_preds = process_response(response.json())
            all_preds = pd.concat([all_preds, batch_preds], ignore_index=True)  # Concatenate batch_preds

        end_time = time.time()
        elapsed_seconds = end_time - start_time
        elapsed_td = datetime.timedelta(seconds=elapsed_seconds)

        preds_levels = get_all_levels(all_preds, df_naf, "code_ape")
        save_predictions(all_preds, method)

        with mlflow.start_run():
            mlflow.log_param("method", method)
            mlflow.log_param("num_samples", len(all_preds))
            mlflow.log_param("elapsed_time", humanize.precisedelta(elapsed_td))
            mlflow.log_param("embedding_model", os.environ["EMBEDDING_MODEL"])
            mlflow.log_param("generation_model", os.environ["GENERATION_MODEL"])

            accs = (preds_levels == ground_truth).mean()
            for lvl, acc in enumerate(accs, 1):
                mlflow.log_metric(f"accuracy_lvl_{lvl}", acc)

        logger.info(f"✅ Finished evaluation for '{method}'")
        return preds_levels

    except httpx.TimeoutException:
        logger.error(
            f"⏳ Timeout during '{method}': Request took more than {humanize.precisedelta(datetime.timedelta(seconds=TIMEOUT))}"
        )
        return pd.DataFrame()

    except httpx.HTTPStatusError as http_exc:
        logger.error(f"🚨 HTTP error during '{method}': {http_exc.response.status_code} - {http_exc.response.text}")
        return pd.DataFrame()

    except httpx.RequestError as req_exc:
        logger.error(f"📡 Network error during '{method}': {type(req_exc).__name__} - {req_exc}")
        return pd.DataFrame()

    except Exception as e:
        logger.error(f"❌ Unexpected error during '{method}': {type(e).__name__} - {e}")
        return pd.DataFrame()


async def evaluate_all(
    methods: List[str],
    queries: List[str],
    df_naf: pd.DataFrame,
    ground_truth: pd.DataFrame,
) -> List[pd.DataFrame]:
    async with httpx.AsyncClient(timeout=httpx.Timeout(TIMEOUT)) as client:
        tasks = [evaluate_method(client, method, queries, df_naf, ground_truth) for method in methods]
        return await asyncio.gather(*tasks)


if __name__ == "__main__":
    assert "MLFLOW_TRACKING_URI" in os.environ, "Please set the MLFLOW_TRACKING_URI environment variable."

    args = parse_args()

    methods = (
        ["flat-embeddings", "flat-rag", "hierarchical-embeddings", "hierarchical-rag"]
        if args.entry_point == "all"
        else [args.entry_point]
    )

    df_test = load_test_data(args.num_samples)
    df_naf = get_df_naf()
    ground_truth = get_all_levels(df_test, df_naf, col="nace2025")
    queries = df_test["libelle"].tolist()

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(args.experiment_name)

    logger.info(f"Evaluating {len(methods)} method(s) with {args.num_samples} samples...")

    if args.entry_point == "all":
        asyncio.run(evaluate_all(methods, queries, df_naf, ground_truth))
    else:

        async def eval_single():
            async with httpx.AsyncClient(timeout=httpx.Timeout(TIMEOUT)) as client:
                return await evaluate_method(client, methods[0], queries, df_naf, ground_truth)

        asyncio.run(eval_single())
