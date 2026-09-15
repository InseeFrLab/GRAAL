"""Shared S3 (SSP Cloud/Onyxia MinIO datalake) helpers.

Consolidates the S3FileSystem construction previously duplicated, with
slightly different env var names, across notice_manager.py, build_eval_set.py
and convert_to_parquet.py. Also lets output paths transparently target S3: any
path prefixed with "s3://" is written through s3fs instead of the local
filesystem, so a script's --output-dir/--output flag can point at S3 without
code changes.
"""

import os

import s3fs

S3_PREFIX = "s3://"


def get_file_system(token: str | None = None) -> s3fs.S3FileSystem:
    """Builds an s3fs.S3FileSystem from the environment (Onyxia/Datalab conventions).

    Reads AWS_ENDPOINT_URL if set (already a full URL), else falls back to
    AWS_S3_ENDPOINT (bare host, prepended with https://).
    """
    endpoint_url = os.environ.get("AWS_ENDPOINT_URL") or f"https://{os.environ['AWS_S3_ENDPOINT']}"
    options = {
        "client_kwargs": {"endpoint_url": endpoint_url},
        "key": os.environ["AWS_ACCESS_KEY_ID"],
        "secret": os.environ["AWS_SECRET_ACCESS_KEY"],
    }
    token = token if token is not None else os.environ.get("AWS_SESSION_TOKEN")
    if token is not None:
        options["token"] = token
    return s3fs.S3FileSystem(**options)


def is_s3_path(path: str) -> bool:
    return path.startswith(S3_PREFIX)


def makedirs(path: str) -> None:
    """os.makedirs locally; a no-op on S3, where prefixes need no creation."""
    if not is_s3_path(path):
        os.makedirs(path, exist_ok=True)


def open_path(path: str, mode: str = "r", **kwargs):
    """open() that transparently supports s3:// paths alongside local ones."""
    if is_s3_path(path):
        return get_file_system().open(path, mode, **kwargs)
    return open(path, mode, **kwargs)


def path_exists(path: str) -> bool:
    if is_s3_path(path):
        return get_file_system().exists(path)
    return os.path.exists(path)


def remove(path: str) -> None:
    if is_s3_path(path):
        get_file_system().rm(path)
    else:
        os.remove(path)
