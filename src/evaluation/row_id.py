"""Stable row identifier shared across the eval pipeline, Langfuse tracing, and the
review apps — the same (text, code) pair always hashes to the same id, so a
prediction in a parquet, a judgment in a review JSONL, and a Langfuse trace can all
be correlated by this one value.
"""

import hashlib


def row_id_for(text: str, code: str) -> str:
    return hashlib.sha1(f"{text}|{code}".encode("utf-8")).hexdigest()[:12]
