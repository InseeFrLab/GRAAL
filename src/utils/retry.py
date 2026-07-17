"""Shared retry helper for LLM-backed calls across the eval pipeline.

Not every classifier catches its own transient failures: BaseClassifier
(Navigator, Agentic-RAG) has its own Python-owned fallback for a stalled/failed
navigator loop, but SummaryAgenticClassifier, SupervisedClassifier, MatchVerifier
and CodeChooser have none of their own — an exception (e.g. an LLM timeout) just
propagates to the caller. Before this module existed, only
src.evaluation.verify_train_labels retried such failures (its own local
`_call_with_retries`); run_eval.py and evaluate_eval_set_multi_method.py each
caught the exception once and gave up. This gives every caller the same
retry-then-give-up behavior instead of each eval script inventing its own.
"""

import logging

logger = logging.getLogger(__name__)


async def call_with_retries(coro_fn, max_attempts: int = 2, what: str = "Call"):
    """Await `coro_fn()`, retrying up to `max_attempts` times on any exception.

    Returns the result, or None if every attempt failed (logged as a warning per
    retry, and as a full exception only once all attempts are exhausted).

    A retry isn't only logged: if the result has an `attempt_count` attribute (cf.
    MatchVerificationInput/MatchVerificationResult/CodeChoice), it's set to the
    attempt number that succeeded (1 = no retry needed), the same "attach after the
    fact, don't ask the model" convention as `tool_call_count` — so which rows
    actually needed a retry is queryable later, not just visible in logs at the time.
    """
    for attempt in range(1, max_attempts + 1):
        try:
            result = await coro_fn()
            if hasattr(result, "attempt_count"):
                result.attempt_count = attempt
            return result
        except Exception:
            if attempt < max_attempts:
                logger.warning(f"{what} failed (attempt {attempt}/{max_attempts}), retrying")
            else:
                logger.exception(f"{what} failed after {max_attempts} attempts")
    return None
