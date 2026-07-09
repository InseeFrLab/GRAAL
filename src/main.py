import asyncio
import logging
import sys
from datetime import datetime

from langfuse import get_client, observe, propagate_attributes

from src.agents.closers.match_verifier import MatchVerificationInput, MatchVerifier
from src.agents.Text2Code.classifiers.agentic_rag import AgenticRAGClassifier
from src.agents.Text2Code.classifiers.navigator_classifier import NavigatorAgenticClassifier
from src.agents.Text2Code.classifiers.supervised_classifier import SupervisedClassifier
from src.config import neo4j_config
from src.navigator.navigator import Navigator
from src.neo4j_graph.graph import Graph
from src.utils.logging import configure_logging
from src.utils.parser import parse_args

configure_logging()
logger = logging.getLogger(__name__)


@observe
async def classify_navigator(
    query: str | list[str], experiment_name: str = "Navigator Classification"
):
    """Classify using agentic method

    Args:
        query: A single query string or a list of query strings
        experiment_name: Name of the experiment

    Returns:
        Single result dict if query is str, list of result dicts if query is list
    """
    get_client().update_current_trace(
        name=experiment_name, tags=[experiment_name], metadata={"experiment_name": experiment_name}
    )

    # Normalize input to always work with a list
    queries = [query] if isinstance(query, str) else query
    is_single = isinstance(query, str)

    logger.info(f"Navigator classification: {len(queries)} query/queries")

    navigator = Navigator(neo4j_config)

    results = []
    for q in queries:
        logger.info(f"Classifying: {q}")
        logger.info(f"Current position of the navigator: {navigator.current_code}")
        classifier = NavigatorAgenticClassifier(navigator)
        result = await classifier(q)
        results.append(result)
        logger.info(f"Le résultat de la classification est : {result}")
        navigator.reset_to_root()

    # Return single result or list based on input type
    return results[0] if is_single else results


@observe
async def classify_agentic_rag(
    query: str | list[str],
    experiment_name: str = "Agentic RAG Classification",
):
    """Classify using embedding retrieval as a Navigator warm-start

    Finds the closest code by vector similarity, then lets the Navigator agent verify
    and refine that starting position instead of walking down from the root
    (cf. classify_navigator).

    Args:
        query: A single query string or a list of query strings
        experiment_name: Name of the experiment

    Returns:
        Single MatchVerificationInput if query is str, list of them if query is list
    """
    get_client().update_current_trace(
        name=experiment_name, tags=[experiment_name], metadata={"experiment_name": experiment_name}
    )

    queries = [query] if isinstance(query, str) else query
    is_single = isinstance(query, str)

    logger.info(f"Agentic RAG classification: {len(queries)} query/queries")

    navigator = Navigator(neo4j_config)
    classifier = AgenticRAGClassifier(navigator)

    results = []
    for q in queries:
        logger.info(f"Classifying: {q}")
        result = await classifier(q)
        results.append(result)
        logger.info(f"Le résultat de la classification est : {result}")

    return results[0] if is_single else results


@observe
async def classify_supervised(
    query: str | list[str],
    experiment_name: str = "Supervised Model Classification",
):
    """Classify using the production supervised model, via the deployed codif-ape-API

    Serves as the reference baseline for comparing the agentic methods
    (cf. cadrage §3.3-B, note de conception). Requires CODIF_APE_API_USERNAME
    and CODIF_APE_API_PASSWORD to be set (CODIF_APE_API_URL is optional).

    Args:
        query: A single query string or a list of query strings
        experiment_name: Name of the experiment

    Returns:
        Single MatchVerificationInput if query is str, list of them if query is list
    """
    get_client().update_current_trace(
        name=experiment_name, tags=[experiment_name], metadata={"experiment_name": experiment_name}
    )

    queries = [query] if isinstance(query, str) else query
    is_single = isinstance(query, str)

    logger.info(f"Supervised model classification: {len(queries)} query/queries")

    classifier = SupervisedClassifier()

    results = []
    for q in queries:
        logger.info(f"Classifying: {q}")
        result = await classifier(q)
        results.append(result)
        logger.info(f"Le résultat de la classification est : {result}")

    return results[0] if is_single else results


async def verify_classification(prediction: MatchVerificationInput, verifier: MatchVerifier):
    """Chain a classifier prediction into the MatchVerifier for double-checking

    Args:
        prediction: Output of a classifier (activity, proposed code, explanation, confidence)
        verifier: MatchVerifier agent instance

    Returns:
        MatchVerificationResult (is_match, confidence, explanation)
    """
    verification = await verifier(prediction)
    logger.info(f"Le résultat de la vérification est : {verification}")
    return verification


@observe
async def process_batch_file(
    filepath: str, method_func, experiment_name: str, verifier: MatchVerifier | None = None
):
    """Process a batch file with queries"""
    get_client().update_current_trace(
        name=experiment_name, tags=[experiment_name], metadata={"experiment_name": experiment_name}
    )

    logger.info(f"Processing batch file: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        queries = [line.strip() for line in f if line.strip()]

    logger.info(f"Found {len(queries)} queries to process")

    results = []
    for i, query in enumerate(queries, 1):
        logger.info(f"Processing {i}/{len(queries)}: {query}")
        result = await method_func(query, experiment_name)
        verification = None
        if verifier is not None and isinstance(result, MatchVerificationInput):
            verification = await verify_classification(result, verifier)
        results.append({"query": query, "result": result, "verification": verification})

    return results


async def main():
    """
    Main entry point
    """
    args = parse_args()
    logger.info(f"Main called with arguments: {args}")

    try:
        # Determine which method(s) to use
        methods_to_run = []

        if args.navigator:
            methods_to_run.append(("navigator", args.navigator, classify_navigator))

        if args.agentic_rag:
            methods_to_run.append(("agentic-rag", args.agentic_rag, classify_agentic_rag))

        if args.supervised:
            methods_to_run.append(("supervised", args.supervised, classify_supervised))

        # No method specified
        if not methods_to_run:
            logger.info("Use --help to see available options")
            return 1

        verifier = MatchVerifier(Graph(neo4j_config)) if args.verify else None

        # Batch file mode
        if args.batch_file:
            if len(methods_to_run) > 1:
                logger.warning("Multiple methods specified, using first one for batch")

            method_name, _, method_func = methods_to_run[0]
            logger.info(f"Batch mode with method: {method_name}")

            results = await process_batch_file(
                args.batch_file, method_func, args.experiment_name, verifier=verifier
            )

            print("\n" + "=" * 80)
            print("BATCH RESULTS")
            print("=" * 80)
            for result in results:
                code = getattr(result["result"], "code", result["result"])
                line = f"  {result['query']:40s} → {code}"
                if result["verification"] is not None:
                    status = "✅" if result["verification"].is_match else "❌"
                    line += f" | verifier: {status} ({result['verification'].confidence:.2f})"
                print(line)
            print("=" * 80)
            return 0

        # Normal mode: run each method with its query
        for method_name, query, method_func in methods_to_run:
            logger.info(f"\n{'=' * 80}")
            logger.info(f"Method: {method_name}")
            logger.info(f"Query: {query}")
            logger.info(f"Experiment: {args.experiment_name}")
            logger.info(f"{'=' * 80}")

            result = await method_func(query, args.experiment_name)

            print(f"\n✅ Result: {result}")

            if verifier is not None and isinstance(result, MatchVerificationInput):
                verification = await verify_classification(result, verifier)
                status = "✅ validé" if verification.is_match else "❌ rejeté"
                print(f"🔍 Verification: {status} ({verification.confidence:.2f})")
                print(f"   {verification.explanation}")

        return 0

    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        return 130

    except Exception as e:
        logger.exception(f"Error: {e}")
        return 1


if __name__ == "__main__":
    langfuse = get_client()
    session_id = f"base_agent_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    with propagate_attributes(session_id=session_id):
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
