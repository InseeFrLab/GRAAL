import asyncio
import logging
import sys

from src.agents.Text2Code.classifiers.navigator_classifier import NavigatorAgenticClassifier
from src.config import neo4j_config
from src.utils.logging import configure_logging
from src.utils.parser import parse_args

configure_logging()
logger = logging.getLogger(__name__)


async def classify_navigator(query: str, experiment_name: str):
    """Classify using agentic method"""
    logger.info(f"Navigator classification: {query}")
    # TODO: add the management for exp_name
    classifier = NavigatorAgenticClassifier(neo4j_config)
    result = await classifier(query)
    return result


async def classify_agentic_rag(query: str, experiment_name: str):
    """Classify using flat embeddings"""
    logger.info(f"Flat embeddings classification: {query}")

    return "10.71C"


async def process_batch_file(filepath: str, method_func, experiment_name: str):
    """Process a batch file with queries"""
    logger.info(f"Processing batch file: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        queries = [line.strip() for line in f if line.strip()]

    logger.info(f"Found {len(queries)} queries to process")

    results = []
    for i, query in enumerate(queries, 1):
        logger.info(f"Processing {i}/{len(queries)}: {query}")
        result = await method_func(query, experiment_name)
        results.append({"query": query, "code": result})

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

        # No method specified
        if not methods_to_run:
            logger.error("No classification method specified!")
            logger.info("Use --help to see available options")
            return 1

        # Batch file mode
        if args.batch_file:
            if len(methods_to_run) > 1:
                logger.warning("Multiple methods specified, using first one for batch")

            method_name, _, method_func = methods_to_run[0]
            logger.info(f"Batch mode with method: {method_name}")

            results = await process_batch_file(args.batch_file, method_func, args.experiment_name)

            print("\n" + "=" * 80)
            print("BATCH RESULTS")
            print("=" * 80)
            for result in results:
                print(f"  {result['query']:40s} → {result['code']}")
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

        return 0

    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        return 130

    except Exception as e:
        logger.exception(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
