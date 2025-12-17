import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NACE codification with several methods")

    methods = parser.add_argument_group("Classification Methods")

    methods.add_argument(
        "--navigator",
        type=str,
        nargs="?",
        const="Boulangerie",
        default=None,
        help="Classify with agentic method. Default query: 'Boulangerie'",
    )
    methods.add_argument(
        "--agentic-rag",
        type=str,
        nargs="?",
        const="Boulangerie",
        default=None,
        metavar="QUERY",
        help="Classify with flat embeddings method. Default query: 'Boulangerie'",
    )

    options = parser.add_argument_group("Options")

    options.add_argument(
        "--experiment-name",
        type=str,
        default="nace-classification",
        help="Experiment name for logging/tracking (default: nace-classification)",
    )

    options.add_argument(
        "--batch-file",
        type=str,
        metavar="FILE",
        help="File containing queries to classify (one per line)",
    )

    return parser.parse_args()
