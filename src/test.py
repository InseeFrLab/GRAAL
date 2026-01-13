import asyncio
import logging
from src.agents.Text2Code.classifiers.navigator_classifier import NavigatorAgenticClassifier
from src.config import neo4j_config
from src.navigator.navigator import Navigator
from src.utils.logging import configure_logging

configure_logging()
logger = logging.getLogger(__name__)

query = "Boulangerie"


async def run_classifier(query: str):
    navigator = Navigator(neo4j_config)
    classifier = NavigatorAgenticClassifier(navigator)
    return await classifier(query)


async def main():
    result = await run_classifier(query)
    print(f"Résultat : {result}")


if __name__ == "__main__":
    asyncio.run(main())
