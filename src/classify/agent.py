import logging

from classify.base import BaseClassifier

logger = logging.getLogger(__name__)


class AgentClassifier(BaseClassifier):
    async def classify_one(self, query: str) -> str:
        try:

            logger.info("📌 Niveau 5 : %s", selected_code)
            return 

        except Exception as e:
            logger.exception("Erreur classification : %s", e)
            raise
