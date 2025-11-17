from api.routes.common import build_classification_router
from classify.agent import 

router = build_classification_router(
    prefix="/agent",
    tag="Agent",
    classifier_cls=AgentClassifier,
)
