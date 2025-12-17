from typing import Optional
from pydantic import BaseModel, Field

from src.agents.base_agent import BaseAgent
from src.agents.closers.match_verifier import MatchVerifier, MatchVerificationInput
from src.agents.Text2Code.classifiers.base_classifier import BaseClassifier
from src.neo4j_graph.Graph import Graph


class Text2CodeOutput(BaseModel):
    code: str = Field(description="The appropriate code for the given activity")
    classifier_confidence: float = Field(description="Confidence level in the proposed code, between 0 and 1", ge=0, le=1)
    explanation: str = Field(description="Concise explanation of the code choice")
    verifier_decision: Optional[bool] = Field(
        default=None,
        description="Indicates whether the verifier validated the match between the activity and the code. Optional field."
    )
    verifier_confidence: Optional[float] = Field(
        default=None,
        description="Confidence level of the verifier's decision, between 0 and 1. Optional field.",
        ge=0,
        le=1
    )
    verifier_explanation: Optional[str] = Field(
        default=None,
        description="Concise explanation provided by the verifier for its decision. Optional field."
    )


class Text2CodeAgent():
    def __init__(self, classifier: BaseClassifier, verifier: bool= True):

        self.classifier = classifier

        if verifier:
            self.verifier = MatchVerifier(self.classifier.graph)

    async def __call__(self, activity: str) -> Text2CodeOutput:

        classifier_output = await self.classifier(activity=activity)

        if hasattr(self, 'verifier'):
            if not isinstance(classifier_output, MatchVerificationInput):
                raise ValueError("The classifier should return a MatchVerifierInput type.")

            verification_result = await self.verifier(classifier_output)
            verifier_decision = verification_result.final_output.is_match
            verifier_confidence = verification_result.final_output.confidence
            verifier_explanation = verification_result.final_output.explanation
        else:
            verifier_decision = None
            verifier_confidence = None
            verifier_explanation = None

        return Text2CodeOutput(
            code=classifier_output.proposed_code,
            classifier_confidence=classifier_output.proposed_confidence,
            explanation=classifier_output.proposed_explanation,
            verifier_decision=verifier_decision,
            verifier_confidence=verifier_confidence,
            verifier_explanation=verifier_explanation
        )
