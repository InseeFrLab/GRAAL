from typing import Optional
from pydantic import BaseModel, Field

from src.agents.base_agent import BaseAgent
from src.agents.closers.match_verifier import MatchVerifier, MatchVerificationInput


class Text2CodeOutput(BaseModel):
    code: str = Field(description="The appropriate code for the given activity")
    confidence: float = Field(description="Confidence level in the proposed code, between 0 and 1", ge=0, le=1)
    explanation: str = Field(description="Concise explanation of the code choice")
    verifier_decision: Optional[bool] = Field(
        default=None,
        description="Indicates whether the verifier validated the match between the activity and the code. Optional field."
    )


class Text2CodeAgent():
    def __init__(self, graph, classifier, verifier: bool= True):
        super().__init__(graph)

        self.classifier = classifier

        if verifier:
            self.verifier = MatchVerifier(graph)

    def __call__(self, activity: str) -> Text2CodeOutput:

        classifier_output = self.classifier(activity=activity)

        if hasattr(self, 'verifier'):
            if not isinstance(classifier_output, MatchVerificationInput):
                raise ValueError("The classifier should return a MatchVerifierInput type.")

            verification_result = self.verifier(classifier_output)
            verifier_decision = verification_result.is_match
        else:
            verifier_decision = None

        return Text2CodeOutput(
            code=classifier_output.proposed_code,
            confidence=classifier_output.confidence,
            explanation=classifier_output.proposed_explanation,
            verifier_decision=verifier_decision
        )
