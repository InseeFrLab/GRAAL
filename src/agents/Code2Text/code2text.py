from typing import Optional

from pydantic import BaseModel, Field

from src.agents.closers.match_verifier import MatchVerificationInput, MatchVerifier
from src.agents.Code2Text.agent import Code2TextAgent
from src.neo4j_graph.graph import Graph


class Code2TextOutput(BaseModel):
    code: str = Field(description="The code for which we want to generate synthetic data")
    generated_description: str = Field(description="The generated text for the given code")
    verifier_decision: Optional[bool] = Field(
        default=None,
        description="Indicates whether the verifier validated the match between the activity and the code. Optional field.",
    )
    verifier_confidence: Optional[float] = Field(
        default=None,
        description="Confidence level of the verifier's decision, between 0 and 1. Optional field.",
        ge=0,
        le=1,
    )
    verifier_explanation: Optional[str] = Field(
        default=None,
        description="Concise explanation provided by the verifier for its decision. Optional field.",
    )


class Code2Text:
    def __init__(self, graph: Graph, verifier: bool = True):
        self.agent = Code2TextAgent(graph=graph)
        if verifier:
            self.verifier = MatchVerifier(self.agent.graph)

    async def __call__(self, code: str) -> Code2TextOutput:
        synth_data_gen_output = await self.agent(code=code)
        print("agent output ", synth_data_gen_output)
        print(type(synth_data_gen_output))
        if hasattr(self, "verifier"):
            match_verifier_input = MatchVerificationInput(
                activity=synth_data_gen_output.generated_description,
                code=synth_data_gen_output.code,
                proposed_explanation="None",
                proposed_confidence=1,
            )
            verification_result = await self.verifier(match_verifier_input)
            verifier_decision = verification_result.final_output.is_match
            verifier_confidence = verification_result.final_output.confidence
            verifier_explanation = verification_result.final_output.explanation
        else:
            verifier_decision = None
            verifier_confidence = None
            verifier_explanation = None

        return Code2TextOutput(
            code=synth_data_gen_output.code,
            generated_description=synth_data_gen_output.generated_description,
            verifier_decision=verifier_decision,
            verifier_confidence=verifier_confidence,
            verifier_explanation=verifier_explanation,
        )
