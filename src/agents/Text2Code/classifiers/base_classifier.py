import logging
import os

from langfuse import get_client

from agents import Runner
from agents.model_settings import ModelSettings
from src.agents.base_agent import BaseAgent
from src.agents.closers.match_verifier import MatchVerificationInput

logger = logging.getLogger(__name__)

_MOVEMENT_TOOLS = {"go_to_child", "go_to_parent"}


def _last_tool_call(result):
    """(name, arguments) of the last tool call in a Runner.run result, or None."""
    for item in reversed(result.new_items):
        if item.type == "tool_call_item":
            name = getattr(item.raw_item, "name", None)
            arguments = getattr(item.raw_item, "arguments", None)
            if name is not None:
                return (name, arguments)
    return None


class BaseClassifier(BaseAgent):
    """Classifiers that drive a Navigator through a Python-owned step loop instead of
    one free-running agentic Runner.run call.

    A single model-controlled loop can't reliably both use tools and know when to
    stop: the SDK only resets tool_choice on "any tool used", with no concept of the
    domain stop condition (is_final=1). So exploration and finalization are split into
    two Agent variants (see __init__) and Python decides when to switch between them,
    using Navigator.is_current_final() as the ground truth instead of the model's
    self-reported completion.
    """

    def get_output_type(self):
        return MatchVerificationInput

    def get_model_settings(self) -> ModelSettings:
        return ModelSettings(temperature=0, tool_choice="required")

    def get_finalize_instructions(self) -> str:
        return """
        Vous avez terminé l'exploration de la hiérarchie NACE ci-dessus. Sur la base des
        informations recueillies, renvoyez votre réponse finale : le code retenu (qui doit
        être une position terminale, is_final = 1), une explication concise de votre choix,
        et votre niveau de confiance entre 0 et 1.
        """

    def __init__(self, graph):
        super().__init__(graph)
        # Exploration: tools forced, no output_type — nothing to finalize yet, so no
        # response_format competes with tool selection. Stops as soon as one tool call
        # executes (no need for the model to react to the tool result itself — Python
        # drives the next step).
        self.exploration_agent = self.agent.clone(
            output_type=None,
            tool_use_behavior="stop_on_first_tool",
        )
        # Finalize: tool_choice="none" rather than dropping `tools` — the conversation
        # history carries prior tool_calls/tool messages, and clearing `tools` while
        # those references remain hangs the backend's chat-template rendering instead
        # of erroring. Keeping tools declared (but disallowed) avoids that.
        self.finalize_agent = self.agent.clone(
            instructions=self.get_finalize_instructions(),
            model_settings=ModelSettings(temperature=0, tool_choice="none"),
        )

    def _fallback_output(self, activity: str) -> MatchVerificationInput:
        return MatchVerificationInput(
            activity=activity,
            code=self.graph.current_code,
            proposed_explanation=(
                "Échec de la génération de la réponse finale par le modèle ; dernière "
                f"position atteinte lors de l'exploration : {self.graph.current_code}."
            ),
            proposed_confidence=0.0,
        )

    async def _run_navigator_loop(
        self, activity: str, initial_prompt: str
    ) -> MatchVerificationInput:
        conversation = initial_prompt
        last_call = None
        max_turns = int(os.environ["MAX_TURNS"])

        for step in range(max_turns):
            result = await Runner.run(self.exploration_agent, conversation, max_turns=1)
            conversation = result.to_input_list()
            call = _last_tool_call(result)

            if call is not None and call == last_call:
                logger.info(f"Navigator loop: repeated call {call} at step {step}, retrying")
                conversation = conversation + [
                    {
                        "role": "user",
                        "content": (
                            "Vous venez de refaire exactement le même appel qu'à l'étape "
                            "précédente, sans faire progresser la navigation. Choisissez une "
                            "action différente : si un enfant correspond à l'activité, appelez "
                            "go_to_child avec son code exact."
                        ),
                    }
                ]
                retry_agent = self.exploration_agent.clone(
                    model_settings=ModelSettings(temperature=0.4, tool_choice="required")
                )
                result = await Runner.run(retry_agent, conversation, max_turns=1)
                conversation = result.to_input_list()
                call = _last_tool_call(result)

            last_call = call

            # Only stop right after an actual move: a read-only lookup (e.g. checking
            # the RAG-suggested starting leaf) must never end the loop on its own, even
            # if that position happens to already be is_final=1 — otherwise the model
            # can "verify" a wrong RAG seed, say so, and still return it unchanged.
            moved = call is not None and call[0] in _MOVEMENT_TOOLS
            if moved and self.graph.is_current_final():
                break
        else:
            logger.warning("Navigator loop: step budget exhausted before reaching is_final")

        try:
            result = await Runner.run(self.finalize_agent, conversation, max_turns=2)
            final_output = result.final_output
        except Exception as e:
            logger.exception(
                "Navigator loop: finalize call failed, falling back to current position"
            )
            # Don't re-raise: the fallback below lets the caller's batch loop continue
            # to the next query. But without this, the exception being swallowed here
            # means the enclosing @observe span (classify_navigator/classify_agentic_rag)
            # completes as a normal success in Langfuse — flag it as an error explicitly
            # so it's visible/filterable there instead of only in application logs.
            get_client().update_current_span(
                level="ERROR",
                status_message=f"Finalize call failed, fell back to last position: {e}",
            )
            final_output = self._fallback_output(activity)

        logger.info(f"Result of the navigator loop: \n {final_output}")
        return final_output
