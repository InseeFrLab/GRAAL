import logging
import os

from langfuse import get_client

from agents import Runner
from agents.model_settings import ModelSettings
from src.agents.base_agent import BaseAgent, count_tool_calls
from src.agents.closers.match_verifier import MatchVerificationInput

logger = logging.getLogger(__name__)

_MOVEMENT_TOOLS = {"go_to_child", "go_to_parent"}
# NAF's deepest path (section -> division -> group -> class -> subclass) is 5 hops;
# +1 slack for one repeated-call retry along the way.
_MAX_FORCED_DESCENT_STEPS = 6


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
        informations recueillies, renvoyez votre réponse finale, dans cet ordre : une
        explication concise qui raisonne sur votre choix avant de le formuler, le code
        retenu (qui doit être une position terminale, is_final = 1), et votre niveau de
        confiance entre 0 et 1.
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
        # Used only when exploration stalls on a non-final node: restricting the toolset
        # to go_to_child alone (combined with the inherited tool_choice="required")
        # leaves the model no action but to keep descending.
        self.forced_descent_agent = self.exploration_agent.clone(
            tools=[t for t in self.tools if t.name == "go_to_child"],
        )
        # Finalize: tool_choice="none" rather than dropping `tools` — the conversation
        # history carries prior tool_calls/tool messages, and clearing `tools` while
        # those references remain hangs the backend's chat-template rendering instead
        # of erroring. Keeping tools declared (but disallowed) avoids that.
        self.finalize_agent = self.agent.clone(
            instructions=self.get_finalize_instructions(),
            model_settings=ModelSettings(temperature=0, tool_choice="none"),
        )

    def _fallback_output(self, activity: str, tool_call_count: int = 0) -> MatchVerificationInput:
        return MatchVerificationInput(
            activity=activity,
            code=self.graph.first_leaf_from(self.graph.current_code),
            proposed_explanation=(
                "Échec de la génération de la réponse finale par le modèle ; dernière "
                f"position atteinte lors de l'exploration : {self.graph.current_code}."
            ),
            proposed_confidence=0.0,
            tool_call_count=tool_call_count,
        )

    async def _step(self, agent, conversation, last_call):
        """Run one exploration turn with `agent`. If it repeats the previous tool call
        verbatim (no progress), retry once at a higher temperature with an explicit
        nudge, using a clone of the same `agent` so any tool restriction is preserved.

        Returns (conversation, call, n_tool_calls) — n_tool_calls counts every tool
        call made in this step (including the repeated-call retry), for the caller's
        running total (cf. _run_navigator_loop's tool_call_count sanity-check stat).
        """
        result = await Runner.run(agent, conversation, max_turns=1)
        conversation = result.to_input_list()
        call = _last_tool_call(result)
        n_tool_calls = count_tool_calls(result)

        if call is not None and call == last_call:
            logger.info(f"Navigator loop: repeated call {call}, retrying")
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
            retry_agent = agent.clone(
                model_settings=ModelSettings(temperature=0.4, tool_choice="required")
            )
            result = await Runner.run(retry_agent, conversation, max_turns=1)
            conversation = result.to_input_list()
            call = _last_tool_call(result)
            n_tool_calls += count_tool_calls(result)

        return conversation, call, n_tool_calls

    async def _force_descent_to_leaf(self, conversation):
        """Exploration stalled on a non-final node (step budget exhausted, still not
        is_final). The model's own stopping point is genuine signal, not an error, but
        it can never be the returned answer — so rather than let finalize accept a
        category in place of a classification, restrict the model to go_to_child only
        and keep forcing descent until it reaches a real leaf (capped to bound the
        loop; a well-formed NAF path is at most a few hops from any node).

        Returns (conversation, n_tool_calls).
        """
        last_call = None
        n_tool_calls = 0
        for _ in range(_MAX_FORCED_DESCENT_STEPS):
            if self.graph.is_current_final():
                break
            conversation = conversation + [
                {
                    "role": "user",
                    "content": (
                        f"La position actuelle ({self.graph.current_code}) n'est pas une "
                        "position terminale (is_final = 0) : elle ne peut pas être votre "
                        "réponse finale. Choisissez un enfant via go_to_child pour continuer "
                        "à descendre vers une position terminale."
                    ),
                }
            ]
            conversation, last_call, step_tool_calls = await self._step(
                self.forced_descent_agent, conversation, last_call
            )
            n_tool_calls += step_tool_calls
        return conversation, n_tool_calls

    async def _run_navigator_loop(
        self, activity: str, initial_prompt: str
    ) -> MatchVerificationInput:
        conversation = initial_prompt
        max_turns = int(os.environ["MAX_TURNS"])
        # Accumulated across every Runner.run call this loop makes (exploration steps,
        # forced descent, finalize) — a sanity-check stat, not model-reported, so it
        # stays valid even down the exception path below (whatever was tallied before
        # the failure).
        tool_calls = 0

        try:
            last_call = None
            for step in range(max_turns):
                conversation, call, step_tool_calls = await self._step(
                    self.exploration_agent, conversation, last_call
                )
                tool_calls += step_tool_calls
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

            # The model's own reasoning plateaued on a non-final node here: recorded
            # below (not discarded) as it's useful signal, but never returned as-is.
            natural_stop_code = None
            if not self.graph.is_current_final():
                natural_stop_code = self.graph.current_code
                conversation, descent_tool_calls = await self._force_descent_to_leaf(conversation)
                tool_calls += descent_tool_calls

            result = await Runner.run(self.finalize_agent, conversation, max_turns=2)
            tool_calls += count_tool_calls(
                result
            )  # expected 0 (tool_choice="none"), counted anyway
            final_output = result.final_output
            # The model is not a reliable source for this field — observed in practice
            # leaking a tool name or a raw code instead of echoing the input text — so
            # it's always overridden with the caller's own known value.
            final_output.activity = activity
            final_output.tool_call_count = tool_calls

            if natural_stop_code is not None:
                final_output.proposed_explanation = (
                    f"[Position atteinte avant descente forcée : {natural_stop_code}] "
                    + (final_output.proposed_explanation or "")
                )

            # Safety net regardless of path taken above: never surface a non-final code.
            if not self.graph.get_code_information(final_output.code).get("is_final"):
                logger.warning(
                    f"Navigator loop: finalize returned non-final code "
                    f"{final_output.code!r}, falling back"
                )
                final_output = self._fallback_output(activity, tool_call_count=tool_calls)
        except Exception as e:
            # Catches failures anywhere above (exploration/retry/forced-descent/finalize
            # Runner.run calls, e.g. openai.APITimeoutError from the shared LLM endpoint)
            # so every failure mode degrades identically instead of only the ones that
            # happen to occur during finalize. Don't re-raise: the fallback below lets
            # the caller's batch loop continue to the next query. But without the
            # explicit span update, the exception being swallowed here means the
            # enclosing @observe span (classify_navigator/classify_agentic_rag)
            # completes as a normal success in Langfuse — flag it as an error
            # explicitly so it's visible/filterable there instead of only in logs.
            logger.exception("Navigator loop failed, falling back to current position")
            get_client().update_current_span(
                level="ERROR",
                status_message=f"Navigator loop failed, fell back to last position: {e}",
            )
            final_output = self._fallback_output(activity, tool_call_count=tool_calls)

        logger.info(f"Result of the navigator loop: \n {final_output}")
        return final_output
