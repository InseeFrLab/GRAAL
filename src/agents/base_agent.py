import logging
import os
from abc import ABC, abstractmethod

from dotenv import load_dotenv
from langfuse.openai import AsyncOpenAI
from pydantic import BaseModel

from agents import (
    Agent,
    Runner,
    set_default_openai_api,
    set_default_openai_client,
    set_tracing_disabled,
)
from agents.model_settings import ModelSettings
from src.neo4j_graph.graph import Graph

logger = logging.getLogger(__name__)

load_dotenv(override=True)

client = AsyncOpenAI(
    base_url=os.environ["OPENAI_BASE_URL"],
    api_key=os.environ["OPENAI_API_KEY"],
    # Fail fast rather than hang: the default 10 min SDK timeout makes a stuck
    # request on the LLM endpoint indistinguishable from a real freeze.
    timeout=60.0,
)

set_default_openai_client(client=client, use_for_tracing=False)
set_default_openai_api("chat_completions")
set_tracing_disabled(True)


def count_tool_calls(result) -> int:
    """Number of tool calls in a single `Runner.run` result's `new_items`.

    Pure Python bookkeeping (not model-reported), attached to output objects after
    the fact as a sanity-check signal — e.g. confirming Navigator/Agentic-RAG
    actually explore the graph rather than finalizing on zero tool calls.
    """
    return sum(1 for item in result.new_items if item.type == "tool_call_item")


class BaseAgent(ABC):
    def __init__(self, graph: Graph):
        super().__init__()
        self.graph = graph
        self.tools = self.get_tools()
        self.output_type = self.get_output_type()
        self.instructions = self.get_instructions()
        self.agent = Agent(
            name=self.get_agent_name(),
            instructions=self.instructions,
            tools=self.tools,
            model=os.environ["GENERATION_MODEL"],
            model_settings=self.get_model_settings(),
            output_type=self.output_type,
        )

    @abstractmethod
    def get_agent_name(self) -> str:
        pass

    @abstractmethod
    def get_instructions(self) -> str:
        pass

    @abstractmethod
    def get_output_type(self) -> BaseModel:
        pass

    @abstractmethod
    def build_prompt(self, *args, **kwargs) -> str:
        pass

    def get_tools(self):
        """Tools this agent may call — every graph navigation tool by default.

        A per-agent decision rather than a property of the graph: an agent that never
        explores still pays for the schemas of the tools it is handed (~500 prompt
        tokens for the five graph tools) and still risks a multi-turn detour on what
        is a single-shot judgment, so the closers (cf. src.agents.closers) override
        this with no tools at all.
        """
        return self.graph.get_tools()

    def get_max_turns(self) -> int:
        """Turn budget for one `__call__` — MAX_TURNS by default.

        Global by default because the exploring agents genuinely need a large budget,
        but overridable: a single-shot agent with no tools has nothing to spend a
        second turn on, and a global 15 only means a failure there takes fifteen times
        longer to surface.
        """
        return int(os.environ["MAX_TURNS"])

    def wrap_output(self, model_output, result):
        """Turn what the model generated into what callers get back.

        The default is the model's own output. Agents whose result carries fields the
        model must not see (bookkeeping counts, logprob-derived calibration) override
        this and build the richer object here, so `get_output_type()` stays exactly
        the schema shown to the model — cf. MatchVerifier.
        """
        return model_output

    async def __call__(self, *args, **kwargs):
        prompt = self.build_prompt(*args, **kwargs)
        result = await Runner.run(self.agent, prompt, max_turns=self.get_max_turns())
        logger.info(f"Result of the __call__ in BaseAgent: \n {result.final_output}")
        output = self.wrap_output(result.final_output, result)
        if hasattr(output, "tool_call_count"):
            output.tool_call_count = count_tool_calls(result)
        return output

    def get_model_settings(self) -> ModelSettings:
        return ModelSettings(
            temperature=0,
        )
