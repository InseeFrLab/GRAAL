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
        self.tools = self.graph.get_tools()
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

    async def __call__(self, *args, **kwargs):
        prompt = self.build_prompt(*args, **kwargs)
        result = await Runner.run(self.agent, prompt, max_turns=int(os.environ["MAX_TURNS"]))
        logger.info(f"Result of the __call__ in BaseAgent: \n {result.final_output}")
        output = result.final_output
        if hasattr(output, "tool_call_count"):
            output.tool_call_count = count_tool_calls(result)
        return output

    def get_model_settings(self) -> ModelSettings:
        return ModelSettings(
            temperature=0,
        )
