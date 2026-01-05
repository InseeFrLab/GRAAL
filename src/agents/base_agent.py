import os
from abc import ABC, abstractmethod

from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel
import base64

from agents import (
    Agent,
    Runner,
    set_default_openai_api,
    set_default_openai_client,
    set_tracing_disabled,
)

from agents.model_settings import ModelSettings
from src.neo4j_graph.graph import Graph

load_dotenv()

LANGFUSE_AUTH = base64.b64encode(
    f"{os.environ['LANGFUSE_PUBLIC_KEY']}:{os.environ['LANGFUSE_SECRET_KEY']}".encode()
).decode()
os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = os.environ["LANGFUSE_BASE_URL"] + "/api/public/otel"
os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"Authorization=Basic {LANGFUSE_AUTH}"

client = AsyncOpenAI(
    base_url=os.environ["OPENAI_BASE_URL"],
    api_key=os.environ["OPENAI_API_KEY"],
)

set_default_openai_client(client=client, use_for_tracing=False)
set_default_openai_api("chat_completions")
#set_tracing_disabled(True)

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
        result = await Runner.run(self.agent, prompt)
        return result

    def get_model_settings(self) -> ModelSettings:
        return ModelSettings(
            temperature=0,
        )
