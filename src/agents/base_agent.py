import os
from dotenv import load_dotenv
from abc import ABC, abstractmethod
from pydantic import BaseModel

from openai import AsyncOpenAI
from agents import (
    Agent,
    Runner,
    function_tool,
    set_default_openai_api,
    set_default_openai_client,
    set_tracing_disabled,
)
from agents.model_settings import ModelSettings

from src.neo4j_graph.Graph import Graph, make_tools

load_dotenv()
client = AsyncOpenAI(
    base_url=os.environ["OPENAI_BASE_URL"],
    api_key=os.environ["OPENAI_API_KEY"],
)
set_default_openai_client(client=client, use_for_tracing=False)
set_default_openai_api("chat_completions")
set_tracing_disabled(True)

class BaseAgent(ABC):

    def __init__(self, graph: Graph):
        super().__init__()
        self.graph = graph
        self.tools = make_tools(graph)

        self.output_type = self.get_output_type()
        self.instructions = self.get_instructions()
        self.agent = Agent(
            name=self.get_agent_name(),
            instructions=self.instructions,
            tools=self.tools,
            model=os.environ["GENERATION_MODEL"],
            model_settings=self.get_model_settings(),
            output_type=self.output_type
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