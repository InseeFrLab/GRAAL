import asyncio

from agents import Agent, GuardrailFunctionOutput, OpenAIChatCompletionsModel, Runner
from dotenv import load_dotenv
from pydantic import BaseModel

# os.chdir("codif-ape-graph-rag/src/")
from llm.client import sync_get_llm_client

client = sync_get_llm_client()
load_dotenv()


class HomeworkOutput(BaseModel):
    is_homework: bool
    reasoning: str


guardrail_agent = Agent(
    name="Guardrail check",
    model=OpenAIChatCompletionsModel(
        model="mistralai/Mistral-Small-24B-Instruct-2501",
        openai_client=client,
    ),
    instructions="Check if the user is asking about homework.",
    output_type=HomeworkOutput,
)

math_tutor_agent = Agent(
    name="Math Tutor",
    model=OpenAIChatCompletionsModel(
        model="mistralai/Mistral-Small-24B-Instruct-2501",
        openai_client=client,
    ),
    handoff_description="Specialist agent for math questions",
    instructions="You provide help with math problems. Explain your reasoning at each step and include examples",
)

history_tutor_agent = Agent(
    name="History Tutor",
    model=OpenAIChatCompletionsModel(
        model="mistralai/Mistral-Small-24B-Instruct-2501",
        openai_client=client,
    ),
    handoff_description="Specialist agent for historical questions",
    instructions="You provide assistance with historical queries. Explain important events and context clearly.",
)


async def homework_guardrail(ctx, agent, input_data):
    result = await Runner.run(guardrail_agent, input_data, context=ctx.context)
    final_output = result.final_output_as(HomeworkOutput)
    return GuardrailFunctionOutput(
        output_info=final_output,
        tripwire_triggered=not final_output.is_homework,
    )


triage_agent = Agent(
    name="Triage Agent",
    model=OpenAIChatCompletionsModel(
        model="mistralai/Mistral-Small-24B-Instruct-2501",
        openai_client=client,
    ),
    instructions="You determine which agent to use based on the user's homework question",
    handoffs=[history_tutor_agent, math_tutor_agent],
    # input_guardrails=[
    #     InputGuardrail(guardrail_function=homework_guardrail),
    # ],
)


async def main():
    result = await Runner.run(triage_agent, "What is the capital of France?")
    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
