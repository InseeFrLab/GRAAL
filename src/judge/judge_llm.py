import os

from dotenv import load_dotenv

os.chdir("codif-ape-graph-rag")
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_openai import ChatOpenAI

from src.judge import judge_no_agentic_prompt
from src.llm.client import sync_get_llm_client
from src.tools import graph, tools

client = sync_get_llm_client()
load_dotenv()

llm = ChatOpenAI(
    model=os.environ["GENERATION_MODEL"],
    temperature=0,
    openai_api_base=os.environ["URL_LLM_API"],
    openai_api_key=os.environ["LLMLAB_API_KEY"],
)

query_str = """MATCH (node {CODE:"96.21H"}) RETURN node"""
res = graph.query(query_str)[0]["node"]
res_1 = res["text_content"]

query_str = """MATCH (node {CODE:"20.42Y"}) RETURN node"""
res = graph.query(query_str)[0]["node"]
res_2 = res["text_content"]


# --- Create agent ---

chain = judge_no_agentic_prompt | llm

# Use the chain
response = chain.invoke(
    {
        "input": f"""L'activé à coder est 'boulanger'.

Voici deux potentiels codes:
1) {res_1} \n
2) {res_2}

Veuillez déterminer lequel des deux codes est le plus approprié pour l'activité donnée et expliquer pourquoi."""
    }
)


agent = create_tool_calling_agent(llm, tools, judge_no_agentic_prompt)
executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=30,  # Allow more iterations for thorough exploration
    handle_parsing_errors=True,
)
