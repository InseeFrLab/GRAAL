import os
import logging
import time

from dotenv import load_dotenv
import s3fs

from src.agents.NaiveCode2Text.prompts import prompt_builder, wording_generator
from src.agents.NaiveCode2Text.code_retrieval import code_sampler, code_specifier
from src.neo4j_graph.graph import Graph, Neo4JConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv(override=True)

POPULATION_PATH = "projet-ape/data/08112022_27102024/naf2025/split/df_train.parquet"
CODE_COLUMN = "nace2025"
N_CODES = 5
MODEL = "devstral-2:123b"
TEMPERATURE = 2

start = time.perf_counter()

fs = s3fs.S3FileSystem(
    client_kwargs={'endpoint_url': 'https://'+'minio.lab.sspcloud.fr'},
    key=os.environ["AWS_ACCESS_KEY_ID"],
    secret=os.environ["AWS_SECRET_ACCESS_KEY"],
    token=os.environ["AWS_SESSION_TOKEN"]
)

# Sampling from original data
logger.info("Sampling from data...")
code_list = code_sampler.sample_codes_lazy(
    fs=fs,
    population_path=POPULATION_PATH,
    code_column=CODE_COLUMN,
    n_codes=N_CODES
)

# NAF to NACE
logger.info("Transforming codes from NAF to NACE...")
code_list = [code_specifier.NAF_to_NACE(code) for code in code_list]

print(code_list)

results = []

# Neo4j connection
logger.info("Connecting to Neo4j graph...")
notice_graph = Graph(Neo4JConfig(
    url=os.environ["NEO4J_URL"],
    username=os.environ["NEO4J_USERNAME"],
    password=os.environ["NEO4J_PWD"]
))

# Prompt generation
logger.info("Generating prompts...")
for i, code in enumerate(code_list):
    logger.info(f"Processing step {i+1}...")
    code_spec = code_specifier.get_code_information(notice_graph, code)
    system_prompt = prompt_builder.build_system_prompt()
    user_prompt = prompt_builder.business_oriented_user_prompt(code_spec)
    result = wording_generator.ask_model(
        system_prompt=system_prompt,
        model=MODEL,
        user_prompt=user_prompt,
        temperature=TEMPERATURE
    )
    result = wording_generator.retrieve_wording(result)
    results.append(result)

end = time.perf_counter()

logger.info("Saving results...")
file_name = f"generation_{MODEL}_temp{TEMPERATURE}".replace(":", "-").replace(".", "") + ".txt"

with open(
        file=f"src/agents/NaiveCode2Text/sample_results/{file_name}",
        mode='w'
        ) as f:
    f.write(f"{N_CODES} wordings have been generated in {end-start:.2f} sec.\n\n")
    for i in range(N_CODES):
        f.write("====================================")
        f.write("\n")
        code = code_list[i]
        f.write("Code: " + code + "\n")
        code_spec = code_specifier.get_code_information(notice_graph, code)
        f.write("Name: " + code_spec["name"] + "\n")
        f.write("Result: " + results[i] + "\n")
        f.write("\n")
