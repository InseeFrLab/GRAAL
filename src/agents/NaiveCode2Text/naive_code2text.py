import os
import logging
from logging.handlers import RotatingFileHandler
import time
import traceback
import asyncio

from dotenv import load_dotenv
import s3fs
from openai import AsyncOpenAI

from src.agents.NaiveCode2Text.config_naive import \
    MODEL, TEMPERATURE, OUTPUT_PATH, N_CODES, POPULATION_PATH, CODE_COLUMN, \
    OUTPUT_FORMAT, SAVE_BATCH_SIZE, LANGUAGE, NB_LABELS, PROMPT_PATH, \
    INCLUDES_DIVIDER, EXAMPLES_DIVIDER, EXCLUDE_DIVIDER, RANDOM_SPEC_SAMPLING, \
    RANDOM_INCLUDES_GEOM_PROB, RANDOM_INCLUDES_MIN, RANDOM_INCLUDES_MAX, \
    RANDOM_EXAMPLES_GEOM_PROB, RANDOM_EXAMPLES_MIN, RANDOM_EXAMPLES_MAX, \
    GENERATION_BATCH_SIZE
from src.agents.NaiveCode2Text.prompts import prompt_builder, label_generator
from src.agents.NaiveCode2Text.code_retrieval import code_sampler, code_specifier
from src.neo4j_graph.graph import Graph, Neo4JConfig

# Logger
logging.basicConfig(
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

handler = RotatingFileHandler(
    "naive_code2text.log",
    maxBytes=10_000_000,  # 10MB
    backupCount=5
)

formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(message)s"
)

handler.setFormatter(formatter)
logger.addHandler(handler)

# Environment
load_dotenv(override=True)

if __name__ == "__main__":
    # Clock for speed testing
    if OUTPUT_FORMAT == ".txt":
        start = time.perf_counter()

    # Access configurations
    FS = s3fs.S3FileSystem(
        client_kwargs={'endpoint_url': os.environ["AWS_ENDPOINT_URL"]},
        key=os.environ["AWS_ACCESS_KEY_ID"],
        secret=os.environ["AWS_SECRET_ACCESS_KEY"],
        token=os.environ["AWS_SESSION_TOKEN"]
    )

    # LLM_API_KEY = os.environ["LLM_API_KEY"]
    URL_GENERATION_API = os.environ["URL_GENERATION_API"]
    LLM_CLIENT = AsyncOpenAI(base_url=URL_GENERATION_API)

    # Sampling from original data
    logger.info("Sampling from data...")
    code_list = code_sampler.sample_codes_lazy(
        fs=FS,
        population_path=POPULATION_PATH,
        code_column=CODE_COLUMN,
        n_codes=N_CODES
    )

    # NAF to NACE
    logger.info("Transforming codes from NAF to NACE...")
    code_list = [code_specifier.NAF_to_NACE(code) for code in code_list]

    # Neo4j connection
    logger.info("Connecting to Neo4j graph...")
    notice_graph = Graph(Neo4JConfig(
        url=os.environ["NEO4J_URL"],
        username=os.environ["NEO4J_USERNAME"],
        password=os.environ["NEO4J_PWD"]
    ))

    # Define an automatic name for output
    file_name = f"generation_{MODEL}_temp{TEMPERATURE}".replace(":", "-").replace(".", "") \
                + OUTPUT_FORMAT
    FINAL_PATH = OUTPUT_PATH + file_name

    # Prompt generation
    logger.info("Generating prompts...")

    name_list = []
    label_list = []

    # Model set up
    LabelGenerationModel = label_generator.build_label_generation_model(NB_LABELS)

    system_prompt = prompt_builder.build_system_prompt(
            prompt_path=PROMPT_PATH,
            language=LANGUAGE,
            nb_labels=NB_LABELS
        )

    user_prompts = []

    for i, code in enumerate(code_list):
        logger.info(f"Processing step {i+1}...")

        try:

            # Get code details from Neo4j
            code_details = code_specifier.get_code_information(
                graph=notice_graph,
                code=code
                )

            # For exportation purpose
            name_list.append(code_details["name"])

            # Build prompts
            user_prompt = prompt_builder.build_user_prompt(
                code_details=code_details,
                language=LANGUAGE,
                nb_labels=NB_LABELS,
                includes_divider=INCLUDES_DIVIDER,
                examples_divider=EXAMPLES_DIVIDER,
                excludes_divider=EXCLUDE_DIVIDER,
                random_spec_sampling=RANDOM_SPEC_SAMPLING,
                random_includes_geom_prob=RANDOM_INCLUDES_GEOM_PROB,
                random_includes_min=RANDOM_INCLUDES_MIN,
                random_includes_max=RANDOM_INCLUDES_MAX,
                random_examples_geom_prob=RANDOM_EXAMPLES_GEOM_PROB,
                random_examples_min=RANDOM_EXAMPLES_MIN,
                random_examples_max=RANDOM_EXAMPLES_MAX
                )

            user_prompts.append(user_prompt)

            if len(user_prompts) == GENERATION_BATCH_SIZE:

                # Ask the chatbot
                generations = asyncio.run(label_generator.ask_model_multiple(
                    system_prompt=system_prompt,
                    user_prompts=user_prompts,
                    llm_client=LLM_CLIENT,
                    model=MODEL,
                    temperature=TEMPERATURE,
                    LabelGeneration=LabelGenerationModel,
                    max_concurrency=GENERATION_BATCH_SIZE
                ))

                for generation in generations:
                    label_list.append(generation.labels)

                user_prompts = []

        except Exception as e:
            label_list.append(["None"]*10)
            name_list.append(code_details["name"])
            logger.warn(f"Error raised, skipping...\nDetails: {e}")
            logger.info(f"Traceback:\n{traceback.format_exc()}")

        if OUTPUT_FORMAT == ".parquet" and (i+1) % SAVE_BATCH_SIZE == 0:
            logger.info("Saving intermediate results...")
            label_generator.export_to_parquet(
                codes=code_list[i+1-SAVE_BATCH_SIZE:i+1],
                names=name_list,
                labels=label_list,
                file_path=FINAL_PATH,
                fs=FS
                )
            label_list = []
            name_list = []

    end = time.perf_counter()

    if OUTPUT_FORMAT == ".txt":
        logger.info("Saving results to txt...")
        label_generator.export_to_txt(
            codes=code_list,
            names=name_list,
            labels=label_list,
            file_path=FINAL_PATH,
            generation_time=end-start
            )

    elif OUTPUT_FORMAT == ".parquet":
        logger.info("Saving final results...")
        first_unsaved_index = SAVE_BATCH_SIZE*(N_CODES//SAVE_BATCH_SIZE)
        if first_unsaved_index < N_CODES:
            label_generator.export_to_parquet(
                codes=code_list[SAVE_BATCH_SIZE*(N_CODES//SAVE_BATCH_SIZE):],
                names=name_list,
                labels=label_list,
                file_path=FINAL_PATH,
                fs=FS
                )
