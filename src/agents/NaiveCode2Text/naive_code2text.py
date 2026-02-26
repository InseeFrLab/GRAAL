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
    GENERATION_BATCH_SIZE, CONVERT_NAF_TO_NACE, CONVERT_TO_PROPER_NAF, \
    LABEL_COLUMN, N_FEWSHOT, USE_FEWSHOT, EXHAUSTIVE_SAMPLING, N_SAMPLES_PER_CODE, \
    MODEL_NAME
from src.agents.NaiveCode2Text.prompts import prompt_builder, label_generator, fewshot_builder
from src.agents.NaiveCode2Text.code_retrieval import code_sampler, code_specifier, fewshot_sampler
from langchain_neo4j import Neo4jGraph

# Logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

handler = RotatingFileHandler(
    "naive_code2text.log",
    maxBytes=10_000_000,  # 10MB
    backupCount=5
)

formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(message)s"
)

handler.setFormatter(formatter)
root_logger.addHandler(handler)

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

    # Neo4j connection
    root_logger.info("Connecting to Neo4j graph...")
    notice_graph = Neo4jGraph(
        url=os.environ["NEO4J_URL"],
        username=os.environ["NEO4J_USERNAME"],
        password=os.environ["NEO4J_PWD"]
    )

    # LLM_API_KEY = os.environ["LLM_API_KEY"]
    URL_GENERATION_API = os.environ["URL_GENERATION_API"]
    LLM_CLIENT = AsyncOpenAI(base_url=URL_GENERATION_API)

    # Sampling from original data
    code_list = []

    root_logger.info("Sampling from data...")
    if EXHAUSTIVE_SAMPLING:
        code_list += code_sampler.get_all_codes(
                        graph=notice_graph,
                        n_samples_per_code=N_SAMPLES_PER_CODE
                    )

        N_EXHAUSTIVE = len(code_list)

        if N_EXHAUSTIVE >= N_CODES:
            root_logger.warn("Exhaustive sampling requested, but N_CODES is too low.")
            root_logger.warn("Keeping exhaustivity, but sampling 0 code randomly.")
            N_CODES = 0

        else:
            N_CODES -= N_EXHAUSTIVE

    if N_CODES > 0:
        code_list += code_sampler.sample_codes_lazy(
                        fs=FS,
                        population_path=POPULATION_PATH,
                        code_column=CODE_COLUMN,
                        n_codes=N_CODES
                    )

    # NAF to NACE
    if CONVERT_NAF_TO_NACE:
        root_logger.info("Transforming codes from NAF to NACE...")
        new_code_list = [code_specifier.NAF_to_NACE(code) for code in code_list]

    elif CONVERT_TO_PROPER_NAF:
        root_logger.info("Transforming codes from NAF to NACE...")
        new_code_list = [code_specifier.to_proper_NAF(code) for code in code_list]

    else:
        new_code_list = code_list

    # Define an automatic name for output
    if EXHAUSTIVE_SAMPLING:
        exhaust_string = "_exhaustive"
    else:
        exhaust_string = ""

    if MODEL_NAME is None and MODEL_NAME:
        MODEL_NAME = MODEL
        if MODEL_NAME is None:
            MODEL_NAME = ""

    file_name = f"generation_{MODEL_NAME}_temp{TEMPERATURE}_{LANGUAGE}_fewshot{N_FEWSHOT}"\
                .replace(":", "-").replace(".", "") + exhaust_string + OUTPUT_FORMAT
    FINAL_PATH = OUTPUT_PATH + file_name

    # Few-shot sampling
    if USE_FEWSHOT:
        root_logger.info("Sampling examples for few-shot...")
        codes_fewshot = fewshot_sampler.sample_fewshot_lazy_multi(
            fs=FS,
            population_path=POPULATION_PATH,
            code_column=CODE_COLUMN,
            codes=code_list,
            label_column=LABEL_COLUMN,
            n_fewshot=N_FEWSHOT
        )

    # Prompt generation
    root_logger.info("Creating prompts...")

    # System prompt
    if USE_FEWSHOT:
        system_prompt = fewshot_builder.build_fewshot_system_prompt(
                prompt_path=PROMPT_PATH,
                language=LANGUAGE,
                nb_labels=NB_LABELS
            )
    else:
        system_prompt = prompt_builder.build_system_prompt(
                prompt_path=PROMPT_PATH,
                language=LANGUAGE,
                nb_labels=NB_LABELS
            )

    valid_items = []

    # User prompt
    for i, (new_code, fewshot) in enumerate(zip(new_code_list, codes_fewshot)):
        try:

            # Get code details from Neo4j
            code_details = code_specifier.get_code_information(
                graph=notice_graph,
                code=new_code
                )

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

            if USE_FEWSHOT:
                user_prompt += fewshot_builder.add_fewshot_user_prompt(
                    fewshot=fewshot,
                    language=LANGUAGE
                )

            valid_items.append({
                "code": new_code,
                "name": code_details["name"],
                "prompt": user_prompt
            })

        except Exception as e:
            root_logger.warning(f"Error preparing code {new_code}, skipping...\nDetails: {e}")
            root_logger.info(traceback.format_exc())
            continue

    # Model set up
    root_logger.info("Generating labels...")

    LabelGenerationModel = label_generator.build_label_generation_model(NB_LABELS)

    results_buffer = []

    for i in range(0, len(valid_items), GENERATION_BATCH_SIZE):
        root_logger.info(f"Processing batch {i//GENERATION_BATCH_SIZE}...")

        batch = valid_items[i:i + GENERATION_BATCH_SIZE]
        prompts = [item["prompt"] for item in batch]

        try:
            generations = asyncio.run(
                label_generator.ask_model_multiple(
                    system_prompt=system_prompt,
                    user_prompts=prompts,
                    llm_client=LLM_CLIENT,
                    model=MODEL,
                    temperature=TEMPERATURE,
                    LabelGeneration=LabelGenerationModel,
                    max_concurrency=len(prompts)
                )
            )

            for item, generation in zip(batch, generations):
                results_buffer.append({
                    "code": item["code"],
                    "name": item["name"],
                    "labels": generation.labels
                })

        except Exception as e:
            root_logger.warning(f"Batch generation failed, skipping... Details: {e}")
            root_logger.info(traceback.format_exc())

        if OUTPUT_FORMAT == ".parquet" and len(results_buffer) >= SAVE_BATCH_SIZE:
            root_logger.info("Saving intermediate results...")

            try:
                codes = [r["code"] for r in results_buffer]
                names = [r["name"] for r in results_buffer]
                labels = [r["labels"] for r in results_buffer]

                label_generator.export_to_parquet(
                    codes=codes,
                    names=names,
                    labels=labels,
                    file_path=FINAL_PATH,
                    fs=FS
                )

            except Exception as e:
                root_logger.warning(f"Buffer exportation failed, dropped... Details: {e}")
                root_logger.info(traceback.format_exc())

            results_buffer = []

    end = time.perf_counter()

    if OUTPUT_FORMAT == ".txt":
        root_logger.info("Saving results to txt...")

        codes = [r["code"] for r in results_buffer]
        names = [r["name"] for r in results_buffer]
        labels = [r["labels"] for r in results_buffer]

        label_generator.export_to_txt(
            codes=codes,
            names=names,
            labels=labels,
            file_path=FINAL_PATH,
            generation_time=end-start
            )

    if OUTPUT_FORMAT == ".parquet" and results_buffer:
        root_logger.info("Saving final remaining results...")

        try:
            codes = [r["code"] for r in results_buffer]
            names = [r["name"] for r in results_buffer]
            labels = [r["labels"] for r in results_buffer]

            label_generator.export_to_parquet(
                codes=codes,
                names=names,
                labels=labels,
                file_path=FINAL_PATH,
                fs=FS
            )

        except Exception as e:
            root_logger.warning(f"Buffer exportation failed, dropped... Details: {e}")
            root_logger.info(traceback.format_exc())
