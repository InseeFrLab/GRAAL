import os
import logging
from logging.handlers import RotatingFileHandler
import time
import traceback
import asyncio

from dotenv import load_dotenv
import s3fs
from openai import AsyncOpenAI
from langchain_neo4j import Neo4jGraph

from src.agents.NaiveCode2Text import config_naive as cfg
from src.agents.NaiveCode2Text.code_retrieval import code_sampler, code_specifier
from src.agents.NaiveCode2Text.data_preprocessing import NAF_preprocessing
from src.agents.NaiveCode2Text.fewshot import fewshot_prompt_builder, fewshot_sampler
from src.agents.NaiveCode2Text.label_generation import label_generator, label_exportation
from src.agents.NaiveCode2Text.prompt_creation import system_prompt_builder, \
    exhaustive_user_prompt_builder, random_user_prompt_builder

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

    # ======================== INIT ==========================
    root_logger.info("Initialization...")

    # Clock for speed testing
    if cfg.OUTPUT_FORMAT == ".txt":
        start = time.perf_counter()

    # Access configurations
    FS = s3fs.S3FileSystem(
        client_kwargs={'endpoint_url': os.environ["AWS_ENDPOINT_URL"]},
        key=os.environ["AWS_ACCESS_KEY_ID"],
        secret=os.environ["AWS_SECRET_ACCESS_KEY"],
        token=os.environ["AWS_SESSION_TOKEN"]
    )

    # Neo4j connection
    NOTICE_GRAPH = Neo4jGraph(
        url=os.environ["NEO4J_URL"],
        username=os.environ["NEO4J_USERNAME"],
        password=os.environ["NEO4J_PWD"]
    )

    # Connecting to Generation Client
    URL_GENERATION_API = os.environ["URL_GENERATION_API"]
    LLM_CLIENT = AsyncOpenAI(base_url=URL_GENERATION_API)

    # Model set up
    LabelGenerationModel = label_generator.build_label_generation_model(cfg.N_LABELS_PER_GEN)

    # Automatic name for output
    FINAL_PATH = label_exportation.create_file_name(
        output_path=cfg.OUTPUT_PATH,
        output_format=cfg.OUTPUT_FORMAT,
        temperature=cfg.TEMPERATURE,
        language=cfg.LANGUAGE,
        exhaustive_sampling=cfg.EXHAUSTIVE_SAMPLING,
        use_fewshot=cfg.USE_FEWSHOT,
        n_fewshot=cfg.N_FEWSHOT,
        model_name=cfg.MODEL_NAME,
        model=cfg.MODEL,
        )

    # ======================== SAMPLING CODES ==========================
    root_logger.info("Sampling from data...")

    code_list = []

    if cfg.EXHAUSTIVE_SAMPLING:
        code_list += code_specifier.get_all_codes(
                        graph=NOTICE_GRAPH,
                        n_samples_per_code=cfg.MIN_PROMPTS_PER_CODE
                    )

        code_list = [NAF_preprocessing.to_bad_NAF(code) for code in code_list]

        N_EXHAUSTIVE = len(code_list)

    if cfg.N_RANDOM_CODES > 0:
        code_list += code_sampler.sample_codes_lazy(
                        fs=FS,
                        population_path=cfg.POPULATION_PATH,
                        code_column=cfg.CODE_COLUMN,
                        n_codes=cfg.N_RANDOM_CODES
                    )

    # NAF to NACE
    if cfg.CONVERT_NAF_TO_NACE:
        root_logger.info("Transforming codes from NAF to NACE...")
        new_code_list = [NAF_preprocessing.NAF_to_NACE(code) for code in code_list]

    elif cfg.CONVERT_TO_PROPER_NAF:
        root_logger.info("Transforming codes to proper NAF...")
        new_code_list = [NAF_preprocessing.to_proper_NAF(code) for code in code_list]

    else:
        new_code_list = code_list

    # ======================== FEW-SHOT ==========================
    if cfg.USE_FEWSHOT:
        root_logger.info("Sampling examples for few-shot...")
        codes_fewshot = fewshot_sampler.sample_fewshot_lazy_multi(
            fs=FS,
            population_path=cfg.POPULATION_PATH,
            code_column=cfg.CODE_COLUMN,
            codes=code_list,
            label_column=cfg.LABEL_COLUMN,
            n_fewshot=cfg.N_FEWSHOT
        )

    # ======================== PROMPT CREATION ==========================
    root_logger.info("Creating prompts...")

    # System prompt
    if cfg.USE_FEWSHOT:
        system_prompt = fewshot_prompt_builder.build_fewshot_system_prompt(
                prompt_path=cfg.PROMPT_PATH,
                language=cfg.LANGUAGE,
                nb_labels=cfg.N_LABELS_PER_GEN
            )
    else:
        system_prompt = system_prompt_builder.build_system_prompt(
                prompt_path=cfg.PROMPT_PATH,
                language=cfg.LANGUAGE,
                nb_labels=cfg.N_LABELS_PER_GEN
            )

    valid_items = []

    # User prompt
    for i, (new_code, fewshot) in enumerate(zip(new_code_list, codes_fewshot)):
        try:

            # Get code details from Neo4j
            code_details = code_specifier.get_code_information(
                graph=NOTICE_GRAPH,
                code=new_code
                )

            # First for the exhaustivity part
            if cfg.EXHAUSTIVE_SAMPLING and i <= N_EXHAUSTIVE:
                user_prompts = exhaustive_user_prompt_builder.build_user_prompts(
                    code_details=code_details,
                    language=cfg.LANGUAGE,
                    nb_labels=cfg.N_LABELS_PER_GEN,
                    includes_divider=cfg.INCLUDES_DIVIDER,
                    examples_divider=cfg.EXAMPLES_DIVIDER,
                    excludes_divider=cfg.EXCLUDE_DIVIDER,
                    n_spec=cfg.N_SPEC_PER_PROMPT
                )

                for user_prompt in user_prompts:
                    if cfg.USE_FEWSHOT:
                        user_prompt += fewshot_prompt_builder.add_fewshot_user_prompt(
                            fewshot=fewshot,
                            language=cfg.LANGUAGE
                        )

                    valid_items.append({
                        "code": new_code,
                        "name": code_details["name"],
                        "prompt": user_prompt
                    })

            # For the random part
            else:
                user_prompt = random_user_prompt_builder.build_user_prompt(
                    code_details=code_details,
                    language=cfg.LANGUAGE,
                    nb_labels=cfg.N_LABELS_PER_GEN,
                    includes_divider=cfg.INCLUDES_DIVIDER,
                    examples_divider=cfg.EXAMPLES_DIVIDER,
                    excludes_divider=cfg.EXCLUDE_DIVIDER,
                    random_includes_geom_prob=cfg.RANDOM_INCLUDES_GEOM_PROB,
                    random_includes_min=cfg.RANDOM_INCLUDES_MIN,
                    random_includes_max=cfg.RANDOM_INCLUDES_MAX,
                    random_examples_geom_prob=cfg.RANDOM_EXAMPLES_GEOM_PROB,
                    random_examples_min=cfg.RANDOM_EXAMPLES_MIN,
                    random_examples_max=cfg.RANDOM_EXAMPLES_MAX
                    )

                if cfg.USE_FEWSHOT:
                    user_prompt += fewshot_prompt_builder.add_fewshot_user_prompt(
                        fewshot=fewshot,
                        language=cfg.LANGUAGE
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

    # ======================== LABEL GENERATION ==========================
    root_logger.info(f"Generating {len(valid_items)*cfg.N_LABELS_PER_GEN} labels...")

    results_buffer = []

    for i in range(0, len(valid_items), cfg.GENERATION_BATCH_SIZE):
        root_logger.info(f"Processing batch {i//cfg.GENERATION_BATCH_SIZE}...")

        batch = valid_items[i:i + cfg.GENERATION_BATCH_SIZE]
        prompts = [item["prompt"] for item in batch]

        try:
            generations = asyncio.run(
                label_generator.ask_model_multiple(
                    system_prompt=system_prompt,
                    user_prompts=prompts,
                    llm_client=LLM_CLIENT,
                    model=cfg.MODEL,
                    temperature=cfg.TEMPERATURE,
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
            root_logger.warning(f"Batch generation failed, retrying... Details: {e}")
            root_logger.info(traceback.format_exc())
            try:
                generations = asyncio.run(
                    label_generator.ask_model_multiple(
                        system_prompt=system_prompt,
                        user_prompts=prompts,
                        llm_client=LLM_CLIENT,
                        model=cfg.MODEL,
                        temperature=cfg.TEMPERATURE,
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

                root_logger.info("The new trial was succesful.")

            except Exception as ex:
                root_logger.warning(f"Batch generation failed, skipping... Details: {ex}")
                root_logger.info(traceback.format_exc())

        # ======================== INTERMEDIATE SAVE ==========================
        if cfg.OUTPUT_FORMAT == ".parquet" and len(results_buffer) >= cfg.SAVE_BATCH_SIZE:
            root_logger.info("Saving intermediate results...")

            try:
                codes = [r["code"] for r in results_buffer]
                names = [r["name"] for r in results_buffer]
                labels = [r["labels"] for r in results_buffer]

                label_exportation.export_to_parquet(
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

    # ======================== FINAL SAVE ==========================

    if cfg.OUTPUT_FORMAT == ".txt":
        root_logger.info("Saving results to txt...")

        codes = [r["code"] for r in results_buffer]
        names = [r["name"] for r in results_buffer]
        labels = [r["labels"] for r in results_buffer]

        label_exportation.export_to_txt(
            codes=codes,
            names=names,
            labels=labels,
            file_path=FINAL_PATH,
            generation_time=end-start
            )

    if cfg.OUTPUT_FORMAT == ".parquet" and results_buffer:
        root_logger.info("Saving final remaining results...")

        try:
            codes = [r["code"] for r in results_buffer]
            names = [r["name"] for r in results_buffer]
            labels = [r["labels"] for r in results_buffer]

            label_exportation.export_to_parquet(
                codes=codes,
                names=names,
                labels=labels,
                file_path=FINAL_PATH,
                fs=FS
            )

        except Exception as e:
            root_logger.warning(f"Buffer exportation failed, dropped... Details: {e}")
            root_logger.info(traceback.format_exc())
