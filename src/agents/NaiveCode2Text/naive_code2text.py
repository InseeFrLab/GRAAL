import os
import logging
import time
import traceback
import asyncio

from dotenv import load_dotenv
import s3fs
from openai import AsyncOpenAI
from langchain_neo4j import Neo4jGraph
import hydra
from omegaconf import DictConfig

from src.agents.NaiveCode2Text.code_retrieval import code_sampler, code_specifier
from src.agents.NaiveCode2Text.data_preprocessing import NAF_preprocessing
from src.agents.NaiveCode2Text.fewshot import fewshot_prompt_builder, fewshot_sampler
from src.agents.NaiveCode2Text.label_generation import label_generator, label_exportation
from src.agents.NaiveCode2Text.prompt_creation import system_prompt_builder, \
    exhaustive_user_prompt_builder, random_user_prompt_builder


@hydra.main(
    version_base=None,
    config_path="src/agents/NaiveCode2Text/runtime_config",
    config_name="config"
    )
def main(cfg: DictConfig):

    # ======================== INIT ==========================
    # Logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    handler = logging.StreamHandler()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    handler.setFormatter(formatter)
    root_logger.addHandler(handler)

    # Environment
    load_dotenv(override=True)

    root_logger.info("Initialization...")

    # Clock for speed testing
    if cfg["export"]["output_format"] == ".txt":
        start = time.perf_counter()

    # Access configurations
    FS = s3fs.S3FileSystem(
        client_kwargs={'endpoint_url': "https://" + os.environ["AWS_S3_ENDPOINT"]},
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
    LabelGenerationModel = label_generator.build_label_generation_model(
        nb_labels=cfg["main"]["n_labels_per_gen"]
    )

    # Automatic name for output
    FINAL_PATH = label_exportation.create_file_name(
        output_path=cfg["export"]["output_path"],
        output_format=cfg["export"]["output_format"],
        temperature=cfg["llm"]["temperature"],
        language=cfg["main"]["language"],
        exhaustive_sampling=cfg["main"]["exhaustive_sampling"],
        use_fewshot=cfg["main"]["use_fewshot"],
        n_fewshot=cfg["fewshot"]["n_fewshot"],
        model_name=cfg["export"]["model_name"],
        model=cfg["llm"]["model"],
        )

    # ======================== SAMPLING CODES ==========================
    root_logger.info("Sampling from data...")

    code_list = []

    if cfg["main"]["exhaustive_sampling"]:
        code_list += code_specifier.get_all_codes(
                        graph=NOTICE_GRAPH,
                        n_samples_per_code=cfg["exhaustivity"]["min_prompts_per_code"]
                    )

        code_list = [NAF_preprocessing.to_bad_NAF(code) for code in code_list]

        N_EXHAUSTIVE = len(code_list)

    if cfg["main"]["n_random_codes"] > 0:
        code_list += code_sampler.sample_codes_lazy(
                        fs=FS,
                        population_path=cfg["sampling"]["population_path"],
                        code_column=cfg["sampling"]["code_column"],
                        n_codes=cfg["main"]["n_random_codes"]
                    )

    # NAF to NACE
    if cfg["naf"]["convert_naf_to_nace"]:
        root_logger.info("Transforming codes from NAF to NACE...")
        new_code_list = [NAF_preprocessing.NAF_to_NACE(code) for code in code_list]

    elif cfg["naf"]["convert_to_proper_naf"]:
        root_logger.info("Transforming codes to proper NAF...")
        new_code_list = [NAF_preprocessing.to_proper_NAF(code) for code in code_list]

    else:
        new_code_list = code_list

    # ======================== FEW-SHOT ==========================
    if cfg["main"]["use_fewshot"]:
        root_logger.info("Sampling examples for few-shot...")
        codes_fewshot = fewshot_sampler.sample_fewshot_lazy_multi(
            fs=FS,
            population_path=cfg["sampling"]["population_path"],
            code_column=cfg["sampling"]["code_column"],
            codes=code_list,
            label_column=cfg["fewshot"]["label_column"],
            n_fewshot=cfg["fewshot"]["n_fewshot"]
        )

    # ======================== CODE DETAILS ==========================
    root_logger.info("Getting details of every code...")
    unique_codes = list(set(new_code_list))

    code_details = code_specifier.get_code_list_information(
                graph=NOTICE_GRAPH,
                code_list=unique_codes
                )

    # ======================== PROMPT CREATION ==========================
    root_logger.info("Creating prompts...")

    # System prompt
    if cfg["main"]["use_fewshot"]:
        system_prompt = fewshot_prompt_builder.build_fewshot_system_prompt(
                prompt_path=cfg["prompt"]["prompt_path"],
                language=cfg["main"]["language"],
                nb_labels=cfg["main"]["n_labels_per_gen"]
            )
    else:
        system_prompt = system_prompt_builder.build_system_prompt(
                prompt_path=cfg["prompt"]["prompt_path"],
                language=cfg["main"]["language"],
                nb_labels=cfg["main"]["n_labels_per_gen"]
            )

    valid_items = []

    # User prompt
    for i, (new_code, fewshot) in enumerate(zip(new_code_list, codes_fewshot)):
        code_spec = code_details[new_code]

        try:

            # First for the exhaustivity part
            if cfg["main"]["exhaustive_sampling"] and i <= N_EXHAUSTIVE:
                user_prompts = exhaustive_user_prompt_builder.build_user_prompts(
                    code_details=code_spec,
                    language=cfg["main"]["language"],
                    nb_labels=cfg["main"]["n_labels_per_gen"],
                    includes_divider=cfg["spec"]["includes_divider"],
                    examples_divider=cfg["spec"]["examples_divider"],
                    excludes_divider=cfg["spec"]["excludes_divider"],
                    n_spec=cfg["exhaustivity"]["n_spec_per_prompt"]
                )

                for user_prompt in user_prompts:
                    if cfg["main"]["use_fewshot"]:
                        user_prompt += fewshot_prompt_builder.add_fewshot_user_prompt(
                            fewshot=fewshot,
                            language=cfg["main"]["language"]
                        )

                    valid_items.append({
                        "code": new_code,
                        "name": code_spec["name"],
                        "prompt": user_prompt
                    })

            # For the random part
            elif cfg["main"]["n_random_codes"] > 0:
                user_prompt = random_user_prompt_builder.build_user_prompt(
                    code_details=code_spec,
                    language=cfg["main"]["language"],
                    nb_labels=cfg["main"]["n_labels_per_gen"],
                    includes_divider=cfg["spec"]["includes_divider"],
                    examples_divider=cfg["spec"]["examples_divider"],
                    excludes_divider=cfg["spec"]["excludes_divider"],
                    random_includes_geom_prob=cfg["random"]["includes"]["geom_prob"],
                    random_includes_min=cfg["random"]["includes"]["min"],
                    random_includes_max=cfg["random"]["includes"]["max"],
                    random_examples_geom_prob=cfg["random"]["examples"]["geom_prob"],
                    random_examples_min=cfg["random"]["examples"]["min"],
                    random_examples_max=cfg["random"]["examples"]["max"]
                    )

                if cfg["main"]["use_fewshot"]:
                    user_prompt += fewshot_prompt_builder.add_fewshot_user_prompt(
                        fewshot=fewshot,
                        language=cfg["main"]["language"]
                    )

                valid_items.append({
                    "code": new_code,
                    "name": code_spec["name"],
                    "prompt": user_prompt
                })

        except Exception as e:
            root_logger.warning(f"Error preparing code {new_code}, skipping...\nDetails: {e}")
            root_logger.info(traceback.format_exc())
            continue

    # ======================== LABEL GENERATION ==========================
    root_logger.info(f"Generating {len(valid_items)*cfg["main"]["n_labels_per_gen"]} labels...")

    results_buffer = []
    n_batches = len(valid_items) // cfg["llm"]["generation_batch_size"]
    if len(valid_items) % cfg["llm"]["generation_batch_size"] > 0:
        n_batches += 1

    for i in range(0, len(valid_items), cfg["llm"]["generation_batch_size"]):
        root_logger.info(
            f"Processing batch {(i // cfg["llm"]["generation_batch_size"]) + 1}/{n_batches}..."
        )

        batch = valid_items[i:i + cfg["llm"]["generation_batch_size"]]
        prompts = [item["prompt"] for item in batch]

        try:
            generations = asyncio.run(
                label_generator.ask_model_multiple(
                    system_prompt=system_prompt,
                    user_prompts=prompts,
                    llm_client=LLM_CLIENT,
                    model=cfg["llm"]["model"],
                    temperature=cfg["llm"]["temperature"],
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
                        model=cfg["llm"]["model"],
                        temperature=cfg["llm"]["temperature"],
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
        if (cfg["export"]["output_format"] == ".parquet") and \
                (len(results_buffer) >= cfg["export"]["save_batch_size"]):
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

    if cfg["export"]["output_format"] == ".txt":
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

    if cfg["export"]["output_format"] == ".parquet" and results_buffer:
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


if __name__ == "__main__":
    main()
