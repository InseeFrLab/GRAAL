# For code sampling
POPULATION_PATH = "projet-ape/data/08112022_27102024/naf2025/split/df_train.parquet"
CODE_COLUMN = "nace2025"
CONVERT_NAF_TO_NACE = False
CONVERT_TO_PROPER_NAF = True
EXHAUSTIVE_SAMPLING = True
N_SAMPLES_PER_CODE = 5      # If exhaustive sampling

# For prompt creation
PROMPT_PATH = "src/agents/NaiveCode2Text/prompts/"

# To retrieve specifications of every code correctly:
INCLUDES_DIVIDER = "\n-"
EXAMPLES_DIVIDER = "\n"
EXCLUDE_DIVIDER = "\n"

# Randomization for specifications
RANDOM_SPEC_SAMPLING = True
RANDOM_INCLUDES_GEOM_PROB = 0.3
RANDOM_INCLUDES_MIN = 1
RANDOM_INCLUDES_MAX = None      # None = up to the max number of includes
RANDOM_EXAMPLES_GEOM_PROB = 0.2
RANDOM_EXAMPLES_MIN = 1
RANDOM_EXAMPLES_MAX = None      # None = up to the max number of examples per include

# Exportation
OUTPUT_PATH = "projet-ape/synthetic_data_test/naive/NAF2025_FR/"
OUTPUT_FORMAT = ".parquet"          # .txt or .parquet
SAVE_BATCH_SIZE = 1000              # If choosing .parquet output format
MODEL_NAME = "gemma-3-27b-it"       # For file name

# LLM Hyperparameters
MODEL = None
TEMPERATURE = 1.4
LANGUAGE = "French"
GENERATION_BATCH_SIZE = 20

# Generation specifications
N_CODES = 20000                 # Number of codes to sample, multiple of GENERATION_BATCH_SIZE
NB_LABELS = 10                  # Number of labels to generate per code

# Few-shot specifications:
USE_FEWSHOT = True
LABEL_COLUMN = "libelle"
N_FEWSHOT = 6
