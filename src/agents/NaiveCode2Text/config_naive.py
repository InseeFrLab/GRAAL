# Main specifications
EXHAUSTIVE_SAMPLING = True
USE_FEWSHOT = True
N_RANDOM_CODES = 0              # Number of codes to sample randomly
N_LABELS_PER_GEN = 10           # Number of labels in one LLM generation
N_FEWSHOT = 6

# Exportation
OUTPUT_PATH = "projet-ape/synthetic_data_test/naive/NAF2025_FR/"
OUTPUT_FORMAT = ".parquet"                      # .txt or .parquet
SAVE_BATCH_SIZE = 1000                          # If choosing .parquet output format
MODEL_NAME = "google/gemma-3-27b-it"            # For file name

# LLM Hyperparameters
MODEL = "google/gemma-3-27b-it"
TEMPERATURE = 1.4
LANGUAGE = "French"
GENERATION_BATCH_SIZE = 20

# For code sampling
POPULATION_PATH = "projet-ape/data/08112022_27102024/naf2025/split/df_train.parquet"
CODE_COLUMN = "nace2025"
CONVERT_NAF_TO_NACE = False
CONVERT_TO_PROPER_NAF = True

# To retrieve specifications of every code correctly:
INCLUDES_DIVIDER = "\n-"
EXAMPLES_DIVIDER = "\n"
EXCLUDE_DIVIDER = "\n"

# For prompt creation
PROMPT_PATH = "src/agents/NaiveCode2Text/prompt_creation/"

# For exhaustivity
MIN_PROMPTS_PER_CODE = 5
N_SPEC_PER_PROMPT = 5           # Number of specifications per prompt

# Randomization part
RANDOM_INCLUDES_GEOM_PROB = 0.3
RANDOM_INCLUDES_MIN = 1
RANDOM_INCLUDES_MAX = None      # None = up to the max number of includes
RANDOM_EXAMPLES_GEOM_PROB = 0.2
RANDOM_EXAMPLES_MIN = 1
RANDOM_EXAMPLES_MAX = None      # None = up to the max number of examples per include

# Few-shot specifications:
LABEL_COLUMN = "libelle"
