# For code sampling
POPULATION_PATH = "projet-ape/data/08112022_27102024/naf2025/split/df_train.parquet"
CODE_COLUMN = "nace2025"

# For prompt creation
PROMPT_PATH = "src/agents/NaiveCode2Text/prompts/"

# To retrieve specifications of every code correctly:
INCLUDES_DIVIDER = "\n-"
EXAMPLES_DIVIDER = "\n"
EXCLUDE_DIVIDER = "\n"

# Randomization for specifications
RANDOM_SPEC_SAMPLING = False
RANDOM_INCLUDES_GEOM_PROB = 0.7
RANDOM_INCLUDES_MIN = 1
RANDOM_INCLUDES_MAX = None      # None = up to the max number of includes
RANDOM_EXAMPLES_GEOM_PROB = 0.5
RANDOM_EXAMPLES_MIN = 1
RANDOM_EXAMPLES_MAX = None      # None = up to the max number of examples per include

# Exportation
OUTPUT_PATH = "src/agents/NaiveCode2Text/sample_results/"
OUTPUT_FORMAT = ".txt"          # .txt or .parquet
BATCH_SIZE = 1000               # If choosing .parquet output format

# LLM Hyperparameters
MODEL = "gpt-oss:20b"
TEMPERATURE = 1.4
LANGUAGE = "English"

# Generation specifications
N_CODES = 5                     # Number of codes to sample
NB_LABELS = 10                  # Number of labels to generate per code
