# %%
import polars as pl

# %%
eval_dataset = pl.read_parquet("../data/eval/eval_set.parquet")
eval_dataset.head()

# %%
eval = pl.read_parquet("dataset_eval.parquet")
eval.head()
# %%
