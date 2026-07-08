import polars as pl

from src.evaluation.build_eval_set import stratified_sample


def test_stratum_smaller_than_cap_is_kept_whole_with_weight_one():
    # Stratum "1071C" has only 2 rows, well under n_per_stratum=10.
    df = pl.DataFrame({"code": ["10.71C", "10.71C", "62.01Z", "62.01Z", "62.01Z"]})
    sampled = stratified_sample(df, code_column="code", n_per_stratum=10, stratum_depth=5)

    small_stratum = sampled.filter(pl.col("eval_stratum") == "1071C")
    assert len(small_stratum) == 2
    assert small_stratum["ipw_weight"].to_list() == [1.0, 1.0]


def test_capped_stratum_gets_inverse_probability_weight():
    # Stratum "6201Z" has 8 rows, capped to 4: weight should be 8/4 = 2.
    df = pl.DataFrame({"code": ["62.01Z"] * 8 + ["10.71C"] * 2})
    sampled = stratified_sample(df, code_column="code", n_per_stratum=4, stratum_depth=5)

    capped_stratum = sampled.filter(pl.col("eval_stratum") == "6201Z")
    assert len(capped_stratum) == 4
    assert capped_stratum["ipw_weight"].to_list() == [2.0, 2.0, 2.0, 2.0]

    whole_stratum = sampled.filter(pl.col("eval_stratum") == "1071C")
    assert len(whole_stratum) == 2
    assert whole_stratum["ipw_weight"].to_list() == [1.0, 1.0]


def test_helper_columns_are_not_leaked():
    df = pl.DataFrame({"code": ["10.71C", "62.01Z"]})
    sampled = stratified_sample(df, code_column="code", n_per_stratum=10, stratum_depth=5)
    assert "_norm_code" not in sampled.columns
    assert "_population_count" not in sampled.columns
    assert "_sampled_count" not in sampled.columns
    assert "eval_stratum" in sampled.columns
    assert "ipw_weight" in sampled.columns


def test_reproducible_for_fixed_seed():
    df = pl.DataFrame({"code": ["62.01Z"] * 10, "id": list(range(10))})
    sampled_a = stratified_sample(df, code_column="code", n_per_stratum=4, stratum_depth=5, seed=1)
    sampled_b = stratified_sample(df, code_column="code", n_per_stratum=4, stratum_depth=5, seed=1)
    assert sampled_a["id"].to_list() == sampled_b["id"].to_list()
