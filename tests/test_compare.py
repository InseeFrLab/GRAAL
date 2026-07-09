import polars as pl

from src.evaluation.compare import compare, mcnemar_test, paired_bootstrap_diff


def _make_df(y_true, y_pred, strata):
    return pl.DataFrame({"apet2025": y_true, "prediction": y_pred, "eval_stratum": strata})


def test_mcnemar_no_difference_when_predictions_identical():
    y_true = ["10.71C", "62.01Z", "01.11Z", "43.99C"] * 5
    y_pred = ["10.71C", "62.02A", "01.11Z", "99.99Z"] * 5
    p_value = mcnemar_test(y_true, y_pred, y_pred)
    assert p_value == 1.0


def test_mcnemar_detects_clear_difference():
    # Method A always wrong, method B always right on 20 discordant rows.
    y_true = [f"{i:02d}.00Z" for i in range(20)]
    y_pred_a = ["99.99Z"] * 20
    y_pred_b = list(y_true)
    p_value = mcnemar_test(y_true, y_pred_a, y_pred_b)
    assert p_value < 0.05


def test_paired_bootstrap_diff_matches_manual_point_estimate():
    y_true = ["10.71C", "10.71C", "62.01Z", "62.01Z"]
    y_pred_a = ["10.71C", "99.99Z", "62.01Z", "99.99Z"]  # 2/4 correct
    y_pred_b = ["10.71C", "10.71C", "62.01Z", "62.01Z"]  # 4/4 correct
    strata = ["A", "A", "B", "B"]
    point_diff, (lo, hi) = paired_bootstrap_diff(
        y_true, y_pred_a, y_pred_b, strata, n_resamples=300, seed=1
    )
    assert point_diff == 4 / 4 - 2 / 4
    assert lo <= point_diff <= hi


def test_compare_flags_significant_difference():
    n = 20
    y_true = [f"{i:02d}.00Z" for i in range(n)]
    y_pred_a = ["99.99Z"] * n  # always wrong
    y_pred_b = list(y_true)  # always right
    strata = ["S"] * n
    df_a = _make_df(y_true, y_pred_a, strata)
    df_b = _make_df(y_true, y_pred_b, strata)

    report = compare(df_a, df_b, n_resamples=300, seed=1)
    assert report["n"] == n
    assert report["accuracy_a"] == 0.0
    assert report["accuracy_b"] == 1.0
    assert report["diff"] == 1.0
    assert report["diff_ci"][0] > 0
    assert report["mcnemar_p_value"] < 0.05
    assert report["significant"] is True


def test_compare_no_significant_difference_when_predictions_identical():
    n = 20
    y_true = [f"{i:02d}.00Z" for i in range(n)]
    strata = ["S"] * n
    df_a = _make_df(y_true, y_true, strata)
    df_b = _make_df(y_true, y_true, strata)

    report = compare(df_a, df_b, n_resamples=300, seed=1)
    assert report["diff"] == 0.0
    assert report["significant"] is False


def test_compare_raises_on_misaligned_ground_truth():
    df_a = _make_df(["10.71C", "62.01Z"], ["10.71C", "62.01Z"], ["A", "B"])
    df_b = _make_df(["10.71C", "01.11Z"], ["10.71C", "01.11Z"], ["A", "B"])
    try:
        compare(df_a, df_b)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")
