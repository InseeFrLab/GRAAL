from src.evaluation.bootstrap import bootstrap_ci
from src.evaluation.metrics import accuracy_at_depth


def _accuracy_metric(y_true, y_pred, weights):
    return accuracy_at_depth(y_true, y_pred, weights=weights)


def _synthetic_dataset():
    # Two strata of 4 rows each: stratum "A" is 3/4 correct, stratum "B" is 1/4 correct.
    y_true = ["10.71C", "10.71C", "10.71C", "10.71C", "62.01Z", "62.01Z", "62.01Z", "62.01Z"]
    y_pred = ["10.71C", "10.71C", "10.71C", "99.99Z", "62.01Z", "99.99Z", "99.99Z", "99.99Z"]
    strata = ["A", "A", "A", "A", "B", "B", "B", "B"]
    return y_true, y_pred, strata


def test_point_estimate_matches_accuracy_at_depth():
    y_true, y_pred, strata = _synthetic_dataset()
    point, _ = bootstrap_ci(y_true, y_pred, strata, _accuracy_metric, n_resamples=200)
    assert point == accuracy_at_depth(y_true, y_pred)


def test_ci_bounds_are_in_unit_interval_and_contain_point_estimate():
    y_true, y_pred, strata = _synthetic_dataset()
    point, (lo, hi) = bootstrap_ci(y_true, y_pred, strata, _accuracy_metric, n_resamples=500)
    assert 0.0 <= lo <= hi <= 1.0
    # The point estimate need not sit exactly inside a percentile CI in general,
    # but for a symmetric, well-behaved statistic like accuracy here it should.
    assert lo <= point <= hi


def test_reproducible_for_fixed_seed():
    y_true, y_pred, strata = _synthetic_dataset()
    result_a = bootstrap_ci(y_true, y_pred, strata, _accuracy_metric, n_resamples=300, seed=7)
    result_b = bootstrap_ci(y_true, y_pred, strata, _accuracy_metric, n_resamples=300, seed=7)
    assert result_a == result_b


def test_resampling_never_mixes_rows_across_strata():
    # Unequal stratum sizes (4 vs 6) with per-row unique, stratum-tagged codes:
    # a resample that leaked rows across strata would break the exact 4/6 count.
    y_true = [f"A{i}" for i in range(4)] + [f"B{i}" for i in range(6)]
    y_pred = list(y_true)
    strata = ["A"] * 4 + ["B"] * 6

    def recording_metric(t, p, w):
        n_a = sum(1 for code in t if code.startswith("A"))
        n_b = sum(1 for code in t if code.startswith("B"))
        assert (n_a, n_b) == (4, 6)
        return accuracy_at_depth(t, p, weights=w)

    bootstrap_ci(y_true, y_pred, strata, recording_metric, n_resamples=50, seed=1)


def test_weights_are_resampled_alongside_rows():
    y_true, y_pred, strata = _synthetic_dataset()
    weights = [1, 1, 1, 1, 5, 5, 5, 5]
    point_unweighted, _ = bootstrap_ci(y_true, y_pred, strata, _accuracy_metric, n_resamples=200)
    point_weighted, _ = bootstrap_ci(
        y_true, y_pred, strata, _accuracy_metric, n_resamples=200, weights=weights
    )
    # Stratum B (weight 5, only 1/4 correct) dominates the weighted estimate,
    # so the weighted accuracy should be markedly lower than the unweighted one.
    assert point_weighted < point_unweighted


def test_raises_on_length_mismatch():
    try:
        bootstrap_ci(["a", "b"], ["a"], ["s", "s"], _accuracy_metric)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")
