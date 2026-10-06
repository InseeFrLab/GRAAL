import math

from src.evaluation.metrics import (
    accuracy_at_depth,
    evaluate,
    failure_rate,
    low_confidence_rate,
    normalize_code,
    roc_auc,
    verifier_metrics,
)


def test_normalize_code():
    assert normalize_code("10.71C") == "1071C"
    assert normalize_code(" 10 71 c ") == "1071C"
    assert normalize_code(None) is None
    assert normalize_code("") is None


def test_leaf_accuracy():
    y_true = ["10.71C", "62.01Z", "01.11Z"]
    y_pred = ["1071C", "62.02A", "01.11Z"]
    assert accuracy_at_depth(y_true, y_pred) == 2 / 3


def test_depth_accuracy():
    y_true = ["10.71C", "62.01Z"]
    y_pred = ["10.72B", "43.99C"]
    # Division (depth 2): 10 == 10 mais 62 != 43
    assert accuracy_at_depth(y_true, y_pred, depth=2) == 1 / 2
    # Classe (depth 4): 1071 != 1072
    assert accuracy_at_depth(y_true, y_pred, depth=4) == 0.0


def test_missing_prediction_counts_as_error():
    assert accuracy_at_depth(["10.71C"], [None]) == 0.0
    assert accuracy_at_depth(["10.71C"], [None], depth=2) == 0.0


def test_missing_ground_truth_is_skipped():
    assert accuracy_at_depth([None, "10.71C"], ["10.71C", "10.71C"]) == 1.0


def test_length_mismatch_raises():
    try:
        accuracy_at_depth(["a"], [])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


def test_failure_rate():
    assert failure_rate(["10.71C", None, "", "62.01Z"]) == 2 / 4
    assert math.isnan(failure_rate([]))


def test_evaluate_report():
    y_true = ["10.71C", "62.01Z", "01.11Z", "43.99C"]
    y_pred = ["10.71C", "62.02A", None, "43.99C"]
    report = evaluate(y_true, y_pred)
    assert report["n"] == 4
    assert report["leaf_accuracy"] == 2 / 4
    assert report["failure_rate"] == 1 / 4
    assert report["accuracy_depth_2"] == 3 / 4
    assert set(report) == {
        "n",
        "leaf_accuracy",
        "failure_rate",
        "accuracy_depth_2",
        "accuracy_depth_3",
        "accuracy_depth_4",
    }


def test_weighted_accuracy_matches_manual_computation():
    y_true = ["10.71C", "62.01Z", "01.11Z"]
    y_pred = ["1071C", "62.02A", "01.11Z"]  # correct, wrong, correct
    weights = [1, 2, 1]
    # correct weight = 1 (row 0) + 1 (row 2) = 2 ; total weight = 1 + 2 + 1 = 4
    assert accuracy_at_depth(y_true, y_pred, weights=weights) == 2 / 4
    # unweighted accuracy on the same data is unaffected (2/3, not 2/4)
    assert accuracy_at_depth(y_true, y_pred) == 2 / 3


def test_weighted_accuracy_length_mismatch_raises():
    try:
        accuracy_at_depth(["a"], ["a"], weights=[1, 2])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


def test_evaluate_includes_weighted_keys_when_weights_given():
    y_true = ["10.71C", "62.01Z", "01.11Z"]
    y_pred = ["1071C", "62.02A", "01.11Z"]
    weights = [1, 2, 1]
    report = evaluate(y_true, y_pred, weights=weights)
    assert report["leaf_accuracy"] == 2 / 3
    assert report["leaf_accuracy_weighted"] == 2 / 4
    assert "accuracy_depth_2_weighted" in report


def test_evaluate_omits_weighted_and_confidence_keys_by_default():
    report = evaluate(["10.71C"], ["10.71C"])
    assert "leaf_accuracy_weighted" not in report
    assert "low_confidence_rate" not in report


def test_low_confidence_rate():
    assert low_confidence_rate([0.0, 0.9, None, 0.0]) == 2 / 4
    assert math.isnan(low_confidence_rate([]))


def test_evaluate_includes_low_confidence_rate_when_given():
    y_true = ["10.71C", "62.01Z"]
    y_pred = ["10.71C", "62.01Z"]
    confidences = [0.0, 0.9]
    report = evaluate(y_true, y_pred, confidences=confidences)
    assert report["low_confidence_rate"] == 0.5


def test_roc_auc_ranks_and_ties():
    assert roc_auc([True, False], [0.9, 0.1]) == 1.0
    assert roc_auc([True, False], [0.1, 0.9]) == 0.0
    assert roc_auc([True, False], [0.5, 0.5]) == 0.5
    # Le score None est ignoré, pas compté comme 0.
    assert roc_auc([True, False, True], [0.9, 0.1, None]) == 1.0
    assert math.isnan(roc_auc([True, True], [0.2, 0.3]))


def test_verifier_metrics_counts_errors_as_the_positive_class():
    # Deux codes faux (un rejeté, un accepté), deux bons (un rejeté à tort).
    report = verifier_metrics(
        expected_match=[False, False, True, True],
        verdicts=[False, True, False, True],
        p_match=[0.1, 0.8, 0.4, 0.9],
        error_prevalence=0.2,
    )
    assert report["n_errors"] == 2 and report["n_correct"] == 2
    assert report["error_recall"] == 0.5
    assert report["false_rejection_rate"] == 0.5
    assert report["rejection_precision"] == 0.5
    assert report["balanced_accuracy"] == 0.5
    # Erreurs : scores 1-p = 0.9 et 0.2 ; bons : 0.6 et 0.1 -> 3 paires gagnées sur 4.
    assert report["roc_auc"] == 0.75
    # À 20 % d'erreurs, un rejet sur cinq seulement porte sur une vraie erreur.
    assert math.isclose(report["rejection_precision_at_prevalence"], 0.2)
