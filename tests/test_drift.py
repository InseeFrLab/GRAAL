import numpy as np

from src.evaluation.drift import (
    calibrate_threshold,
    drift_report,
    ks_drift,
    psi,
    psi_categorical,
    rolling_drift_reports,
    wasserstein_drift,
)


def _confidence_scores(n, low, high, seed):
    rng = np.random.default_rng(seed)
    return rng.uniform(low, high, size=n)


def test_wasserstein_zero_for_identical_distributions():
    reference = _confidence_scores(500, 0.6, 1.0, seed=1)
    current = _confidence_scores(500, 0.6, 1.0, seed=2)
    # Same distribution, different draws: should be small, not exactly 0.
    assert wasserstein_drift(reference, reference) == 0.0
    assert wasserstein_drift(reference, current) < 0.05


def test_wasserstein_detects_shift():
    reference = _confidence_scores(500, 0.6, 1.0, seed=1)
    shifted = _confidence_scores(500, 0.1, 0.5, seed=2)
    assert wasserstein_drift(reference, shifted) > 0.3


def test_ks_drift_no_shift_high_pvalue():
    reference = _confidence_scores(500, 0.6, 1.0, seed=1)
    current = _confidence_scores(500, 0.6, 1.0, seed=2)
    result = ks_drift(reference, current)
    assert result["p_value"] > 0.05


def test_ks_drift_detects_shift():
    reference = _confidence_scores(500, 0.6, 1.0, seed=1)
    shifted = _confidence_scores(500, 0.1, 0.5, seed=2)
    result = ks_drift(reference, shifted)
    assert result["p_value"] < 0.01


def test_psi_no_shift_is_small():
    reference = _confidence_scores(1000, 0.6, 1.0, seed=1)
    current = _confidence_scores(1000, 0.6, 1.0, seed=2)
    assert psi(reference, current) < 0.1  # seuil usuel de stabilité


def test_psi_detects_shift():
    reference = _confidence_scores(1000, 0.6, 1.0, seed=1)
    shifted = _confidence_scores(1000, 0.1, 0.5, seed=2)
    assert psi(reference, shifted) > 0.25  # seuil usuel de dérive significative


def test_psi_categorical_identical_is_zero():
    labels = ["10.71C"] * 50 + ["62.01Z"] * 30 + ["43.99C"] * 20
    assert psi_categorical(labels, labels) == 0.0


def test_psi_categorical_detects_shift():
    reference = ["10.71C"] * 50 + ["62.01Z"] * 30 + ["43.99C"] * 20
    # Same categories, frequencies reversed
    shifted = ["10.71C"] * 20 + ["62.01Z"] * 30 + ["43.99C"] * 50
    assert psi_categorical(reference, shifted) > 0.25


def test_psi_categorical_handles_unseen_category():
    reference = ["10.71C"] * 50 + ["62.01Z"] * 50
    current = ["10.71C"] * 50 + ["01.11Z"] * 50  # 62.01Z disappeared, 01.11Z appeared
    assert psi_categorical(reference, current) > 0


def test_calibrate_threshold_is_positive_and_reproducible():
    reference = _confidence_scores(500, 0.6, 1.0, seed=1)
    threshold_a = calibrate_threshold(reference, wasserstein_drift, seed=42, n_resamples=100)
    threshold_b = calibrate_threshold(reference, wasserstein_drift, seed=42, n_resamples=100)
    assert threshold_a == threshold_b
    assert threshold_a > 0


def test_drift_report_no_drift():
    reference = _confidence_scores(500, 0.6, 1.0, seed=1)
    current = _confidence_scores(500, 0.6, 1.0, seed=2)
    report = drift_report(reference, current, n_resamples=100)
    assert report["any_drift"] is False
    assert report["ks"]["is_drift"] is False


def test_drift_report_flags_injected_drift():
    reference = _confidence_scores(500, 0.6, 1.0, seed=1)
    shifted = _confidence_scores(500, 0.1, 0.5, seed=2)
    report = drift_report(reference, shifted, n_resamples=100)
    assert report["any_drift"] is True
    assert report["wasserstein"]["is_drift"] is True
    assert report["ks"]["is_drift"] is True
    assert report["psi"]["is_drift"] is True


def test_rolling_drift_reports_tracks_progressive_shift():
    reference = _confidence_scores(500, 0.6, 1.0, seed=1)
    windows = [
        _confidence_scores(200, 0.6, 1.0, seed=10),  # pas de dérive
        _confidence_scores(200, 0.6, 1.0, seed=11),  # pas de dérive
        _confidence_scores(200, 0.1, 0.5, seed=12),  # dérive injectée
    ]
    reports = rolling_drift_reports(reference, windows, n_resamples=100)
    assert len(reports) == 3
    assert reports[0]["any_drift"] is False
    assert reports[1]["any_drift"] is False
    assert reports[2]["any_drift"] is True
