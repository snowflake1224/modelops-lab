"""Tests for evaluation metrics."""

import numpy as np

from packages.evaluation.metrics import compute_metrics, find_optimal_threshold


def test_compute_metrics():
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, 200)
    y_proba = rng.random(200)
    result = compute_metrics(y_true, y_proba, n_bootstrap=50)
    assert 0 <= result.precision <= 1
    assert 0 <= result.recall <= 1
    assert 0 <= result.roc_auc <= 1
    assert "precision" in result.bootstrap_ci


def test_find_optimal_threshold():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    threshold = find_optimal_threshold(y_true, y_proba)
    assert 0 < threshold < 1


def test_paired_bootstrap_ci_brackets_signal():
    # Separable imbalanced problem: paired resampling must preserve ranking signal
    y_true = np.array([0] * 180 + [1] * 20)
    y_proba = np.concatenate([np.linspace(0.01, 0.2, 180), np.linspace(0.75, 0.99, 20)])
    result = compute_metrics(y_true, y_proba, threshold=0.5, n_bootstrap=100, seed=7)
    lo, hi = result.bootstrap_ci["pr_auc"]
    assert lo > 0.7
    assert lo <= result.pr_auc <= hi + 1e-9
