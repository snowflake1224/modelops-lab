"""Tests for drift detection."""

import numpy as np

from packages.drift.detector import detect_feature_drift, population_stability_index


def test_psi_no_drift():
    rng = np.random.default_rng(42)
    ref = rng.normal(0, 1, 1000)
    cur = rng.normal(0, 1, 1000)
    psi = population_stability_index(ref, cur)
    assert psi < 0.2


def test_psi_with_drift():
    rng = np.random.default_rng(42)
    ref = rng.normal(0, 1, 1000)
    cur = rng.normal(2, 1, 1000)
    psi = population_stability_index(ref, cur)
    assert psi > 0.1


def test_detect_feature_drift():
    rng = np.random.default_rng(42)
    ref = rng.normal(0, 1, 500)
    cur = rng.normal(1, 1, 500)
    result = detect_feature_drift(ref, cur, "feature_1", psi_threshold=0.1, ks_threshold=0.05)
    assert result.drift_detected
