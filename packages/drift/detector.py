"""Drift detection: PSI and KS test."""

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class DriftResult:
    feature: str
    psi: float
    ks_statistic: float
    ks_pvalue: float
    psi_drift: bool
    ks_drift: bool
    drift_detected: bool


def population_stability_index(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
    """Calculate PSI between reference and current distributions."""
    breakpoints = np.percentile(expected, np.linspace(0, 100, n_bins + 1))
    breakpoints = np.unique(breakpoints)
    if len(breakpoints) < 2:
        return 0.0

    expected_counts = np.histogram(expected, bins=breakpoints)[0].astype(float)
    actual_counts = np.histogram(actual, bins=breakpoints)[0].astype(float)

    expected_pct = expected_counts / max(expected_counts.sum(), 1)
    actual_pct = actual_counts / max(actual_counts.sum(), 1)

    # Avoid division by zero
    expected_pct = np.clip(expected_pct, 0.0001, None)
    actual_pct = np.clip(actual_pct, 0.0001, None)

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi)


def detect_feature_drift(
    reference: np.ndarray,
    current: np.ndarray,
    feature_name: str,
    psi_threshold: float = 0.2,
    ks_threshold: float = 0.05,
) -> DriftResult:
    psi = population_stability_index(reference, current)
    ks_stat, ks_pvalue = stats.ks_2samp(reference, current)
    psi_drift = psi > psi_threshold
    ks_drift = ks_pvalue < ks_threshold
    return DriftResult(
        feature=feature_name,
        psi=psi,
        ks_statistic=float(ks_stat),
        ks_pvalue=float(ks_pvalue),
        psi_drift=psi_drift,
        ks_drift=ks_drift,
        drift_detected=psi_drift or ks_drift,
    )


def detect_dataset_drift(
    reference: np.ndarray,
    current: np.ndarray,
    feature_names: list[str],
    psi_threshold: float = 0.2,
    ks_threshold: float = 0.05,
) -> list[DriftResult]:
    results = []
    for i, name in enumerate(feature_names):
        results.append(
            detect_feature_drift(
                reference[:, i], current[:, i], name, psi_threshold, ks_threshold
            )
        )
    return results
