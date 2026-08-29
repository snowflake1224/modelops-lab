"""Evaluation metrics and statistical analysis."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


@dataclass
class MetricResult:
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float
    confusion: list[list[int]]
    threshold: float
    calibration: dict[str, Any] = field(default_factory=dict)
    bootstrap_ci: dict[str, tuple[float, float]] = field(default_factory=dict)


def compute_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
    n_bootstrap: int = 200,
    seed: int = 42,
) -> MetricResult:
    y_pred = (y_proba >= threshold).astype(int)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)
    cm = confusion_matrix(y_true, y_pred).tolist()

    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10, strategy="quantile")
    calibration = {"prob_true": prob_true.tolist(), "prob_pred": prob_pred.tolist()}

    rng = np.random.default_rng(seed)
    bootstrap_ci: dict[str, tuple[float, float]] = {}
    for name, fn in [
        ("precision", lambda: precision_score(y_true[rng.integers(0, len(y_true), len(y_true))], (y_proba[rng.integers(0, len(y_true), len(y_true))] >= threshold).astype(int), zero_division=0)),
        ("recall", lambda: recall_score(y_true[rng.integers(0, len(y_true), len(y_true))], (y_proba[rng.integers(0, len(y_true), len(y_true))] >= threshold).astype(int), zero_division=0)),
        ("f1", lambda: f1_score(y_true[rng.integers(0, len(y_true), len(y_true))], (y_proba[rng.integers(0, len(y_true), len(y_true))] >= threshold).astype(int), zero_division=0)),
        ("pr_auc", lambda: average_precision_score(y_true[rng.integers(0, len(y_true), len(y_true))], y_proba[rng.integers(0, len(y_true), len(y_true))])),
    ]:
        samples = [fn() for _ in range(n_bootstrap)]
        bootstrap_ci[name] = (float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5)))

    return MetricResult(
        precision=precision,
        recall=recall,
        f1=f1,
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        confusion=cm,
        threshold=threshold,
        calibration=calibration,
        bootstrap_ci=bootstrap_ci,
    )


def find_optimal_threshold(y_true: np.ndarray, y_proba: np.ndarray, metric: str = "f1") -> float:
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    if metric == "f1":
        f1s = 2 * precisions * recalls / (precisions + recalls + 1e-8)
        idx = int(np.argmax(f1s))
        return float(thresholds[min(idx, len(thresholds) - 1)]) if len(thresholds) else 0.5
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    j_scores = tpr - fpr
    idx = int(np.argmax(j_scores))
    return float(thresholds[idx]) if len(thresholds) else 0.5


def metrics_to_dict(result: MetricResult) -> dict[str, Any]:
    return {
        "precision": result.precision,
        "recall": result.recall,
        "f1": result.f1,
        "roc_auc": result.roc_auc,
        "pr_auc": result.pr_auc,
        "threshold": result.threshold,
        "confusion_matrix": result.confusion,
        "bootstrap_ci": result.bootstrap_ci,
    }
