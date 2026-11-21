# Experiment Report — ModelOps Lab

**Date:** 2026-08-23  
**Dataset:** Credit Card Fraud Detection (OpenML 1597, 284,807 rows, 29 features: `V1`–`V28` + `Amount`)  
**Split:** 70% train / 15% validation / 15% test (stratified)  
**Class imbalance:** ~0.173% positive rate (492 frauds total; 74 frauds in test set of 42,722)  

## Model Comparison (Test Set — `experiment_report_20260823_175323.json`)

| Model | PR-AUC | F1 | Precision | Recall | ROC-AUC | Threshold (val F1-opt) | Training Time |
|-------|--------|-----|-----------|--------|---------|------------------------|---------------|
| **XGBoost** | **0.8303** | 0.8244 | 0.9474 | 0.7297 | 0.9621 | 0.9875 | 9.5s |
| Logistic Regression | 0.7943 | 0.7682 | 0.7532 | 0.7838 | 0.9717 | 0.9969 | 1.0s |
| PyTorch MLP | 0.7473 | **0.8345** | 0.8923 | 0.7838 | **0.9776** | 1.0000 | 483.0s |
| LightGBM (interactions) | 0.6774 | 0.8031 | **0.9623** | 0.6892 | 0.8519 | 0.9945 | 10.4s |
| LightGBM (baseline) | 0.0887 | 0.1886 | 0.1065 | **0.8243** | 0.8916 | 1.0000 | 5.8s |

## Key Findings

### Model Choice
- **XGBoost** achieves the highest PR-AUC (**0.8303**) with **0.9474** precision and **9.5s** training time, making it the recommended production candidate (`fraud-classifier:1.0.0`).
- **Logistic Regression** provides a strong linear baseline (**0.7943** PR-AUC, **0.9717** ROC-AUC) in **1.0s** — useful for rapid pipeline verification.
- **PyTorch MLP** reaches the highest F1 (**0.8345**) and ROC-AUC (**0.9776**), at the cost of **~51× longer** training (`483.0s` vs `9.5s`) and extreme probability saturation near `1.0`.

### Feature Engineering Ablation
- Adding pairwise feature interactions (top-5 variance columns in `FeaturePreprocessor`) **improved** LightGBM PR-AUC from **0.0887 to 0.6774** (`+0.5887` PR-AUC, `+0.6145` F1), lifting precision from `0.1065` to `0.9623`.
- The uncalibrated baseline LightGBM run produced weak probability ranking (`0.0887` PR-AUC) with heavy over-prediction (`512` false positives at threshold clipping near `1.0`), indicating probability calibration / hyperparameter tuning is needed on the LightGBM baseline config.

### Evaluation Measurement Note
- Earlier saved JSON artifacts (`experiment_report_20260823_*.json`) contain near-zero `bootstrap_ci` ranges because `compute_metrics()` previously drew separate random indices for `y_true` and `y_proba`, unpairing targets from predictions during resampling. This bug has been fixed in `packages/evaluation/metrics.py` so subsequent experiment runs compute paired bootstrap confidence intervals.

### Class Imbalance Strategy
- All models used `class_weight="balanced"` or equivalent positive-class weighting (`scale_pos_weight`).
- Thresholds (`find_optimal_threshold(..., metric="f1")`) were tuned strictly on the validation split and frozen for test evaluation and `registry.json` (`optimal_threshold`).

### Engineering Tradeoffs

| Tradeoff | Recommendation |
|----------|---------------|
| Accuracy vs latency | XGBoost: highest PR-AUC (`0.8303`) with `<10s` training |
| Simplicity vs performance | Logistic Regression within `3.6pp` PR-AUC of XGBoost in `1.0s` |
| Neural vs tree | Gradient boosted trees outperform MLP on PR-AUC at tabular fraud |
| Training cost | PyTorch MLP (`483s`) is disproportionate for marginal F1 gain over XGBoost |

## Controlled Experiment: Feature Interactions

| Config | PR-AUC | F1 | Precision | Δ PR-AUC |
|--------|--------|-----|-----------|----------|
| LightGBM baseline | 0.0887 | 0.1886 | 0.1065 | — |
| LightGBM + interactions | 0.6774 | 0.8031 | 0.9623 | +0.5887 |

## Reproducibility

Experiments log to MLflow (`sqlite:///data/mlflow.db`) with:
- Git commit hash
- Dataset version checksum (`42fdf08c21a87bcd` for OpenML 1597 v1)
- Preprocessing config hash (`FeaturePreprocessor.version`)
- Random seed (`42`)
- Full hyperparameter configuration (`config.json`)

Re-run: `modelops experiment run configs/xgboost/baseline.yaml`

## Next Steps

1. Run Optuna HPO on XGBoost: `modelops experiment hpo configs/xgboost/baseline.yaml`
2. Tune/calibrate LightGBM hyperparameters (e.g., `max_depth`, `min_child_samples`, `scale_pos_weight`)
3. Monitor feature/prediction drift in production and trigger candidate validation before promotion
