# Engineering FAQ

Use this guide when walking through ModelOps Lab's ML lifecycle, leakage prevention, imbalance evaluation, and serving consistency guarantees.

---

## 1. 60-Second Elevator Pitch

> ModelOps Lab is an end-to-end MLOps platform for tabular fraud detection (284,807 transactions across 29 features, 0.173% fraud rate). It enforces leakage-free stratified splitting before preprocessing, tracks XGBoost, LightGBM, Logistic Regression, and PyTorch MLP runs plus Optuna HPO in MLflow, tunes thresholds on validation PR curves with paired bootstrap confidence intervals, gates model promotion (`development -> validation -> staging -> production`) in a versioned registry, serves both online FastAPI and batch predictions through a single shared `InferenceEngine`, and monitors production traffic with PSI/KS drift checks that trigger human-approved retraining candidates.

---

## 2. Core Technical Q&A

### Q1: Why optimize PR-AUC and validation-tuned thresholds instead of accuracy or default 0.50?
- At `0.173%` fraud prevalence, a trivial constant all-zero classifier scores `99.83%` accuracy and can still achieve high ROC-AUC when false-positive rate grows slowly over `42,648` negative samples.
- **PR-AUC** evaluates precision vs. recall directly on the minority class. Furthermore, training with `class_weight="balanced"` shifts raw predicted probabilities upward toward `0.98–0.99`, so calling `predict()` with a `0.50` cutoff floods alerts; `find_optimal_threshold(y_val, y_proba_val, metric="f1")` picks the operating threshold on the **validation** split and freezes it into `registry.json` (`0.9875` for production XGBoost `1.0.0`).

### Q2: How do you guarantee zero training/serving skew and zero data leakage?
- `DataPipeline.run()` performs stratified `70/15/15` splits **before** touching `FeaturePreprocessor`.
- `FeaturePreprocessor.fit()` computes quantile clipping bounds, top-5 variance interaction pairs, and `RobustScaler` medians/IQRs strictly on `x_train`, storing its configuration hash (`PreprocessorConfig`) and serializing `preprocessor.joblib` beside the trained model.
- At serving time, both `apps/inference/main.py` (`FastAPI`) and `apps/inference/batch.py` share one `InferenceEngine` class that loads the exact `(model, preprocessor.joblib, optimal_threshold)` bundle atomically from the registry entry.

### Q3: Walk through a nasty statistical bug you caught during audit and how you fixed it.
- Earlier saved reports (`experiments/reports/experiment_report_20260823_175323.json`) had `bootstrap_ci` ranges near `0.001` despite `0.8303` PR-AUC because `compute_metrics()` drew **two independent** `rng.integers()` index arrays for `y_true` and `y_proba`, breaking `(truth, prediction)` row alignment during bootstrap resampling.
- Fixing `compute_metrics()` in `packages/evaluation/metrics.py` to sample one paired index array `idx` per bootstrap iteration restores valid `2.5%–97.5%` empirical CIs, verified by `test_paired_bootstrap_ci_brackets_signal` in `tests/test_metrics.py`.

### Q4: How do you prevent automated retraining from silently degrading production?
- `RetrainingPipeline.create_candidate()` logs a monitoring trigger event, runs `TrainingPipeline`, and registers the model strictly in `Development`. `validate_candidate()` blocks progression unless candidate PR-AUC `>= 0.95 * production_pr_auc`, `ModelRegistry.promote()` requires both `validation_passed=True` and `PR-AUC >= 0.50` before `Production`, and promotion demotes the old version to `Archived` so `modelops model rollback` restores service in one step.

### Q5: What did the LightGBM feature-interaction ablation teach you?
- Default LightGBM baseline collapsed to `0.0887` PR-AUC (`512` false positives at `0.1065` precision), whereas adding top-5 pairwise variance interactions raised LightGBM to `0.6774` PR-AUC and `0.9623` precision (`+0.5887` PR-AUC), while tuned XGBoost achieved `0.8303` PR-AUC in `9.5s` (beating PyTorch MLP's `0.7473` PR-AUC at `483s`).
