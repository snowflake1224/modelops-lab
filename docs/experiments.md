# Experiments

## Running Experiments

```bash
# Single experiment
modelops experiment run configs/lightgbm/baseline.yaml

# All baselines
python scripts/run_experiments.py

# Hyperparameter optimization
modelops experiment hpo configs/lightgbm/baseline.yaml --n-trials 20
```

## Experiment Configuration

YAML configs specify:
- Dataset and seed
- Preprocessing options
- Model type and hyperparameters
- Class weight strategy

Example: `configs/lightgbm/baseline.yaml`

## Tracked Metadata

Every experiment records in MLflow:
- Dataset version
- Git commit hash
- Model type and hyperparameters
- Feature/preprocessing version
- Random seed
- All evaluation metrics
- Training duration
- Environment metadata (Python version, package versions)
- Model artifact and preprocessor

## Metrics

For imbalanced classification:

| Metric | Why |
|--------|-----|
| PR-AUC | Primary metric for imbalanced data; focuses on positive class |
| ROC-AUC | Overall discrimination ability |
| Precision | Cost of false alarms |
| Recall | Cost of missed fraud |
| F1 | Harmonic mean for threshold selection |
| Calibration | Probability reliability |
| Bootstrap CI | Statistical confidence on key metrics |

## Controlled Experiments

### Model Comparison
Compare all four model types with identical preprocessing and data splits.

### Feature Ablation
`configs/lightgbm/with_interactions.yaml` enables pairwise feature interactions.

### Class Imbalance
All models use `class_weight: balanced` or equivalent `scale_pos_weight`.

### Threshold Selection
Optimal F1 threshold computed on validation set, applied to test evaluation.

## Hyperparameter Optimization

Optuna with:
- Model-specific search spaces
- Median pruner for early stopping
- PR-AUC as objective
- Results logged to MLflow

## Reproducibility

Re-run any experiment:
```bash
modelops experiment run configs/<model>/baseline.yaml
```

Same config + seed + data version → reproducible results (within floating-point tolerance).
