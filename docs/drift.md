# Drift Detection

## Techniques

### Population Stability Index (PSI)

Measures distribution shift between reference and current data.

| PSI Value | Interpretation |
|-----------|---------------|
| < 0.1 | No significant change |
| 0.1 - 0.2 | Moderate change |
| > 0.2 | Significant drift (default threshold) |

### Kolmogorov-Smirnov Test

Non-parametric test comparing cumulative distributions.

- H₀: distributions are identical
- Default threshold: p-value < 0.05

## Configuration

```env
DRIFT_PSI_THRESHOLD=0.2
DRIFT_KS_THRESHOLD=0.05
```

## Usage

```bash
modelops drift check
```

Compares training reference distribution against test distribution for all features.

## Drift Response

When drift exceeds thresholds:

1. **Drift detected** per feature
2. **Monitoring event** created with drifted feature list
3. **Model flagged** for review
4. **Retraining candidate** optionally created

No automatic model replacement.

## Per-Feature Results

```json
{
  "feature": "V14",
  "psi": 0.35,
  "ks_statistic": 0.12,
  "ks_pvalue": 0.001,
  "psi_drift": true,
  "ks_drift": true,
  "drift_detected": true
}
```

## Limitations

- Univariate drift only (no multivariate detection)
- Reference distribution is static (training data)
- No concept drift detection (requires labels)
- Thresholds are global, not per-feature
