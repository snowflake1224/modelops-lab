# Reproducibility

## What's Captured

| Artifact | Location |
|----------|----------|
| Code revision | Git commit in MLflow params |
| Dataset version | Metadata checksum + version |
| Configuration | YAML config logged as artifact |
| Random seed | MLflow param + numpy seed |
| Dependencies | `environment.json` artifact |
| Preprocessor | Serialized joblib artifact |
| Model weights | Serialized model artifact |

## Single-Command Reproduction

```bash
modelops experiment run configs/lightgbm/baseline.yaml
```

Given the same:
- Git commit
- Dataset version (cached in `data/raw/`)
- Configuration file
- Random seed

Results are reproducible within floating-point tolerance.

## Environment Locking

```bash
uv sync  # Installs exact versions from lockfile
```

CI runs on Ubuntu with Python 3.12 for consistency.

## Artifact Structure

```
data/artifacts/models/<run_id>/
├── model.joblib (or model.pt)
├── preprocessor.joblib
└── config.json
```

## Lineage Query

Answer "What produced this production model?":

```bash
# Via API
curl http://localhost:8080/lineage/<record_id>

# Via registry
modelops model list
```

Returns: dataset version, preprocessing version, experiment ID, git commit, full config.

## Limitations

- PyTorch results may vary slightly across hardware (CPU vs GPU)
- LightGBM/XGBoost may have minor differences across versions
- OpenML download is cached; force re-download with `force=True`
