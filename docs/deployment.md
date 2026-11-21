# Deployment

## Docker Compose

```bash
docker compose up -d
```

### Services

| Service | Port | Purpose |
|---------|------|---------|
| postgres | 5432 | Metadata storage |
| mlflow | 5000 | Experiment tracking |
| inference | 8000 | Prediction API |
| api | 8080 | Platform API |
| prometheus | 9090 | Metrics |
| grafana | 3000 | Dashboards |

## Inference API

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/predict` | Single prediction |
| POST | `/predict/batch` | Batch prediction (max 1000) |
| GET | `/health` | Liveness probe |
| GET | `/ready` | Readiness probe (model loaded) |
| GET | `/model` | Model metadata |
| GET | `/version` | Service version |
| GET | `/metrics` | Prometheus metrics |

### Example

`InferenceEngine.predict()` expects all **29** training features (`V1`–`V28` and `Amount`) and raises `400 Bad Request` (`Missing features: [...]`) if any feature is omitted:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "V1": -1.36, "V2": -0.07, "V3": 2.54, "V4": 1.38, "V5": -0.34,
      "V6": 0.46, "V7": 0.24, "V8": 0.10, "V9": 0.36, "V10": 0.09,
      "V11": -0.55, "V12": -0.62, "V13": -0.99, "V14": -0.31, "V15": 1.47,
      "V16": -0.47, "V17": 0.21, "V18": 0.03, "V19": 0.40, "V20": 0.25,
      "V21": -0.02, "V22": 0.28, "V23": -0.11, "V24": 0.07, "V25": 0.13,
      "V26": -0.19, "V27": 0.13, "V28": -0.02, "Amount": 149.62
    }
  }'
```

> **Note:** Pass raw unscaled feature values; `InferenceEngine` applies the saved `preprocessor.joblib` (`RobustScaler` + quantile clipping bounds + optional interactions) inside the serving path automatically.

## Security

- No secrets in repository (`.env.example` only)
- Environment-based credentials
- Request size limits (1MB default)
- Batch size limits (1000 default)
- Input schema validation via Pydantic
- Container isolation via Docker
- Safe model loading (no arbitrary pickle execution in API)

## Future Kubernetes Migration

The Docker Compose setup maps cleanly to Kubernetes:
- Each service → Deployment + Service
- PostgreSQL → StatefulSet or managed DB
- ConfigMaps for configs
- Secrets for credentials
- HPA on inference service

## Batch Inference

```bash
modelops batch data/input.parquet --version 1.0.0
```

Independent of online API, shares `InferenceEngine`.
