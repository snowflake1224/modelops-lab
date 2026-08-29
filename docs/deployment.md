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

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": {"V1": -1.36, "V2": -0.07, "V3": 2.54, "V4": 1.38, "V5": -0.34}}'
```

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
