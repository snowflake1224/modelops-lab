# Monitoring

## Stack

- **Prometheus**: Live `/metrics` endpoint exposed by `apps/inference/main.py` via `prometheus_client` and scraped by `infra/prometheus/prometheus.yml`
- **Grafana**: Pre-provisioned Prometheus datasource (`infra/grafana/provisioning/datasources/datasource.yml`); dashboard JSON files can be imported via UI or provisioned into `infra/grafana/`
- **MonitoringStore**: File-backed event & performance history store (`data/monitoring/{events,performance}.json`) queried by `apps/api` (`/monitoring/*`, `/drift/events`)
- **Planned / Dependency-Declared**: `opentelemetry-*` and `structlog` are declared in `pyproject.toml` for future distributed tracing and structured JSON logging integration beyond request-ID headers (`X-Request-ID`) and Prometheus metrics

## System Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `inference_requests_total` | Counter | Request count by endpoint and status |
| `inference_latency_seconds` | Histogram | Prediction latency |
| `prediction_probability` | Histogram | Score distribution |

## ML Metrics

Tracked via monitoring store:
- Prediction distribution
- Confidence distribution
- Class distribution
- Feature distribution (for drift)
- Delayed ground-truth performance

## Performance Monitoring

When labels become available:

```python
store.record_performance(y_true, y_proba, threshold=0.5)
```

Calculates precision, recall, F1, PR-AUC, calibration over time.

### Degradation Detection

If PR-AUC drops >10% between consecutive snapshots, a `performance_degradation` event is created.

## Grafana

Access at http://localhost:3000 (`admin`/`admin`).

The Prometheus datasource (`http://prometheus:9090`) is pre-provisioned as default so you can build or import dashboards against the scraped inference metrics (`inference_requests_total`, `inference_latency_seconds`, `prediction_probability`).

## API Endpoints

```
GET /monitoring/events
GET /monitoring/performance
GET /drift/events
```

## Retraining Policy

Monitoring events trigger retraining candidates but never auto-deploy:

1. Drift detected → monitoring event
2. Performance degradation → monitoring event
3. Events reviewed → retraining candidate created
4. Candidate validated → staged for approval
5. Explicit promotion → production
