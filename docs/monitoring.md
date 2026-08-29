# Monitoring

## Stack

- **OpenTelemetry**: Distributed tracing instrumentation
- **Prometheus**: Metrics collection and storage
- **Grafana**: Visualization dashboards

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

## Grafana Dashboard

Access at http://localhost:3000 (admin/admin)

Prometheus datasource pre-configured. Dashboard panels:
- Request rate
- Latency percentiles
- Error rate
- Prediction distribution

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
