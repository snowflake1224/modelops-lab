"""Platform API for experiments, models, and monitoring."""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import mlflow
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from packages.config.settings import get_settings
from packages.drift.detector import detect_dataset_drift
from packages.monitoring.store import MonitoringStore
from packages.registry.store import ModelRegistry, ModelStage
from packages.training.pipeline import TrainingPipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    yield


app = FastAPI(title="ModelOps Lab API", version="0.1.0", lifespan=lifespan)


class TrainRequest(BaseModel):
    config_path: str


class PromoteRequest(BaseModel):
    name: str
    version: str
    target_stage: str


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/experiments/train")
async def launch_training(req: TrainRequest):
    config_path = Path(req.config_path)
    if not config_path.exists():
        raise HTTPException(status_code=404, detail="Config not found")
    try:
        pipeline = TrainingPipeline(config_path)
        result = pipeline.run()
        return {"status": "completed", **result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/experiments")
async def list_experiments():
    client = mlflow.tracking.MlflowClient()
    experiments = client.search_experiments()
    return [{"id": e.experiment_id, "name": e.name} for e in experiments]


@app.get("/experiments/{run_id}/metrics")
async def get_metrics(run_id: str):
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    return {"metrics": run.data.metrics, "params": run.data.params}


@app.get("/models")
async def list_models():
    settings = get_settings()
    registry = ModelRegistry(settings.data_dir / "registry")
    versions = registry.list_versions()
    return [
        {
            "name": v.name,
            "version": v.version,
            "stage": v.stage.value,
            "metrics": v.metrics,
            "created_at": v.created_at,
        }
        for v in versions
    ]


@app.post("/models/promote")
async def promote_model(req: PromoteRequest):
    settings = get_settings()
    registry = ModelRegistry(settings.data_dir / "registry")
    try:
        mv = registry.promote(req.name, req.version, ModelStage(req.target_stage))
        return {"status": "promoted", "stage": mv.stage.value}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.get("/monitoring/events")
async def get_monitoring_events(event_type: str | None = None):
    store = MonitoringStore()
    events = store.get_events(event_type)
    return [{"id": e.id, "type": e.event_type, "severity": e.severity, "details": e.details, "created_at": e.created_at} for e in events]


@app.get("/monitoring/performance")
async def get_performance():
    store = MonitoringStore()
    history = store.get_performance_history()
    return [{"timestamp": p.timestamp, "metrics": p.metrics, "n_samples": p.n_samples} for p in history]


@app.get("/drift/events")
async def get_drift_events():
    store = MonitoringStore()
    return store.get_events("drift")


@app.get("/lineage/{record_id}")
async def get_lineage(record_id: str):
    from packages.registry.lineage import LineageTracker
    tracker = LineageTracker()
    record = tracker.get(record_id)
    if not record:
        raise HTTPException(status_code=404, detail="Lineage record not found")
    return record.__dict__
