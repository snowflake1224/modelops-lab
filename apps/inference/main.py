"""FastAPI inference service."""

import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

from apps.inference.engine import InferenceEngine
from packages.config.settings import get_settings

REQUEST_COUNT = Counter("inference_requests_total", "Total inference requests", ["endpoint", "status"])
REQUEST_LATENCY = Histogram("inference_latency_seconds", "Inference latency", ["endpoint"])
PREDICTION_DIST = Histogram("prediction_probability", "Prediction probability distribution", buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

engine: InferenceEngine | None = None


class PredictRequest(BaseModel):
    features: dict[str, float] = Field(..., description="Feature name to value mapping")


class BatchPredictRequest(BaseModel):
    instances: list[dict[str, float]] = Field(..., max_length=1000)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    settings = get_settings()
    engine = InferenceEngine(settings.data_dir / "registry")
    try:
        engine.load()
    except RuntimeError:
        pass  # Service starts but not ready until model registered
    yield


app = FastAPI(title="ModelOps Lab Inference", version="0.1.0", lifespan=lifespan)


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4())[:8])
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.get("/ready")
async def ready():
    if engine and engine.is_ready:
        return {"status": "ready", "model_version": engine.model_version}
    raise HTTPException(status_code=503, detail="Model not loaded")


@app.get("/model")
async def model_info():
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    return engine.get_model_info()


@app.get("/version")
async def version():
    return {"service": "modelops-inference", "version": "0.1.0"}


@app.post("/predict")
async def predict(request: Request, body: PredictRequest):
    if not engine or not engine.is_ready:
        REQUEST_COUNT.labels(endpoint="predict", status="error").inc()
        raise HTTPException(status_code=503, detail="Model not ready")

    try:
        with REQUEST_LATENCY.labels(endpoint="predict").time():
            result = engine.predict([body.features], request_id=request.state.request_id)
        for p in result.probabilities:
            PREDICTION_DIST.observe(p)
        REQUEST_COUNT.labels(endpoint="predict", status="success").inc()
        return {
            "prediction": result.predictions[0],
            "probability": result.probabilities[0],
            "model_version": result.model_version,
            "latency_ms": result.latency_ms,
            "request_id": result.request_id,
        }
    except ValueError as e:
        REQUEST_COUNT.labels(endpoint="predict", status="error").inc()
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        REQUEST_COUNT.labels(endpoint="predict", status="error").inc()
        raise HTTPException(status_code=500, detail="Prediction failed") from e


@app.post("/predict/batch")
async def predict_batch(request: Request, body: BatchPredictRequest):
    settings = get_settings()
    if len(body.instances) > settings.max_batch_size:
        raise HTTPException(status_code=400, detail=f"Batch size exceeds max {settings.max_batch_size}")

    if not engine or not engine.is_ready:
        raise HTTPException(status_code=503, detail="Model not ready")

    try:
        with REQUEST_LATENCY.labels(endpoint="predict_batch").time():
            result = engine.predict(body.instances, request_id=request.state.request_id)
        REQUEST_COUNT.labels(endpoint="predict_batch", status="success").inc()
        return {
            "predictions": result.predictions,
            "probabilities": result.probabilities,
            "model_version": result.model_version,
            "latency_ms": result.latency_ms,
            "request_id": result.request_id,
        }
    except ValueError as e:
        REQUEST_COUNT.labels(endpoint="predict_batch", status="error").inc()
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.get("/metrics")
async def metrics():
    return JSONResponse(content=generate_latest().decode(), media_type="text/plain")
