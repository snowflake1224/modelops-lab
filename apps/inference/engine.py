"""Shared inference logic for online and batch serving."""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from packages.features.preprocessing import FeaturePreprocessor
from packages.models.base import BaseModel, ModelType
from packages.registry.store import ModelRegistry


@dataclass
class PredictionResult:
    predictions: list[int]
    probabilities: list[float]
    model_version: str
    latency_ms: float
    request_id: str


class InferenceEngine:
    """Loads production model and preprocessor; shared by API and batch."""

    def __init__(self, registry_path: Path, model_name: str = "fraud-classifier"):
        self.registry = ModelRegistry(registry_path)
        self.model_name = model_name
        self.model: BaseModel | None = None
        self.preprocessor: FeaturePreprocessor | None = None
        self.model_version: str = "unknown"
        self.threshold: float = 0.5
        self._loaded = False

    def load(self, version: str | None = None) -> None:
        if version:
            mv = self.registry.get(self.model_name, version)
        else:
            mv = self.registry.get_production(self.model_name)
        if not mv:
            raise RuntimeError(f"No production model found for {self.model_name}")

        artifact_dir = Path(mv.artifact_path).parent
        model_path = Path(mv.artifact_path)
        model_type = ModelType(mv.config.get("model", {}).get("type", "lightgbm"))

        if model_type == ModelType.PYTORCH_MLP:
            from packages.models.pytorch_mlp import PyTorchMLPModel

            self.model = PyTorchMLPModel.load(model_path)
        else:
            self.model = BaseModel.load(model_path)

        preprocessor_path = artifact_dir / "preprocessor.joblib"
        if preprocessor_path.exists():
            self.preprocessor = joblib.load(preprocessor_path)

        self.model_version = mv.version
        self.threshold = mv.config.get("optimal_threshold", 0.5)
        self._loaded = True

    @property
    def is_ready(self) -> bool:
        return self._loaded and self.model is not None

    def predict(
        self,
        features: list[dict[str, float]],
        request_id: str = "unknown",
    ) -> PredictionResult:
        if not self.is_ready or self.model is None:
            raise RuntimeError("Inference engine not loaded")

        start = time.perf_counter()
        import pandas as pd

        df = pd.DataFrame(features)
        model = self.model
        if self.preprocessor:
            expected = self.preprocessor.feature_names_
            missing = set(expected) - set(df.columns)
            if missing:
                raise ValueError(f"Missing features: {sorted(missing)}")
            columns = expected if all(c in df.columns for c in expected) else list(df.columns)
            x = self.preprocessor.transform(df[columns])
        else:
            x = df.values.astype(np.float64)

        proba = model.predict_proba(x)
        preds = (proba >= self.threshold).astype(int)
        latency = (time.perf_counter() - start) * 1000

        return PredictionResult(
            predictions=preds.tolist(),
            probabilities=proba.tolist(),
            model_version=self.model_version,
            latency_ms=latency,
            request_id=request_id,
        )

    def get_model_info(self) -> dict[str, Any]:
        return {
            "name": self.model_name,
            "version": self.model_version,
            "threshold": self.threshold,
            "ready": self.is_ready,
        }
