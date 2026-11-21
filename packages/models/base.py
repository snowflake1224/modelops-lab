"""Base model interface for consistent training/evaluation."""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import joblib
import numpy as np


class ModelType(StrEnum):
    LOGISTIC_REGRESSION = "logistic_regression"
    LIGHTGBM = "lightgbm"
    XGBOOST = "xgboost"
    PYTORCH_MLP = "pytorch_mlp"


@dataclass
class ModelConfig:
    model_type: ModelType
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    class_weight: str | dict[str, float] | None = "balanced"
    threshold: float = 0.5
    random_seed: int = 42


class BaseModel(ABC):
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model: Any = None
        self.training_duration_s: float = 0.0
        self.feature_names_: list[str] = []

    @abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray, x_val: np.ndarray | None = None, y_val: np.ndarray | None = None) -> "BaseModel":
        ...

    @abstractmethod
    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        ...

    def predict(self, x: np.ndarray, threshold: float | None = None) -> np.ndarray:
        thresh = threshold if threshold is not None else self.config.threshold
        proba = self.predict_proba(x)
        return (proba >= thresh).astype(int)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "config": self.config,
                "model": self.model,
                "feature_names": self.feature_names_,
                "training_duration_s": self.training_duration_s,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> "BaseModel":
        data = joblib.load(path)
        instance = create_model(data["config"])
        instance.model = data["model"]
        instance.feature_names_ = data.get("feature_names", [])
        instance.training_duration_s = data.get("training_duration_s", 0.0)
        return instance

    def get_model_size_bytes(self, path: Path | None = None) -> int:
        if path and path.exists():
            return path.stat().st_size
        import io
        buffer = io.BytesIO()
        joblib.dump(self.model, buffer)
        return buffer.tell()


def create_model(config: ModelConfig) -> BaseModel:
    if config.model_type == ModelType.LOGISTIC_REGRESSION:
        from packages.models.logistic import LogisticRegressionModel

        return LogisticRegressionModel(config)
    if config.model_type == ModelType.LIGHTGBM:
        from packages.models.lightgbm_model import LightGBMModel

        return LightGBMModel(config)
    if config.model_type == ModelType.XGBOOST:
        from packages.models.xgboost_model import XGBoostModel

        return XGBoostModel(config)
    if config.model_type == ModelType.PYTORCH_MLP:
        from packages.models.pytorch_mlp import PyTorchMLPModel

        return PyTorchMLPModel(config)
    raise ValueError(f"Unknown model type: {config.model_type}")


def timed_fit(model: BaseModel, x: np.ndarray, y: np.ndarray, x_val: np.ndarray | None = None, y_val: np.ndarray | None = None) -> BaseModel:
    start = time.perf_counter()
    model.fit(x, y, x_val, y_val)
    model.training_duration_s = time.perf_counter() - start
    return model
