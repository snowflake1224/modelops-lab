"""Logistic regression baseline."""

import time

import numpy as np
from sklearn.linear_model import LogisticRegression

from packages.models.base import BaseModel, ModelConfig


class LogisticRegressionModel(BaseModel):
    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        x_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "LogisticRegressionModel":
        del x_val, y_val
        start = time.perf_counter()
        params = {
            "max_iter": self.config.hyperparameters.get("max_iter", 1000),
            "C": self.config.hyperparameters.get("C", 1.0),
            "class_weight": self.config.class_weight,
            "random_state": self.config.random_seed,
            "solver": "lbfgs",
        }
        self.model = LogisticRegression(**params)
        self.model.fit(x, y)
        self.training_duration_s = time.perf_counter() - start
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(x)[:, 1]
