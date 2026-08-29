"""XGBoost classifier."""

import time

import numpy as np
import xgboost as xgb

from packages.models.base import BaseModel


class XGBoostModel(BaseModel):
    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        x_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "XGBoostModel":
        start = time.perf_counter()
        hp = {
            "n_estimators": self.config.hyperparameters.get("n_estimators", 200),
            "learning_rate": self.config.hyperparameters.get("learning_rate", 0.05),
            "max_depth": self.config.hyperparameters.get("max_depth", 6),
            "subsample": self.config.hyperparameters.get("subsample", 0.8),
            "colsample_bytree": self.config.hyperparameters.get("colsample_bytree", 0.8),
            "random_state": self.config.random_seed,
            "eval_metric": "aucpr",
            "verbosity": 0,
        }
        if self.config.class_weight == "balanced":
            scale = (y == 0).sum() / max((y == 1).sum(), 1)
            hp["scale_pos_weight"] = scale
        self.model = xgb.XGBClassifier(**hp)
        eval_set = [(x_val, y_val)] if x_val is not None and y_val is not None else None
        self.model.fit(x, y, eval_set=eval_set, verbose=False)
        self.training_duration_s = time.perf_counter() - start
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(x)[:, 1]
