"""Reproducible preprocessing for training and inference."""

import hashlib
import json
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler


@dataclass
class PreprocessorConfig:
    scale: bool = True
    add_interactions: bool = False
    clip_outliers: bool = True
    clip_quantile: float = 0.995
    version: str = "v1"


class FeaturePreprocessor:
    """Fit on training data only; transform consistently at inference."""

    def __init__(self, config: PreprocessorConfig | None = None):
        self.config = config or PreprocessorConfig()
        self.scaler_: RobustScaler | None = None
        self.feature_names_: list[str] = []
        self.clip_bounds_: dict[str, tuple[float, float]] = {}
        self._fitted = False

    @property
    def version(self) -> str:
        payload = asdict(self.config)
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]

    def fit(self, x: pd.DataFrame, y: pd.Series | None = None) -> "FeaturePreprocessor":
        del y  # target not used in preprocessing to avoid leakage
        df = x.copy()
        if self.config.clip_outliers:
            for col in df.columns:
                low = df[col].quantile(1 - self.config.clip_quantile)
                high = df[col].quantile(self.config.clip_quantile)
                self.clip_bounds_[col] = (float(low), float(high))
                df[col] = df[col].clip(low, high)

        if self.config.add_interactions:
            # Top-variance pairwise interactions (limited to prevent explosion)
            variances = df.var().sort_values(ascending=False)
            top_cols = variances.head(5).index.tolist()
            for i, c1 in enumerate(top_cols):
                for c2 in top_cols[i + 1 :]:
                    df[f"{c1}_x_{c2}"] = df[c1] * df[c2]

        self.feature_names_ = list(df.columns)
        if self.config.scale:
            self.scaler_ = RobustScaler()
            self.scaler_.fit(df.values)
        self._fitted = True
        return self

    def transform(self, x: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Preprocessor must be fitted before transform")
        df = x.copy()
        for col, (low, high) in self.clip_bounds_.items():
            if col in df.columns:
                df[col] = df[col].clip(low, high)
        if self.config.add_interactions:
            for name in self.feature_names_:
                if "_x_" in name and name not in df.columns:
                    c1, c2 = name.split("_x_")
                    if c1 in df.columns and c2 in df.columns:
                        df[name] = df[c1] * df[c2]
        df = df.reindex(columns=self.feature_names_, fill_value=0.0)
        values = df.values.astype(np.float64)
        if self.scaler_ is not None:
            values = self.scaler_.transform(values)
        return values

    def fit_transform(self, x: pd.DataFrame, y: pd.Series | None = None) -> np.ndarray:
        return self.fit(x, y).transform(x)
