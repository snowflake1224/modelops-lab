"""Credit Card Fraud Detection dataset from OpenML."""

import hashlib
from pathlib import Path

import numpy as np
import openml
import pandas as pd

from packages.data.adapter import DatasetAdapter, DatasetMetadata


class CreditCardFraudAdapter(DatasetAdapter):
    """OpenML dataset 1597 - Credit Card Fraud Detection."""

    DATASET_ID = 1597
    TARGET = "Class"

    @property
    def name(self) -> str:
        return "credit_card_fraud"

    def download(self, output_dir: Path) -> Path:
        dataset = openml.datasets.get_dataset(self.DATASET_ID)
        x, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
        df = pd.concat([x, y], axis=1)
        for col in df.columns:
            if df[col].dtype.name == "category" or df[col].dtype == object:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        path = output_dir / "credit_card_fraud_raw.parquet"
        df.to_parquet(path, index=False)
        return path

    def load(self, path: Path) -> pd.DataFrame:
        df = pd.read_parquet(path)
        # Ensure numeric types for sklearn/torch compatibility
        for col in df.columns:
            if df[col].dtype.name == "category" or df[col].dtype == object:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    def validate(self, df: pd.DataFrame) -> list[str]:
        errors: list[str] = []
        if self.TARGET not in df.columns:
            errors.append(f"Missing target column: {self.TARGET}")
        if df.empty:
            errors.append("Dataset is empty")
        if df[self.TARGET].nunique() != 2:
            errors.append("Target must be binary")
        if df.isnull().sum().sum() > 0:
            errors.append("Dataset contains missing values")
        return errors

    def get_metadata(self, df: pd.DataFrame) -> DatasetMetadata:
        feature_cols = [c for c in df.columns if c != self.TARGET]
        checksum = hashlib.sha256(
            pd.util.hash_pandas_object(df, index=True).values.tobytes()
        ).hexdigest()[:16]
        return DatasetMetadata(
            name=self.name,
            version="v1",
            source=f"openml:{self.DATASET_ID}",
            target_column=self.TARGET,
            feature_columns=feature_cols,
            row_count=len(df),
            positive_rate=float(df[self.TARGET].mean()),
            checksum=checksum,
            extra={"n_features": len(feature_cols)},
        )


def get_adapter(name: str) -> DatasetAdapter:
    adapters = {"credit_card_fraud": CreditCardFraudAdapter}
    if name not in adapters:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(adapters)}")
    return adapters[name]()
