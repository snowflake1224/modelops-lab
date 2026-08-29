"""Integration test for training pipeline (uses synthetic data fallback)."""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path


@pytest.fixture
def synthetic_dataset(tmp_path):
    """Create synthetic fraud-like dataset for testing without network."""
    rng = np.random.default_rng(42)
    n = 500
    features = {f"V{i}": rng.normal(0, 1, n) for i in range(1, 11)}
    features["Amount"] = rng.exponential(50, n)
    features["Class"] = (rng.random(n) < 0.05).astype(int)
    df = pd.DataFrame(features)
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    df.to_parquet(raw_dir / "credit_card_fraud.parquet")
    return tmp_path


def test_data_pipeline_split(synthetic_dataset, monkeypatch):
    from packages.data.pipeline import DataPipeline
    from packages.config.settings import get_settings

    # Monkeypatch data dir
    monkeypatch.setenv("DATA_DIR", str(synthetic_dataset))
    pipeline = DataPipeline(seed=42)

    # Use pre-saved data
    from packages.data.datasets.credit_card_fraud import CreditCardFraudAdapter
    adapter = CreditCardFraudAdapter()
    df = pd.read_parquet(synthetic_dataset / "raw" / "credit_card_fraud.parquet")
    metadata = adapter.get_metadata(df)

    from packages.features.preprocessing import FeaturePreprocessor
    from sklearn.model_selection import StratifiedShuffleSplit

    target = "Class"
    feature_cols = [c for c in df.columns if c != target]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    train_idx, test_idx = next(sss.split(df, df[target]))
    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]

    preprocessor = FeaturePreprocessor()
    preprocessor.fit(train_df[feature_cols], train_df[target])
    x_train = preprocessor.transform(train_df[feature_cols])
    x_test = preprocessor.transform(test_df[feature_cols])

    assert x_train.shape[0] > 0
    assert x_test.shape[0] > 0


def test_model_training():
    from packages.models.base import ModelConfig, ModelType, create_model
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, (200, 10))
    y = (rng.random(200) < 0.1).astype(int)

    config = ModelConfig(model_type=ModelType.LOGISTIC_REGRESSION)
    model = create_model(config)
    model.fit(x, y)
    proba = model.predict_proba(x[:10])
    assert len(proba) == 10
