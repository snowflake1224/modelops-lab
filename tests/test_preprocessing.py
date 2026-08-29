"""Tests for preprocessing."""

import numpy as np
import pandas as pd
import pytest

from packages.features.preprocessing import FeaturePreprocessor, PreprocessorConfig


@pytest.fixture
def sample_data():
    rng = np.random.default_rng(42)
    n = 100
    return pd.DataFrame({
        "f1": rng.normal(0, 1, n),
        "f2": rng.normal(0, 1, n),
        "f3": rng.uniform(0, 1, n),
    }), pd.Series(rng.integers(0, 2, n))


def test_preprocessor_fit_transform(sample_data):
    x, y = sample_data
    preprocessor = FeaturePreprocessor(PreprocessorConfig(scale=True))
    result = preprocessor.fit_transform(x, y)
    assert result.shape == (100, 3)
    assert preprocessor.version


def test_preprocessor_consistency(sample_data):
    x, y = sample_data
    preprocessor = FeaturePreprocessor()
    preprocessor.fit(x, y)
    preprocessor.transform(x)
    infer_result = preprocessor.transform(x.iloc[:10])
    assert infer_result.shape == (10, len(preprocessor.feature_names_))


def test_interactions(sample_data):
    x, y = sample_data
    preprocessor = FeaturePreprocessor(PreprocessorConfig(add_interactions=True))
    result = preprocessor.fit_transform(x, y)
    assert result.shape[1] > 3


def test_unfitted_transform_raises(sample_data):
    x, _ = sample_data
    preprocessor = FeaturePreprocessor()
    with pytest.raises(RuntimeError):
        preprocessor.transform(x)
