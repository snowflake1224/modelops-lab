"""Fixtures for tests."""

import pytest

@pytest.fixture(autouse=True)
def mlflow_store(tmp_path, monkeypatch):
    db_path = tmp_path / "mlflow.db"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{db_path}")
