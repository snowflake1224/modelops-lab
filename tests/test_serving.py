"""Tests for inference service."""

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from apps.inference.main import app


@pytest.fixture
def client(tmp_path, monkeypatch):
    isolated_settings = SimpleNamespace(
        data_dir=tmp_path,
        max_batch_size=1000,
    )
    monkeypatch.setattr("apps.inference.main.get_settings", lambda: isolated_settings)
    with TestClient(app) as test_client:
        yield test_client


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_version(client):
    response = client.get("/version")
    assert response.status_code == 200


def test_ready_without_model(client):
    response = client.get("/ready")
    assert response.status_code == 503


def test_predict_without_model(client):
    response = client.post("/predict", json={"features": {"V1": 0.1, "V2": -0.5}})
    assert response.status_code == 503
