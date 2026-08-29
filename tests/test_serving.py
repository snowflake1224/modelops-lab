"""Tests for inference service."""

import pytest
from fastapi.testclient import TestClient

from apps.inference.main import app


@pytest.fixture
def client():
    return TestClient(app)


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
