"""Tests for model registry."""

import pytest

from packages.registry.store import ModelRegistry, ModelStage, PromotionError


@pytest.fixture
def registry(tmp_path):
    return ModelRegistry(tmp_path / "registry")


def test_register_model(registry):
    mv = registry.register("test-model", "1.0", "/path/model", {"pr_auc": 0.8}, {}, "abc123")
    assert mv.stage == ModelStage.DEVELOPMENT


def test_promote_to_validation(registry):
    registry.register("test-model", "1.0", "/path", {"pr_auc": 0.8}, {}, "abc")
    mv = registry.promote("test-model", "1.0", ModelStage.VALIDATION)
    assert mv.stage == ModelStage.VALIDATION
    assert mv.validation_passed


def test_cannot_promote_to_production_without_validation(registry):
    registry.register("test-model", "1.0", "/path", {"pr_auc": 0.8}, {}, "abc")
    with pytest.raises(PromotionError):
        registry.promote("test-model", "1.0", ModelStage.PRODUCTION)


def test_production_promotion(registry):
    registry.register("test-model", "1.0", "/path", {"pr_auc": 0.8}, {}, "abc")
    registry.promote("test-model", "1.0", ModelStage.VALIDATION)
    registry.promote("test-model", "1.0", ModelStage.STAGING)
    mv = registry.promote("test-model", "1.0", ModelStage.PRODUCTION)
    assert mv.stage == ModelStage.PRODUCTION


def test_rollback(registry):
    registry.register("test-model", "1.0", "/path", {"pr_auc": 0.8}, {}, "abc")
    registry.promote("test-model", "1.0", ModelStage.VALIDATION)
    registry.promote("test-model", "1.0", ModelStage.STAGING)
    registry.promote("test-model", "1.0", ModelStage.PRODUCTION)
    registry.register("test-model", "2.0", "/path2", {"pr_auc": 0.7}, {}, "def")
    registry.promote("test-model", "2.0", ModelStage.VALIDATION)
    registry.promote("test-model", "2.0", ModelStage.STAGING)
    registry.promote("test-model", "2.0", ModelStage.PRODUCTION)
    rolled = registry.rollback("test-model")
    assert rolled is not None
    assert rolled.version == "1.0"
