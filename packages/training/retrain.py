"""Retraining workflow: monitoring event → candidate → validation → staging."""

import uuid
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from packages.config.settings import get_settings
from packages.evaluation.metrics import compute_metrics, metrics_to_dict
from packages.monitoring.store import MonitoringStore
from packages.registry.store import ModelRegistry, ModelStage
from packages.training.pipeline import TrainingPipeline, get_git_commit


class RetrainingPipeline:
    def __init__(self):
        self.settings = get_settings()
        self.registry = ModelRegistry(self.settings.data_dir / "registry")
        self.monitoring = MonitoringStore()

    def create_candidate(self, trigger_event_id: str | None = None, config_path: Path | None = None) -> dict[str, Any]:
        """Create retraining candidate from monitoring event."""
        config_path = config_path or self.settings.configs_dir / "lightgbm" / "baseline.yaml"
        candidate_id = str(uuid.uuid4())[:8]

        self.monitoring.create_event(
            "retrain_candidate",
            "info",
            {"candidate_id": candidate_id, "trigger_event_id": trigger_event_id, "config": str(config_path)},
        )

        pipeline = TrainingPipeline(config_path)
        result = pipeline.run()

        # Register as development
        version = f"candidate-{candidate_id}"
        self.registry.register(
            name=self.settings.model_name,
            version=version,
            artifact_path=result["model_path"],
            metrics={k: v for k, v in result["test_metrics"].items() if isinstance(v, (int, float))},
            config={"experiment_run_id": result["run_id"]},
            git_commit=get_git_commit(),
        )

        return {"candidate_id": candidate_id, "version": version, "run_id": result["run_id"], "metrics": result["test_metrics"]}

    def validate_candidate(self, version: str) -> dict[str, Any]:
        """Compare candidate against production model."""
        candidate = self.registry.get(self.settings.model_name, version)
        if not candidate:
            raise ValueError(f"Candidate {version} not found")

        production = self.registry.get_production(self.settings.model_name)
        comparison: dict[str, Any] = {
            "candidate_version": version,
            "candidate_metrics": candidate.metrics,
            "production_version": production.version if production else None,
            "production_metrics": production.metrics if production else {},
            "passed": False,
        }

        candidate_pr_auc = candidate.metrics.get("pr_auc", 0)
        prod_pr_auc = production.metrics.get("pr_auc", 0) if production else 0

        # Candidate must beat production or no production exists
        if candidate_pr_auc >= prod_pr_auc * 0.95:
            comparison["passed"] = True
            self.registry.promote(self.settings.model_name, version, ModelStage.VALIDATION)
            candidate.validation_passed = True

        return comparison

    def approve_for_staging(self, version: str) -> dict[str, Any]:
        mv = self.registry.promote(self.settings.model_name, version, ModelStage.STAGING, metrics=None)
        return {"version": mv.version, "stage": mv.stage.value}

    def rollback(self) -> dict[str, Any]:
        previous = self.registry.rollback(self.settings.model_name)
        if not previous:
            return {"status": "no_rollback_available"}
        return {"status": "rolled_back", "version": previous.version}
