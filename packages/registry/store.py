"""Model registry with lifecycle management."""

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any


class ModelStage(StrEnum):
    DEVELOPMENT = "development"
    VALIDATION = "validation"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"


VALID_TRANSITIONS = {
    ModelStage.DEVELOPMENT: {ModelStage.VALIDATION, ModelStage.ARCHIVED},
    ModelStage.VALIDATION: {ModelStage.STAGING, ModelStage.DEVELOPMENT, ModelStage.ARCHIVED},
    ModelStage.STAGING: {ModelStage.PRODUCTION, ModelStage.VALIDATION, ModelStage.ARCHIVED},
    ModelStage.PRODUCTION: {ModelStage.ARCHIVED},
    ModelStage.ARCHIVED: set(),
}


@dataclass
class ModelVersion:
    name: str
    version: str
    stage: ModelStage
    artifact_path: str
    metrics: dict[str, float]
    config: dict[str, Any]
    lineage_id: str | None
    git_commit: str
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    validation_passed: bool = False


class PromotionError(Exception):
    pass


class ModelRegistry:
    """File-based model registry with promotion rules."""

    def __init__(self, registry_path: Path):
        self.registry_path = registry_path
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self._index_path = registry_path / "registry.json"
        self._versions: dict[str, ModelVersion] = {}
        self._load()

    def _load(self) -> None:
        if self._index_path.exists():
            data = json.loads(self._index_path.read_text())
            for key, v in data.items():
                v["stage"] = ModelStage(v["stage"])
                self._versions[key] = ModelVersion(**v)

    def _save(self) -> None:
        data = {}
        for key, v in self._versions.items():
            d = v.__dict__.copy()
            d["stage"] = v.stage.value
            data[key] = d
        self._index_path.write_text(json.dumps(data, indent=2, default=str))

    def register(
        self,
        name: str,
        version: str,
        artifact_path: str,
        metrics: dict[str, float],
        config: dict,
        git_commit: str,
        lineage_id: str | None = None,
    ) -> ModelVersion:
        key = f"{name}:{version}"
        mv = ModelVersion(
            name=name,
            version=version,
            stage=ModelStage.DEVELOPMENT,
            artifact_path=artifact_path,
            metrics=metrics,
            config=config,
            lineage_id=lineage_id,
            git_commit=git_commit,
        )
        self._versions[key] = mv
        self._save()
        return mv

    def get(self, name: str, version: str) -> ModelVersion | None:
        return self._versions.get(f"{name}:{version}")

    def list_versions(self, name: str | None = None, stage: ModelStage | None = None) -> list[ModelVersion]:
        versions = list(self._versions.values())
        if name:
            versions = [v for v in versions if v.name == name]
        if stage:
            versions = [v for v in versions if v.stage == stage]
        return sorted(versions, key=lambda v: v.created_at, reverse=True)

    def promote(self, name: str, version: str, target_stage: ModelStage, metrics: dict | None = None) -> ModelVersion:
        key = f"{name}:{version}"
        mv = self._versions.get(key)
        if not mv:
            raise PromotionError(f"Model {key} not found")

        allowed = VALID_TRANSITIONS.get(mv.stage, set())
        if target_stage not in allowed:
            raise PromotionError(
                f"Cannot promote from {mv.stage.value} to {target_stage.value}. "
                f"Allowed: {[s.value for s in allowed]}"
            )

        if target_stage == ModelStage.PRODUCTION:
            if not mv.validation_passed:
                raise PromotionError(
                    "Cannot promote to production without passing validation checks"
                )
            # Demote current production
            for v in self._versions.values():
                if v.name == name and v.stage == ModelStage.PRODUCTION:
                    v.stage = ModelStage.ARCHIVED

        if target_stage in {ModelStage.STAGING, ModelStage.PRODUCTION}:
            validation_metrics = metrics or mv.metrics
            if validation_metrics.get("pr_auc", 0) < 0.5:
                raise PromotionError("PR-AUC below minimum threshold (0.5)")

        mv.stage = target_stage
        if target_stage == ModelStage.VALIDATION:
            mv.validation_passed = True
        self._save()
        return mv

    def rollback(self, name: str) -> ModelVersion | None:
        archived = [v for v in self._versions.values() if v.name == name and v.stage == ModelStage.ARCHIVED]
        if not archived:
            return None
        previous = sorted(archived, key=lambda v: v.created_at, reverse=True)[0]
        for v in self._versions.values():
            if v.name == name and v.stage == ModelStage.PRODUCTION:
                v.stage = ModelStage.ARCHIVED
        previous.stage = ModelStage.PRODUCTION
        self._save()
        return previous

    def get_production(self, name: str) -> ModelVersion | None:
        prod = [v for v in self._versions.values() if v.name == name and v.stage == ModelStage.PRODUCTION]
        return prod[0] if prod else None
