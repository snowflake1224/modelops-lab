"""Data lineage tracking."""

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from packages.config.settings import get_settings


@dataclass
class LineageRecord:
    id: str
    dataset_version: str
    preprocessing_version: str
    feature_version: str
    experiment_id: str
    model_artifact: str
    git_commit: str
    config: dict[str, Any]
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    deployment_id: str | None = None


class LineageTracker:
    def __init__(self, path: Path | None = None):
        settings = get_settings()
        self.path = path or settings.data_dir / "lineage" / "lineage.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._records: dict[str, LineageRecord] = {}
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            data = json.loads(self.path.read_text())
            for rid, rec in data.items():
                self._records[rid] = LineageRecord(**rec)

    def _save(self) -> None:
        data = {rid: asdict(rec) for rid, rec in self._records.items()}
        self.path.write_text(json.dumps(data, indent=2, default=str))

    def record(self, **kwargs: Any) -> LineageRecord:
        record_id = str(uuid.uuid4())[:8]
        rec = LineageRecord(id=record_id, **kwargs)
        self._records[record_id] = rec
        self._save()
        return rec

    def get(self, record_id: str) -> LineageRecord | None:
        return self._records.get(record_id)

    def trace_production_model(self, model_version: str) -> LineageRecord | None:
        for rec in self._records.values():
            if model_version in rec.model_artifact:
                return rec
        return None

    def list_records(self) -> list[LineageRecord]:
        return sorted(self._records.values(), key=lambda r: r.created_at, reverse=True)
