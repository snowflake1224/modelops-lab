"""Monitoring events and performance tracking."""

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from packages.config.settings import get_settings
from packages.evaluation.metrics import compute_metrics, metrics_to_dict


@dataclass
class MonitoringEvent:
    id: str
    event_type: str  # drift, performance_degradation, retrain_candidate
    severity: str
    details: dict[str, Any]
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    resolved: bool = False


@dataclass
class PerformanceSnapshot:
    timestamp: str
    metrics: dict[str, float]
    n_samples: int


class MonitoringStore:
    def __init__(self, path: Path | None = None):
        settings = get_settings()
        self.path = path or settings.data_dir / "monitoring"
        self.path.mkdir(parents=True, exist_ok=True)
        self._events_path = self.path / "events.json"
        self._perf_path = self.path / "performance.json"
        self._events: list[MonitoringEvent] = []
        self._performance: list[PerformanceSnapshot] = []
        self._load()

    def _load(self) -> None:
        if self._events_path.exists():
            self._events = [MonitoringEvent(**e) for e in json.loads(self._events_path.read_text())]
        if self._perf_path.exists():
            self._performance = [
                PerformanceSnapshot(**p) for p in json.loads(self._perf_path.read_text())
            ]

    def _save_events(self) -> None:
        self._events_path.write_text(json.dumps([asdict(e) for e in self._events], indent=2))

    def _save_performance(self) -> None:
        self._perf_path.write_text(json.dumps([asdict(p) for p in self._performance], indent=2))

    def create_event(self, event_type: str, severity: str, details: dict) -> MonitoringEvent:
        event = MonitoringEvent(
            id=str(uuid.uuid4())[:8],
            event_type=event_type,
            severity=severity,
            details=details,
        )
        self._events.append(event)
        self._save_events()
        return event

    def record_performance(self, y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5) -> PerformanceSnapshot:
        metrics = compute_metrics(y_true, y_proba, threshold=threshold, n_bootstrap=50)
        snapshot = PerformanceSnapshot(
            timestamp=datetime.now(UTC).isoformat(),
            metrics={k: v for k, v in metrics_to_dict(metrics).items() if isinstance(v, (int, float))},
            n_samples=len(y_true),
        )
        self._performance.append(snapshot)
        self._save_performance()

        # Check degradation
        if len(self._performance) >= 2:
            prev_pr_auc = self._performance[-2].metrics.get("pr_auc", 0)
            curr_pr_auc = snapshot.metrics.get("pr_auc", 0)
            if prev_pr_auc > 0 and (prev_pr_auc - curr_pr_auc) / prev_pr_auc > 0.1:
                self.create_event(
                    "performance_degradation",
                    "warning",
                    {"prev_pr_auc": prev_pr_auc, "curr_pr_auc": curr_pr_auc},
                )
        return snapshot

    def get_events(self, event_type: str | None = None) -> list[MonitoringEvent]:
        if event_type:
            return [e for e in self._events if e.event_type == event_type]
        return self._events

    def get_performance_history(self) -> list[PerformanceSnapshot]:
        return self._performance
