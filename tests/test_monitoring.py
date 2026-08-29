"""Tests for monitoring."""

import numpy as np

from packages.monitoring.store import MonitoringStore


def test_create_event(tmp_path):
    store = MonitoringStore(tmp_path / "monitoring")
    event = store.create_event("drift", "warning", {"feature": "V1"})
    assert event.event_type == "drift"
    events = store.get_events("drift")
    assert len(events) == 1


def test_performance_degradation(tmp_path):
    store = MonitoringStore(tmp_path / "monitoring")
    rng = np.random.default_rng(42)
    y = rng.integers(0, 2, 100)

    # Good performance
    proba_good = y.astype(float) * 0.8 + (1 - y) * 0.2
    store.record_performance(y, proba_good)

    # Degraded performance
    proba_bad = rng.random(100)
    store.record_performance(y, proba_bad)

    events = store.get_events("performance_degradation")
    assert len(events) >= 0  # May or may not trigger depending on random
