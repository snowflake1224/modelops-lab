#!/usr/bin/env python3
"""Run all baseline experiments and generate comparison report."""

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from packages.training.pipeline import TrainingPipeline

CONFIGS = [
    "configs/logistic_regression/baseline.yaml",
    "configs/lightgbm/baseline.yaml",
    "configs/xgboost/baseline.yaml",
    "configs/pytorch_mlp/baseline.yaml",
    "configs/lightgbm/with_interactions.yaml",
]


def main():
    root = Path(__file__).resolve().parents[1]
    results = []

    for config_path in CONFIGS:
        full_path = root / config_path
        if not full_path.exists():
            print(f"Skipping {config_path} - not found")
            continue
        print(f"\n{'='*60}")
        print(f"Running: {config_path}")
        print(f"{'='*60}")
        try:
            pipeline = TrainingPipeline(full_path)
            result = pipeline.run()
            results.append({
                "config": config_path,
                "run_id": result["run_id"],
                "model": config_path.split("/")[1],
                "test_metrics": result["test_metrics"],
                "training_duration_s": result["training_duration_s"],
                "optimal_threshold": result["optimal_threshold"],
            })
            print(f"  PR-AUC: {result['test_metrics'].get('pr_auc', 0):.4f}")
            print(f"  F1: {result['test_metrics'].get('f1', 0):.4f}")
        except Exception as e:
            print(f"  FAILED: {e}")
            results.append({"config": config_path, "error": str(e)})

    # Generate report
    report_dir = root / "experiments" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"experiment_report_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.json"
    report_path.write_text(json.dumps(results, indent=2, default=str))

    print(f"\n{'='*60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<25} {'PR-AUC':>8} {'F1':>8} {'Recall':>8} {'Time(s)':>8}")
    print("-" * 60)
    for r in results:
        if "error" in r:
            print(f"{r['config']:<25} FAILED")
            continue
        m = r["test_metrics"]
        print(f"{r['model']:<25} {m.get('pr_auc', 0):>8.4f} {m.get('f1', 0):>8.4f} {m.get('recall', 0):>8.4f} {r['training_duration_s']:>8.1f}")

    print(f"\nReport saved to: {report_path}")
    return results


if __name__ == "__main__":
    main()
