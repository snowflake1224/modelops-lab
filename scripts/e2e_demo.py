#!/usr/bin/env python3
"""End-to-end platform workflow demonstration."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from packages.config.settings import get_settings
from packages.registry.store import ModelRegistry, ModelStage
from packages.training.pipeline import TrainingPipeline, get_git_commit


def main():
    root = Path(__file__).resolve().parents[1]
    settings = get_settings()
    config = root / "configs" / "xgboost" / "baseline.yaml"

    print("1. Training best model (XGBoost)...")
    result = TrainingPipeline(config).run()
    run_id = result["run_id"]
    print(f"   Run ID: {run_id}, PR-AUC: {result['test_metrics']['pr_auc']:.4f}")

    print("2. Registering model...")
    registry = ModelRegistry(settings.data_dir / "registry")
    mv = registry.register(
        name=settings.model_name,
        version="1.0.0",
        artifact_path=result["model_path"],
        metrics={k: v for k, v in result["test_metrics"].items() if isinstance(v, (int, float))},
        config={"run_id": run_id, "optimal_threshold": result["optimal_threshold"], "model": {"type": "xgboost"}},
        git_commit=get_git_commit(),
    )
    print(f"   Registered: {mv.name}:{mv.version} ({mv.stage.value})")

    print("3. Promoting through lifecycle...")
    registry.promote(settings.model_name, "1.0.0", ModelStage.VALIDATION)
    registry.promote(settings.model_name, "1.0.0", ModelStage.STAGING)
    registry.promote(settings.model_name, "1.0.0", ModelStage.PRODUCTION)
    print("   Model in PRODUCTION")

    print("4. Testing inference engine...")
    from apps.inference.engine import InferenceEngine
    engine = InferenceEngine(settings.data_dir / "registry")
    engine.load()
    import numpy as np
    import joblib
    processed = settings.processed_data_dir / "credit_card_fraud_v1"
    preprocessor = joblib.load(processed / "preprocessor.joblib")
    splits = np.load(processed / "splits.npz")
    sample_features = dict(zip(preprocessor.feature_names_, splits["x_test"][0]))
    pred = engine.predict([sample_features], request_id="e2e-demo")
    print(f"   Prediction: {pred.predictions[0]}, probability: {pred.probabilities[0]:.4f}")

    print("5. Drift check...")
    from packages.drift.detector import detect_dataset_drift
    results = detect_dataset_drift(splits["x_train"][:1000], splits["x_test"][:1000], preprocessor.feature_names_)
    drifted = sum(1 for r in results if r.drift_detected)
    print(f"   Drifted features: {drifted}/{len(results)}")

    print("6. Rollback test...")
    registry.register(
        name=settings.model_name, version="0.9.0",
        artifact_path=result["model_path"],
        metrics={"pr_auc": 0.5}, config={}, git_commit="test",
    )
    registry.promote(settings.model_name, "0.9.0", ModelStage.VALIDATION)
    registry.promote(settings.model_name, "0.9.0", ModelStage.STAGING)
    registry.promote(settings.model_name, "0.9.0", ModelStage.PRODUCTION)
    rolled = registry.rollback(settings.model_name)
    print(f"   Rolled back to: {rolled.version if rolled else 'none'}")

    print("\nEnd-to-end workflow complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
