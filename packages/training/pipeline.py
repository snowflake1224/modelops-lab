"""Training pipeline with MLflow experiment tracking."""

import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import yaml

from packages.config.settings import get_settings
from packages.data.pipeline import DataPipeline
from packages.evaluation.metrics import compute_metrics, find_optimal_threshold, metrics_to_dict
from packages.features.preprocessing import PreprocessorConfig
from packages.models.base import ModelConfig, ModelType, create_model
from packages.registry.lineage import LineageTracker
from packages.registry.store import ModelRegistry, ModelStage


def get_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()[:12]
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def load_experiment_config(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f)


class TrainingPipeline:
    def __init__(self, config_path: Path):
        self.config = load_experiment_config(config_path)
        self.settings = get_settings()
        mlflow.set_tracking_uri(self.settings.mlflow_tracking_uri)
        mlflow.set_experiment(self.config.get("experiment_name", self.settings.mlflow_experiment_name))

    def run(self) -> dict[str, Any]:
        seed = self.config.get("seed", self.settings.random_seed)
        np.random.seed(seed)

        preprocessor_cfg = PreprocessorConfig(**self.config.get("preprocessing", {}))
        pipeline = DataPipeline(
            dataset_name=self.config.get("dataset", "credit_card_fraud"),
            seed=seed,
        )
        split, preprocessor, metadata = pipeline.run(
            self.settings.data_dir,
            preprocessor_config=preprocessor_cfg,
        )

        model_type = ModelType(self.config["model"]["type"])
        model_config = ModelConfig(
            model_type=model_type,
            hyperparameters=self.config["model"].get("hyperparameters", {}),
            class_weight=self.config["model"].get("class_weight", "balanced"),
            threshold=self.config["model"].get("threshold", 0.5),
            random_seed=seed,
        )

        git_commit = get_git_commit()
        run_name = self.config.get("run_name", f"{model_type.value}-{int(time.time())}")

        with mlflow.start_run(run_name=run_name) as run:
            # Log parameters
            mlflow.log_params({
                "model_type": model_type.value,
                "dataset_version": metadata.version,
                "feature_version": preprocessor.version,
                "git_commit": git_commit,
                "seed": seed,
                **{f"hp_{k}": v for k, v in model_config.hyperparameters.items()},
            })
            mlflow.log_params({"class_weight": str(model_config.class_weight)})

            # Environment metadata
            env_meta = {
                "python": sys.version,
                "platform": platform.platform(),
                "packages": {k: v for k, v in [
                    ("numpy", __import__("numpy").__version__),
                    ("sklearn", __import__("sklearn").__version__),
                    ("mlflow", mlflow.__version__),
                ]},
            }
            mlflow.log_dict(env_meta, "environment.json")

            # Train
            model = create_model(model_config)
            model.feature_names_ = split.feature_names
            start = time.perf_counter()
            model.fit(split.x_train, split.y_train, split.x_val, split.y_val)
            training_duration = time.perf_counter() - start
            mlflow.log_metric("training_duration_s", training_duration)

            # Evaluate
            y_proba_val = model.predict_proba(split.x_val)
            optimal_threshold = find_optimal_threshold(split.y_val, y_proba_val, metric="f1")
            val_metrics = compute_metrics(split.y_val, y_proba_val, threshold=optimal_threshold)
            for k, v in metrics_to_dict(val_metrics).items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(f"val_{k}", v)

            y_proba_test = model.predict_proba(split.x_test)
            test_metrics = compute_metrics(split.y_test, y_proba_test, threshold=optimal_threshold)
            for k, v in metrics_to_dict(test_metrics).items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(f"test_{k}", v)

            mlflow.log_metric("optimal_threshold", optimal_threshold)

            # Save artifacts
            artifact_dir = self.settings.model_artifact_path / run.info.run_id
            artifact_dir.mkdir(parents=True, exist_ok=True)
            model_path = artifact_dir / "model.joblib"
            if model_type == ModelType.PYTORCH_MLP:
                model_path = artifact_dir / "model.pt"
            model.save(model_path)

            import joblib
            joblib.dump(preprocessor, artifact_dir / "preprocessor.joblib")

            config_artifact = {
                "experiment_config": self.config,
                "model": {
                    "type": model_type.value,
                    "hyperparameters": model_config.hyperparameters,
                    "class_weight": str(model_config.class_weight),
                },
                "model_config": asdict(model_config),
                "metadata": asdict(metadata),
                "optimal_threshold": optimal_threshold,
            }
            with (artifact_dir / "config.json").open("w") as f:
                json.dump(config_artifact, f, indent=2, default=str)

            mlflow.log_artifacts(str(artifact_dir))

            # Lineage
            lineage = LineageTracker()
            lineage.record(
                dataset_version=metadata.version,
                preprocessing_version=preprocessor.version,
                feature_version=preprocessor.version,
                experiment_id=run.info.run_id,
                model_artifact=str(model_path),
                git_commit=git_commit,
                config=self.config,
            )

            return {
                "run_id": run.info.run_id,
                "model_path": str(model_path),
                "val_metrics": metrics_to_dict(val_metrics),
                "test_metrics": metrics_to_dict(test_metrics),
                "training_duration_s": training_duration,
                "optimal_threshold": optimal_threshold,
            }
