"""Optuna hyperparameter optimization."""

from pathlib import Path
from typing import Any

import mlflow
import optuna
import yaml

from packages.config.settings import get_settings
from packages.data.pipeline import DataPipeline
from packages.evaluation.metrics import compute_metrics, find_optimal_threshold
from packages.models.base import ModelConfig, ModelType, create_model


def get_search_space(trial: optuna.Trial, model_type: ModelType) -> dict[str, Any]:
    if model_type == ModelType.LIGHTGBM:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        }
    if model_type == ModelType.XGBOOST:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        }
    if model_type == ModelType.PYTORCH_MLP:
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
            "dropout": trial.suggest_float("dropout", 0.1, 0.5),
            "epochs": trial.suggest_int("epochs", 10, 50),
        }
    return {"C": trial.suggest_float("C", 0.01, 10.0, log=True)}


class HyperparameterOptimizer:
    def __init__(self, base_config_path: Path, n_trials: int = 20):
        with base_config_path.open() as f:
            loaded = yaml.safe_load(f)
        if not isinstance(loaded, dict):
            raise ValueError(f"Experiment config must be a mapping: {base_config_path}")
        self.base_config = loaded
        self.n_trials = n_trials
        self.settings = get_settings()

    def optimize(self) -> dict[str, Any]:
        seed = self.base_config.get("seed", 42)
        pipeline = DataPipeline(seed=seed)
        split, _, metadata = pipeline.run(self.settings.data_dir)

        model_type = ModelType(self.base_config["model"]["type"])
        mlflow.set_tracking_uri(self.settings.mlflow_tracking_uri)
        mlflow.set_experiment(self.base_config.get("experiment_name", "modelops-hpo"))

        def objective(trial: optuna.Trial) -> float:
            hp = get_search_space(trial, model_type)
            config = ModelConfig(
                model_type=model_type,
                hyperparameters=hp,
                class_weight=self.base_config["model"].get("class_weight", "balanced"),
                random_seed=seed,
            )
            model = create_model(config)
            model.fit(split.x_train, split.y_train, split.x_val, split.y_val)
            y_proba = model.predict_proba(split.x_val)
            threshold = find_optimal_threshold(split.y_val, y_proba)
            metrics = compute_metrics(split.y_val, y_proba, threshold=threshold, n_bootstrap=50)
            return metrics.pr_auc

        study = optuna.create_study(
            direction="maximize",
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
            study_name=f"hpo-{model_type.value}",
        )

        with mlflow.start_run(run_name=f"hpo-{model_type.value}"):
            study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
            mlflow.log_params(study.best_params)
            mlflow.log_metric("best_pr_auc", study.best_value)
            mlflow.log_dict(
                {"trials": [{"params": t.params, "value": t.value} for t in study.trials]},
                "hpo_results.json",
            )

        return {
            "best_params": study.best_params,
            "best_value": study.best_value,
            "n_trials": len(study.trials),
        }
