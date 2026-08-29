"""ModelOps Lab CLI."""

from pathlib import Path
from typing import Optional

import typer
import yaml
from rich.console import Console
from rich.table import Table

from packages.config.settings import get_settings

app = typer.Typer(name="modelops", help="ModelOps Lab CLI")
console = Console()

data_app = typer.Typer(help="Data operations")
experiment_app = typer.Typer(help="Experiment operations")
model_app = typer.Typer(help="Model operations")
app.add_typer(data_app, name="data")
app.add_typer(experiment_app, name="experiment")
app.add_typer(model_app, name="model")


@data_app.command("validate")
def data_validate(dataset: str = "credit_card_fraud"):
    from packages.data.datasets.credit_card_fraud import get_adapter
    settings = get_settings()
    adapter = get_adapter(dataset)
    df, metadata = adapter.ingest(settings.raw_data_dir)
    console.print(f"[green]✓[/green] Dataset valid: {metadata.row_count} rows, {metadata.positive_rate:.4f} positive rate")


@data_app.command("prepare")
def data_prepare(seed: int = 42):
    from packages.data.pipeline import DataPipeline
    settings = get_settings()
    pipeline = DataPipeline(seed=seed)
    split, preprocessor, metadata = pipeline.run(settings.data_dir)
    console.print(f"[green]✓[/green] Data prepared: train={len(split.y_train)}, val={len(split.y_val)}, test={len(split.y_test)}")


@app.command("train")
def train(config: Path = typer.Argument(..., help="Path to experiment config YAML")):
    from packages.training.pipeline import TrainingPipeline
    pipeline = TrainingPipeline(config)
    result = pipeline.run()
    console.print(f"[green]✓[/green] Training complete. Run ID: {result['run_id']}")
    console.print(f"  Test PR-AUC: {result['test_metrics'].get('pr_auc', 'N/A'):.4f}")


@experiment_app.command("run")
def experiment_run(config: Path = typer.Argument(...)):
    train(config)


@experiment_app.command("list")
def experiment_list():
    import mlflow
    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = mlflow.tracking.MlflowClient()
    experiments = client.search_experiments()
    exp_ids = [e.experiment_id for e in experiments] or ["0"]
    runs = client.search_runs(experiment_ids=exp_ids, max_results=20)
    table = Table(title="Recent Experiments")
    table.add_column("Run ID")
    table.add_column("Name")
    table.add_column("PR-AUC")
    for run in runs:
        pr_auc = run.data.metrics.get("test_pr_auc", "N/A")
        table.add_row(run.info.run_id[:8], run.info.run_name or "", f"{pr_auc:.4f}" if isinstance(pr_auc, float) else str(pr_auc))
    console.print(table)


@experiment_app.command("compare")
def experiment_compare(exp1: str, exp2: str):
    import mlflow
    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = mlflow.tracking.MlflowClient()
    experiments = client.search_experiments()
    exp_ids = [e.experiment_id for e in experiments] or ["0"]
    runs = {r.info.run_id: r for r in client.search_runs(experiment_ids=exp_ids, max_results=100)}
    r1 = next((r for rid, r in runs.items() if rid.startswith(exp1)), None)
    r2 = next((r for rid, r in runs.items() if rid.startswith(exp2)), None)
    if not r1 or not r2:
        console.print("[red]Run not found[/red]")
        raise typer.Exit(1)
    table = Table(title="Experiment Comparison")
    table.add_column("Metric")
    table.add_column(exp1)
    table.add_column(exp2)
    for metric in ["test_pr_auc", "test_f1", "test_recall", "test_precision", "training_duration_s"]:
        v1 = r1.data.metrics.get(metric, "N/A")
        v2 = r2.data.metrics.get(metric, "N/A")
        table.add_row(metric, f"{v1:.4f}" if isinstance(v1, float) else str(v1), f"{v2:.4f}" if isinstance(v2, float) else str(v2))
    console.print(table)


@experiment_app.command("hpo")
def experiment_hpo(config: Path, n_trials: int = 10):
    from packages.training.hpo import HyperparameterOptimizer
    optimizer = HyperparameterOptimizer(config, n_trials=n_trials)
    result = optimizer.optimize()
    console.print(f"[green]✓[/green] Best PR-AUC: {result['best_value']:.4f}")
    console.print(f"  Best params: {result['best_params']}")


@model_app.command("register")
def model_register(run_id: str, version: str = "1.0.0"):
    from packages.registry.store import ModelRegistry
    from packages.training.pipeline import get_git_commit
    settings = get_settings()
    artifact_path = settings.model_artifact_path / run_id / "model.joblib"
    registry = ModelRegistry(settings.data_dir / "registry")
    mv = registry.register(
        name=settings.model_name,
        version=version,
        artifact_path=str(artifact_path),
        metrics={},
        config={"run_id": run_id},
        git_commit=get_git_commit(),
    )
    console.print(f"[green]✓[/green] Registered {mv.name}:{mv.version} in {mv.stage.value}")


@model_app.command("promote")
def model_promote(version: str, stage: str = "production"):
    from packages.registry.store import ModelRegistry, ModelStage
    settings = get_settings()
    registry = ModelRegistry(settings.data_dir / "registry")
    try:
        # Auto-validate if promoting to production
        if stage == "production":
            mv = registry.get(settings.model_name, version)
            if mv and not mv.validation_passed:
                registry.promote(settings.model_name, version, ModelStage.VALIDATION)
        mv = registry.promote(settings.model_name, version, ModelStage(stage))
        console.print(f"[green]✓[/green] Promoted to {mv.stage.value}")
    except Exception as e:
        console.print(f"[red]Promotion failed: {e}[/red]")
        raise typer.Exit(1) from e


@model_app.command("rollback")
def model_rollback():
    from packages.training.retrain import RetrainingPipeline
    pipeline = RetrainingPipeline()
    result = pipeline.rollback()
    console.print(f"[green]✓[/green] {result}")


@app.command("serve")
def serve(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run("apps.inference.main:app", host=host, port=port, reload=False)


@app.command("drift")
def drift_check(reference: Optional[Path] = None, current: Optional[Path] = None):
    import numpy as np
    from packages.drift.detector import detect_dataset_drift
    from packages.monitoring.store import MonitoringStore
    settings = get_settings()

    processed = settings.processed_data_dir / "credit_card_fraud_v1"
    splits = np.load(processed / "splits.npz")
    ref = splits["x_train"][:1000]
    cur = splits["x_test"][:1000]

    import joblib
    preprocessor = joblib.load(processed / "preprocessor.joblib")
    results = detect_dataset_drift(
        ref, cur, preprocessor.feature_names_,
        psi_threshold=settings.drift_psi_threshold,
        ks_threshold=settings.drift_ks_threshold,
    )
    drifted = [r for r in results if r.drift_detected]
    if drifted:
        store = MonitoringStore()
        store.create_event("drift", "warning", {"drifted_features": [r.feature for r in drifted[:5]]})
        console.print(f"[yellow]⚠[/yellow] Drift detected in {len(drifted)} features")
    else:
        console.print("[green]✓[/green] No significant drift detected")


@app.command("retrain")
def retrain_candidate(config: Optional[Path] = None):
    from packages.training.retrain import RetrainingPipeline
    pipeline = RetrainingPipeline()
    result = pipeline.create_candidate(config_path=config)
    console.print(f"[green]✓[/green] Candidate created: {result['version']}")
    validation = pipeline.validate_candidate(result["version"])
    console.print(f"  Validation: {'PASSED' if validation['passed'] else 'FAILED'}")


@app.command("batch")
def batch_infer(input_path: Path, version: Optional[str] = None):
    from apps.inference.batch import BatchInference
    runner = BatchInference()
    result = runner.run(input_path, model_version=version)
    console.print(f"[green]✓[/green] Batch complete: {result.n_records} records, {result.n_failures} failures")


if __name__ == "__main__":
    app()
