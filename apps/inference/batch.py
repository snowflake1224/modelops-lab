"""Batch inference pipeline."""

import json
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from apps.inference.engine import InferenceEngine
from packages.config.settings import get_settings


@dataclass
class BatchRunMetadata:
    run_id: str
    input_path: str
    output_path: str
    model_version: str
    n_records: int
    n_failures: int
    started_at: str
    completed_at: str
    status: str


class BatchInference:
    def __init__(self):
        settings = get_settings()
        self.engine = InferenceEngine(settings.data_dir / "registry")
        self.output_dir = settings.data_dir / "batch_outputs"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self, input_path: Path, model_version: str | None = None) -> BatchRunMetadata:
        run_id = str(uuid.uuid4())[:8]
        started = datetime.now(UTC).isoformat()

        self.engine.load(version=model_version)
        df = pd.read_parquet(input_path) if input_path.suffix == ".parquet" else pd.read_csv(input_path)

        predictions = []
        probabilities = []
        failures = 0

        batch_size = 500
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i : i + batch_size]
            try:
                features = batch.to_dict(orient="records")
                result = self.engine.predict(features, request_id=run_id)
                predictions.extend(result.predictions)
                probabilities.extend(result.probabilities)
            except Exception:
                failures += len(batch)
                predictions.extend([-1] * len(batch))
                probabilities.extend([0.0] * len(batch))

        df["prediction"] = predictions
        df["probability"] = probabilities
        output_path = self.output_dir / f"batch_{run_id}.parquet"
        df.to_parquet(output_path, index=False)

        metadata = BatchRunMetadata(
            run_id=run_id,
            input_path=str(input_path),
            output_path=str(output_path),
            model_version=self.engine.model_version,
            n_records=len(df),
            n_failures=failures,
            started_at=started,
            completed_at=datetime.now(UTC).isoformat(),
            status="completed" if failures == 0 else "completed_with_errors",
        )

        meta_path = self.output_dir / f"batch_{run_id}_meta.json"
        meta_path.write_text(json.dumps(asdict(metadata), indent=2))
        return metadata
