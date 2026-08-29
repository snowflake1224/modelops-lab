"""Dataset adapter interface for pluggable data sources."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class DatasetMetadata:
    name: str
    version: str
    source: str
    target_column: str
    feature_columns: list[str]
    row_count: int
    positive_rate: float
    checksum: str
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    extra: dict[str, Any] = field(default_factory=dict)


class DatasetAdapter(ABC):
    """Abstract adapter for tabular classification datasets."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Dataset identifier."""

    @abstractmethod
    def download(self, output_dir: Path) -> Path:
        """Download or materialize raw data to output_dir."""

    @abstractmethod
    def load(self, path: Path) -> pd.DataFrame:
        """Load raw data from path."""

    @abstractmethod
    def validate(self, df: pd.DataFrame) -> list[str]:
        """Return list of validation errors (empty if valid)."""

    @abstractmethod
    def get_metadata(self, df: pd.DataFrame) -> DatasetMetadata:
        """Extract dataset metadata."""

    def ingest(self, output_dir: Path, force: bool = False) -> tuple[pd.DataFrame, DatasetMetadata]:
        output_dir.mkdir(parents=True, exist_ok=True)
        raw_path = output_dir / f"{self.name}.parquet"
        if not raw_path.exists() or force:
            downloaded = self.download(output_dir)
            df = self.load(downloaded)
            df.to_parquet(raw_path, index=False)
        else:
            df = pd.read_parquet(raw_path)
        errors = self.validate(df)
        if errors:
            raise ValueError(f"Dataset validation failed: {errors}")
        metadata = self.get_metadata(df)
        return df, metadata
