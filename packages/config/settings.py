"""Application settings loaded from environment."""

from functools import lru_cache
from pathlib import Path
from typing import ClassVar

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=("settings_",),
    )

    project_root: Path = PROJECT_ROOT
    database_url: str = "postgresql://modelops:modelops@localhost:5432/modelops"
    mlflow_tracking_uri: str = "sqlite:///data/mlflow.db"
    mlflow_experiment_name: str = "modelops-lab"
    model_stage: str = "production"
    model_name: str = "fraud-classifier"
    model_artifact_path: Path = PROJECT_ROOT / "data" / "artifacts" / "models"
    inference_host: str = "0.0.0.0"
    inference_port: int = 8000
    api_host: str = "0.0.0.0"
    api_port: int = 8080
    max_request_size_bytes: int = 1_048_576
    max_batch_size: int = 1000
    prometheus_port: int = 9464
    otel_service_name: str = "modelops-inference"
    drift_psi_threshold: float = 0.2
    drift_ks_threshold: float = 0.05
    secret_key: str = "change-me-in-production"
    allowed_origins: str = "http://localhost:3000"
    redis_url: str = "redis://localhost:6379/0"
    random_seed: int = 42
    git_commit: str = Field(default="unknown")

    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"

    @property
    def raw_data_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_data_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def configs_dir(self) -> Path:
        return self.project_root / "configs"


@lru_cache
def get_settings() -> Settings:
    return Settings()
