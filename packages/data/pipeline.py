"""Train/val/test splitting with reproducibility."""

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

from packages.data.adapter import DatasetMetadata
from packages.data.datasets.credit_card_fraud import get_adapter
from packages.data.validation import validate_schema
from packages.features.preprocessing import FeaturePreprocessor, PreprocessorConfig


@dataclass
class DataSplit:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    metadata: DatasetMetadata
    preprocessor_version: str


class DataPipeline:
    def __init__(
        self,
        dataset_name: str = "credit_card_fraud",
        seed: int = 42,
        test_size: float = 0.15,
        val_size: float = 0.15,
    ):
        self.dataset_name = dataset_name
        self.seed = seed
        self.test_size = test_size
        self.val_size = val_size
        self.adapter = get_adapter(dataset_name)

    def run(
        self,
        data_dir: Path,
        preprocessor_config: PreprocessorConfig | None = None,
        force_download: bool = False,
    ) -> tuple[DataSplit, FeaturePreprocessor, DatasetMetadata]:
        raw_dir = data_dir / "raw"
        processed_dir = data_dir / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)

        df, metadata = self.adapter.ingest(raw_dir, force=force_download)
        target = metadata.target_column
        feature_cols = metadata.feature_columns

        report = validate_schema(df, feature_cols + [target])
        if not report.passed:
            raise ValueError(report.errors)

        # Stratified split: train / temp -> val / test
        sss1 = StratifiedShuffleSplit(n_splits=1, test_size=self.test_size + self.val_size, random_state=self.seed)
        train_idx, temp_idx = next(sss1.split(df, df[target]))

        temp_df = df.iloc[temp_idx]
        relative_val = self.val_size / (self.test_size + self.val_size)
        sss2 = StratifiedShuffleSplit(n_splits=1, test_size=1 - relative_val, random_state=self.seed)
        val_idx, test_idx = next(sss2.split(temp_df, temp_df[target]))

        train_df = df.iloc[train_idx]
        val_df = temp_df.iloc[val_idx]
        test_df = temp_df.iloc[test_idx]

        config = preprocessor_config or PreprocessorConfig()
        preprocessor = FeaturePreprocessor(config)
        preprocessor.fit(train_df[feature_cols], train_df[target])

        x_train = preprocessor.transform(train_df[feature_cols])
        x_val = preprocessor.transform(val_df[feature_cols])
        x_test = preprocessor.transform(test_df[feature_cols])

        split = DataSplit(
            x_train=x_train,
            y_train=train_df[target].values,
            x_val=x_val,
            y_val=val_df[target].values,
            x_test=x_test,
            y_test=test_df[target].values,
            feature_names=preprocessor.feature_names_,
            metadata=metadata,
            preprocessor_version=preprocessor.version,
        )

        artifact_path = processed_dir / f"{self.dataset_name}_{metadata.version}"
        artifact_path.mkdir(parents=True, exist_ok=True)
        joblib.dump(preprocessor, artifact_path / "preprocessor.joblib")
        np.savez_compressed(
            artifact_path / "splits.npz",
            x_train=x_train,
            y_train=split.y_train,
            x_val=x_val,
            y_val=split.y_val,
            x_test=x_test,
            y_test=split.y_test,
        )
        return split, preprocessor, metadata
