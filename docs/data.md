# Data Pipeline

## Dataset

**Credit Card Fraud Detection** (OpenML dataset 1597)

- 284,807 transactions (`42fdf08c21a87bcd` checksum on OpenML export)
- **29 numerical features**: 28 anonymized PCA features (`V1`–`V28`) + `Amount` (no `Time` column in OpenML 1597 export)
- Binary target: `Class` (`0` = legitimate, `1` = fraud)
- ~0.173% fraud rate (highly imbalanced; 492 positive examples)

## Adapter Interface

```python
class DatasetAdapter(ABC):
    def download(self, output_dir: Path) -> Path: ...
    def load(self, path: Path) -> pd.DataFrame: ...
    def validate(self, df: pd.DataFrame) -> list[str]: ...
    def get_metadata(self, df: pd.DataFrame) -> DatasetMetadata: ...
    def ingest(self, output_dir: Path) -> tuple[pd.DataFrame, DatasetMetadata]: ...
```

New datasets implement this interface and register in `get_adapter()`.

## Pipeline Stages

```mermaid
graph LR
    A[Download] --> B[Validate Schema]
    B --> C[Stratified Split]
    C --> D[Fit Preprocessor on Train]
    D --> E[Transform All Splits]
    E --> F[Save Artifacts]
```

### Splitting Strategy

- Train: 70%
- Validation: 15%
- Test: 15%
- Stratified by target class
- Fixed random seed for reproducibility

## Preprocessing

| Step | Training | Inference |
|------|----------|-----------|
| Outlier clipping | Fit quantiles on train | Apply stored bounds |
| Feature interactions | Optional (top-5 variance pairs) | Apply same pairs |
| Scaling | RobustScaler fit on train | Transform with fitted scaler |

Preprocessor version is a hash of configuration, tracked in lineage.

## Data Versioning

Each dataset version includes:
- Checksum (SHA-256 of data)
- Row count and positive rate
- Feature column list
- Source identifier

## Leakage Risks

| Risk | Mitigation |
|------|-----------|
| Target in features | Explicit column exclusion + validation |
| Test data in preprocessing | Fit only on training split |
| Temporal leakage | N/A for this static dataset; time-based splits recommended for production |
| Duplicate transactions | Not applicable to PCA-transformed features |
