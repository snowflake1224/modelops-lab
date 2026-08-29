"""Data validation utilities."""

from dataclasses import dataclass

import pandas as pd


@dataclass
class ValidationReport:
    passed: bool
    errors: list[str]
    warnings: list[str]
    stats: dict


def validate_schema(df: pd.DataFrame, expected_columns: list[str]) -> ValidationReport:
    errors: list[str] = []
    warnings: list[str] = []
    missing = set(expected_columns) - set(df.columns)
    extra = set(df.columns) - set(expected_columns)
    if missing:
        errors.append(f"Missing columns: {sorted(missing)}")
    if extra:
        warnings.append(f"Extra columns: {sorted(extra)}")
    stats = {
        "row_count": len(df),
        "column_count": len(df.columns),
        "null_counts": df.isnull().sum().to_dict(),
    }
    return ValidationReport(passed=len(errors) == 0, errors=errors, warnings=warnings, stats=stats)


def validate_no_leakage(
    train_cols: list[str], test_cols: list[str], target: str
) -> ValidationReport:
    errors: list[str] = []
    if target in train_cols:
        errors.append("Target column found in feature list (leakage risk)")
    if set(train_cols) != set(test_cols):
        errors.append("Train and test feature columns differ")
    return ValidationReport(passed=len(errors) == 0, errors=errors, warnings=[], stats={})
