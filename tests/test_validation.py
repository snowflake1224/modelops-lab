"""Tests for data validation."""

import pandas as pd
import pytest

from packages.data.validation import validate_schema, validate_no_leakage


def test_validate_schema_pass():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "target": [0, 1]})
    report = validate_schema(df, ["a", "b", "target"])
    assert report.passed


def test_validate_schema_fail():
    df = pd.DataFrame({"a": [1, 2]})
    report = validate_schema(df, ["a", "b"])
    assert not report.passed


def test_no_leakage():
    report = validate_no_leakage(["a", "b"], ["a", "b"], "target")
    assert report.passed

    report = validate_no_leakage(["a", "b", "target"], ["a", "b"], "target")
    assert not report.passed
