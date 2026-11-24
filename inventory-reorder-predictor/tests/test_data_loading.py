import pandas as pd
import pytest

from src.data_preprocessing import REQUIRED_COLUMNS, load_raw_data


def test_generated_data_has_required_columns(small_raw):
    assert set(REQUIRED_COLUMNS).issubset(small_raw.columns)
    assert len(small_raw) > 500


def test_load_valid_csv(tmp_path, small_raw):
    path = tmp_path / "data.csv"
    small_raw.to_csv(path, index=False)
    df = load_raw_data(path)
    assert len(df) == len(small_raw)


def test_missing_file_gives_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="generate_data"):
        load_raw_data(tmp_path / "nope.csv")


def test_missing_column_is_rejected(tmp_path, small_raw):
    path = tmp_path / "bad.csv"
    small_raw.drop(columns=["units_sold"]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="units_sold"):
        load_raw_data(path)
