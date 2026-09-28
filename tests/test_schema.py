from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ecg_engine.feature_columns import FEATURE_COLUMNS
from ecg_engine.schema import ECGSchemaError, read_ecg531_csv

FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "another_ecg_features.csv"


def test_fixture_header_is_the_canonical_order():
    frame = pd.read_csv(FIXTURE, nrows=0)
    actual = [column for column in frame.columns if column != "ecg_id"]
    assert actual == list(FEATURE_COLUMNS)
    assert len(FEATURE_COLUMNS) == 531


def test_first_row_is_accepted():
    sample = read_ecg531_csv(str(FIXTURE))
    assert sample.values.shape == (531,)
    assert sample.row_index == 0
    assert sample.rows_in_file == 1


def test_reordered_columns_are_rejected(tmp_path: Path):
    frame = pd.read_csv(FIXTURE)
    swapped = frame.copy()
    swapped.columns = list(swapped.columns)
    names = list(swapped.columns)
    left, right = names.index("PQ_Int_I"), names.index("PQ_Int_II")
    names[left], names[right] = names[right], names[left]
    swapped.columns = names
    path = tmp_path / "swapped.csv"
    swapped.to_csv(path, index=False)
    with pytest.raises(ECGSchemaError, match="не переставляются"):
        read_ecg531_csv(str(path))


def test_unknown_column_is_not_dropped(tmp_path: Path):
    frame = pd.read_csv(FIXTURE)
    frame["extra_feature"] = 0
    path = tmp_path / "extra.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ECGSchemaError, match="531"):
        read_ecg531_csv(str(path))


def test_missing_column_is_not_invented(tmp_path: Path):
    frame = pd.read_csv(FIXTURE).drop(columns=["HA__Global"])
    path = tmp_path / "missing.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ECGSchemaError, match="531"):
        read_ecg531_csv(str(path))


def test_nan_is_not_imputed(tmp_path: Path):
    frame = pd.read_csv(FIXTURE)
    frame.loc[0, "PQ_Int_I"] = None
    path = tmp_path / "nan.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ECGSchemaError, match="NaN не заполняется"):
        read_ecg531_csv(str(path))
