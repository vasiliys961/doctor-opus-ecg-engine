from __future__ import annotations

import pytest

from benchmark.split import SplitLeakageError, assert_disjoint, load_frames


def test_committed_split_has_no_patient_or_record_overlap():
    frames = load_frames()
    assert_disjoint(frames)
    train_patients = {row["patient_id"] for row in frames["train"]}
    val_patients = {row["patient_id"] for row in frames["val"]}
    test_patients = {row["patient_id"] for row in frames["test"]}
    assert train_patients.isdisjoint(val_patients)
    assert train_patients.isdisjoint(test_patients)
    assert val_patients.isdisjoint(test_patients)
    train_ids = {row["ecg_id"] for row in frames["train"]}
    val_ids = {row["ecg_id"] for row in frames["val"]}
    test_ids = {row["ecg_id"] for row in frames["test"]}
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)
    assert "513" not in train_ids | val_ids | test_ids
    assert "00513" not in train_ids | val_ids | test_ids


def test_overlap_stops_the_benchmark():
    frames = {
        "train": [{"ecg_id": "1", "patient_id": "10", "split": "train"}],
        "val": [{"ecg_id": "2", "patient_id": "10", "split": "val"}],
        "test": [{"ecg_id": "3", "patient_id": "11", "split": "test"}],
    }
    with pytest.raises(SplitLeakageError):
        assert_disjoint(frames)
