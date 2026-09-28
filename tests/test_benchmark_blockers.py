from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from benchmark.config import load_config
from benchmark.coverage import COVERAGE_STATEMENT, coverage_accounting, detailed_coverage_rows
from benchmark.dataset_locator import resolve_dataset
from benchmark.label_mapping import TARGET_CODES, validate_binary_targets
from benchmark.metrics import binary_auroc
from benchmark.raw_model.dataset import LazyWaveformDataset
from benchmark.raw_model.train import ensure_benchmark_device, pick_device
from benchmark.reference import classify_feature_table, column_order_status
from ecg_engine.feature_columns import FEATURE_COLUMNS


def test_config_data_paths_are_null_until_user_sets_them():
    config = load_config()
    assert config["data"]["ptbxl_root"] is None
    assert config["data"]["ecgdeli_features"] is None


def test_cli_overrides_environment_and_config(monkeypatch):
    config = {
        "data": {"ptbxl_root": "/yaml/root", "ecgdeli_features": "/yaml/features.csv"},
        "paths": {
            "ptbxl_database": "benchmark/data/ptbxl/ptbxl_database.csv",
            "records500": "benchmark/data/ptbxl/records500",
            "ecgdeli_features": "benchmark/data/ptbxl_plus/ecgdeli_features.csv",
        },
    }
    monkeypatch.setenv("PTBXL_ROOT", "/env/root")
    monkeypatch.setenv("ECGDELI_FEATURES", "/env/features.csv")
    from_env = resolve_dataset(config, search=False)
    assert from_env.root is not None
    assert from_env.root.path == Path("/env/root")
    assert from_env.features.path == Path("/env/features.csv")
    assert from_env.records.path == Path("/env/root/records500")
    from_cli = resolve_dataset(
        config,
        cli_root="/cli/root",
        cli_features="/cli/features.csv",
        search=False,
    )
    assert from_cli.root is not None
    assert from_cli.root.path == Path("/cli/root")
    assert from_cli.features.path == Path("/cli/features.csv")
    assert from_cli.metadata.path == Path("/cli/root/ptbxl_database.csv")


def test_missing_feature_table_is_not_a_model_error(tmp_path):
    report = classify_feature_table(tmp_path / "ecgdeli_features.csv")
    assert report["accepted"] is False
    assert report["reason"] == "NO_FEATURE_FILE"


def test_column_order_is_not_repaired_by_sorting():
    swapped = ["ecg_id", FEATURE_COLUMNS[1], FEATURE_COLUMNS[0], *FEATURE_COLUMNS[2:]]
    assert column_order_status(swapped) == "COLUMN_ORDER_MISMATCH"
    assert column_order_status(["ecg_id", *FEATURE_COLUMNS]) == "OK"


def test_checksum_mismatch_is_not_sent_to_ensemble(tmp_path):
    path = tmp_path / "ecgdeli_features.csv"
    path.write_text("ecg_id," + ",".join(FEATURE_COLUMNS) + "\n1," + ",".join(["0"] * 531) + "\n", encoding="utf-8")
    report = classify_feature_table(path)
    assert report["accepted"] is False
    assert report["reason"] == "OTHER"
    assert "checksum" in str(report["detail"])
    assert report["column_order"] == "OK"
    assert report["feature_columns"] == 531


def test_empty_class_auroc_is_undefined():
    truth = np.zeros(8)
    score = np.linspace(0.1, 0.9, 8)
    value = binary_auroc(truth, score)
    assert np.isnan(value)
    assert value != 0.5


def test_invalid_targets_are_rejected(capsys):
    labels = np.zeros((4, len(TARGET_CODES)))
    labels[0, 0] = 1
    warnings = validate_binary_targets(labels, "train")
    assert any("NORM" in line for line in warnings)
    assert "WARNING:" in capsys.readouterr().out
    broken = labels.copy()
    broken[0, 1] = 2
    with pytest.raises(ValueError):
        validate_binary_targets(broken, "train")
    broken[0, 1] = np.nan
    with pytest.raises(ValueError):
        validate_binary_targets(broken, "train")


def test_cpu_device_flag_stays_on_cpu():
    assert pick_device("cpu").type == "cpu"
    assert ensure_benchmark_device("cpu").type == "cpu"


def test_mps_runtime_error_falls_back_to_cpu(monkeypatch):
    if not torch.backends.mps.is_available():
        pytest.skip("MPS нет в этом окружении")
    import benchmark.raw_model.train as train

    monkeypatch.setattr(train, "pick_device", lambda requested=None: torch.device("mps"))

    class Boom(train.RawECGCNN):
        def forward(self, waveform):
            raise RuntimeError("mps boom")

    monkeypatch.setattr(train, "RawECGCNN", Boom)
    assert ensure_benchmark_device(None).type == "cpu"


def test_coverage_does_not_call_documented_methods_reconstructed():
    accounting = coverage_accounting()
    assert accounting["runtime"] == 0
    assert accounting["method_available"] == 456
    assert accounting["not_executed"] == 36
    assert accounting["unknown"] == 39
    assert "successfully" not in COVERAGE_STATEMENT
    rows = detailed_coverage_rows(published_reference_available=False)
    assert len(rows) == 531
    assert {row["runtime_status"] for row in rows} == {"not_available"}
    assert rows[0].keys() == {
        "feature",
        "family",
        "runtime_status",
        "method_status",
        "execution_status",
        "published_reference_available",
    }


def _write_record(directory: Path) -> Path:
    header = directory / "00000_hr.hea"
    names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    lines = ["00000_hr 12 500 5000"]
    lines.extend(f"00000_hr.dat 16 1000 16 0 0 0 0 {name}" for name in names)
    header.write_text("\n".join(lines) + "\n", encoding="utf-8")
    np.zeros(5000 * 12, dtype="<i2").tofile(directory / "00000_hr.dat")
    return header


def test_lazy_dataset_reads_one_record(tmp_path):
    header = _write_record(tmp_path)
    labels = np.zeros((1, 24), dtype=np.float32)
    dataset = LazyWaveformDataset([header], labels, np.zeros(12), np.ones(12))
    waveform, target = dataset[0]
    assert tuple(waveform.shape) == (12, 5000)
    assert tuple(target.shape) == (24,)
