"""Слой совместимости на записи 00513. Ансамбль на сырой вектор не вызывается."""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.app import app
from ecg_engine.canonical_schema import FEATURE_COLUMNS_531, FEATURE_INDEX_531
from ecg_engine.feature_columns import FEATURE_COLUMNS
from ecg_engine.feature_compatibility import ECGInferenceAdapter, FeatureCompatibilityEngine
from ecg_engine.feature_registry import REGISTRY, split_feature
from ecg_engine.preprocessing import CANONICAL_LEADS, PreprocessingError, arrange_leads
from ecg_engine.raw_extractor import RAW_TO_531_STATUS
from ecg_engine.schema import ECGSchemaError, read_feature_mapping

ROOT = Path(__file__).resolve().parents[1]
SIGNAL = ROOT / "research/ecgdeli_reproduction/data/ptbxl/records500/00000/00513_hr.dat"
HEADER = ROOT / "research/ecgdeli_reproduction/data/ptbxl/records500/00000/00513_hr.hea"
COMPARISON = ROOT / "experiments/raw_to_531/comparison_fiducials_00513_v2.csv"
STEP12 = ROOT / "experiments/raw_to_531/step12_feature_status.csv"
FIXTURE = ROOT / "tests/fixtures/another_ecg_features.csv"
MATRIX = ROOT / "experiments/raw_to_531/step13_test_matrix.csv"
EXACT_LIMIT = 1e-6

_WRITER = ROOT / "experiments/raw_to_531/write_step13_status.py"
_spec = importlib.util.spec_from_file_location("write_step13_status", _WRITER)
_writer = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_writer)
write_step13_status = _writer.write

_INTERVALS = {
    "PQ_Int",
    "PR_Int",
    "QRS_Dur",
    "QT_Int",
    "P_DurFull",
    "T_DurFull",
    "P_Dur",
    "T_Dur",
    "RR_Mean",
    "QT_IntFramingham",
}
_AMPLITUDES = {"P_Amp", "Q_Amp", "R_Amp", "S_Amp", "T_Amp"}


def _published_record_available() -> bool:
    return SIGNAL.is_file() and HEADER.is_file()


def _record_00513() -> tuple[np.ndarray, list[str]]:
    header = HEADER.read_text().splitlines()[0].split()
    leads_in_file = 12
    sampling_declared = int(header[2])
    samples = int(header[3])
    assert sampling_declared == 500
    raw = np.fromfile(SIGNAL, dtype="<i2")
    assert raw.size == samples * leads_in_file
    return raw.reshape(samples, leads_in_file).astype(float) / 1000.0, list(CANONICAL_LEADS)


def _analysis_signal() -> tuple[np.ndarray, list[str]]:
    """Контракт слоя не зависит от gitignored-сигнала PTB-XL.

    Если локальная запись 00513 есть, проверяется она. Иначе достаточно
    конечной матрицы 12 отведений: числа признаков всё равно не считаются.
    """
    if _published_record_available():
        return _record_00513()
    return np.linspace(-0.2, 0.2, 20 * 12, dtype=float).reshape(20, 12), list(CANONICAL_LEADS)


def _family_status(family_names: set[str]) -> str:
    statuses = {row["status"] for row in REGISTRY if row["family"] in family_names}
    assert len(statuses) == 1
    return next(iter(statuses))


def test_canonical_order_matches_training_columns():
    assert FEATURE_COLUMNS_531 == FEATURE_COLUMNS
    assert len(FEATURE_COLUMNS_531) == 531
    assert len(set(FEATURE_COLUMNS_531)) == 531
    assert FEATURE_INDEX_531["PQ_Int_I"] == 0
    assert FEATURE_INDEX_531["HA__Global_count"] == 530


def test_registry_family_counts():
    families = [row["family"] for row in REGISTRY]
    assert sum(name in _INTERVALS for name in families) == 240
    assert sum(name in _AMPLITUDES for name in families) == 180
    assert families.count("P_Morph") == 36
    assert families.count("QT_IntCorr") == 36
    assert families.count("ST_Elev") == 36
    assert families.count("HA") == 3
    assert all(row["exact_proven"] is False for row in REGISTRY)
    assert _family_status(_INTERVALS) == "RECONSTRUCTED"
    assert _family_status(_AMPLITUDES) == "RECONSTRUCTED"
    assert _family_status({"QT_IntCorr"}) == "RECONSTRUCTED"
    assert _family_status({"P_Morph"}) == "NOT_EXECUTED"
    assert _family_status({"ST_Elev"}) == "UNKNOWN"
    assert _family_status({"HA"}) == "UNKNOWN"


def test_record_00513_contract():
    signal, leads = _analysis_signal()
    if _published_record_available():
        assert signal.shape == (5000, 12)
    else:
        assert signal.shape == (20, 12)
    assert leads == list(CANONICAL_LEADS)
    shuffled_names = list(reversed(leads))
    shuffled = signal[:, ::-1]
    ordered, record = arrange_leads(shuffled, shuffled_names, 500)
    np.testing.assert_allclose(ordered, signal)
    assert record.sampling_rate_input == 500
    assert record.sampling_rate_processing == 500
    assert record.filter_config == "none"
    result = FeatureCompatibilityEngine().analyze(signal, 500, leads)
    assert result["feature_count"] == 531
    assert list(result["analysis"]["feature_names"]) == list(FEATURE_COLUMNS_531)
    assert len(result["features"]) == 531
    assert len(result["provenance"]) == 531
    assert len(result["feature_mask"]) == 531
    assert len(result["feature_status"]) == 531
    assert result["raw_to_531_status"] == "NOT_PROVEN"
    assert result["status"] == "RECONSTRUCTED_RAW_MODE"
    assert result["prediction"] is None
    assert set(result["features"]) == {None}
    assert set(result["feature_mask"]) == {0}
    assert RAW_TO_531_STATUS == "NOT_PROVEN"


def test_missing_lead_fails():
    signal, leads = _analysis_signal()
    with pytest.raises(PreprocessingError, match="FAIL"):
        FeatureCompatibilityEngine().analyze(signal[:, :11], 500, leads[:11])


def test_pmorph_status_00513():
    signal, leads = _analysis_signal()
    result = FeatureCompatibilityEngine().analyze(signal, 500, leads)
    morph = [
        status
        for name, status in zip(FEATURE_COLUMNS_531, result["feature_status"])
        if split_feature(name)[0] == "P_Morph"
    ]
    assert len(morph) == 36
    assert set(morph) == {"NOT_EXECUTED"}


def test_st_elev_status():
    assert _family_status({"ST_Elev"}) == "UNKNOWN"
    assert all(row["status"] == "UNKNOWN" for row in REGISTRY if row["family"] == "ST_Elev")


def test_ha_global_status():
    assert _family_status({"HA"}) == "UNKNOWN"
    assert all(row["status"] == "UNKNOWN" for row in REGISTRY if row["family"] == "HA")


def test_forensic_exact_cells_only():
    frame = pd.read_csv(COMPARISON)
    exact = frame[frame["status"] == "EXACT"]
    assert len(exact) == 238
    assert (exact["abs_error"].abs() <= EXACT_LIMIT).all()
    framingham = frame[frame["feature"].isin(["QT_IntFramingham_Global", "QT_IntFramingham_Global_iqr"])]
    assert (framingham["abs_error"] > EXACT_LIMIT).all()
    step12 = pd.read_csv(STEP12)
    qt = step12[step12["family"] == "QT_IntCorr"]
    measured = qt[qt["abs_error"].notna()]
    assert not (measured["abs_error"] <= EXACT_LIMIT).all()


def test_raw_vector_is_rejected_by_ensemble_schema():
    signal, leads = _analysis_signal()
    result = FeatureCompatibilityEngine().analyze(signal, 500, leads)
    features = {name: value for name, value in zip(FEATURE_COLUMNS_531, result["features"])}
    with pytest.raises(ECGSchemaError):
        read_feature_mapping(features)


def test_raw_features_endpoint_has_no_prediction():
    signal, leads = _analysis_signal()
    with TestClient(app) as client:
        response = client.post(
            "/api/ecg/raw/features",
            json={"signal": signal.tolist(), "sampling_rate": 500, "lead_names": leads},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "RECONSTRUCTED_RAW_MODE"
    assert body["raw_to_531_status"] == "NOT_PROVEN"
    assert body["feature_count"] == 531
    assert body["prediction"] is None
    assert "predictions" not in body
    assert len(body["feature_mask"]) == 531
    assert len(body["feature_status"]) == 531
    assert len(body["features"]) == 531
    assert len(body["provenance"]) == 531


def test_adapter_is_not_trained():
    with pytest.raises(NotImplementedError, match="ADAPTER_NOT_TRAINED"):
        ECGInferenceAdapter().predict([0.0], [1])


def test_step13_artifacts_and_matrix():
    write_step13_status()
    summary = json.loads((ROOT / "experiments/raw_to_531/step13_summary.json").read_text())
    assert summary["raw_to_531_status"] == "NOT_PROVEN"
    assert summary["raw_inference"]["supported"] is False
    assert summary["total_features"] == 531
    status = pd.read_csv(ROOT / "experiments/raw_to_531/step13_feature_status.csv")
    assert list(status["feature"]) == list(FEATURE_COLUMNS_531)
    assert len(status) == 531
    unknown = status[status["status"].isin(["UNKNOWN", "NOT_EXECUTED"])]
    assert (unknown["value_00513"].isna() | (unknown["value_00513"] == "")).all()
    rows = [
        ("00513-intervals", "intervals", "RECONSTRUCTED", _family_status(_INTERVALS)),
        ("00513-amplitudes", "amplitudes", "RECONSTRUCTED", _family_status(_AMPLITUDES)),
        ("00513-pmorph", "P_Morph", "NOT_EXECUTED", _family_status({"P_Morph"})),
        ("00513-qt", "QT_IntCorr", "RECONSTRUCTED", _family_status({"QT_IntCorr"})),
        ("00513-st", "ST_Elev", "UNKNOWN", _family_status({"ST_Elev"})),
        ("00513-ha", "HA_Global", "UNKNOWN", _family_status({"HA"})),
    ]
    with MATRIX.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["test_id", "record", "feature_family", "expected_status", "actual_status", "passed", "notes"],
        )
        writer.writeheader()
        for test_id, family, expected, actual in rows:
            writer.writerow(
                {
                    "test_id": test_id,
                    "record": "00513",
                    "feature_family": family,
                    "expected_status": expected,
                    "actual_status": actual,
                    "passed": str(expected == actual).lower(),
                    "notes": "registry status; values are not claimed exact",
                }
            )
    saved = pd.read_csv(MATRIX, dtype=str)
    assert (saved["passed"] == "true").all()
    assert len(saved) == 6
