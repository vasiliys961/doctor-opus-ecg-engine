from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.app import app
from backend.vision import UnsupportedImage, prepare_image
from ecg_engine.feature_columns import FEATURE_COLUMNS
from ecg_engine.raw_extractor import RAW_TO_531_STATUS, RawECGExtractor
from ecg_engine.schema import ECGSchemaError, read_ecg531_csv, read_feature_mapping

FIXTURE = "tests/fixtures/another_ecg_features.csv"


def test_fixture_keeps_canonical_column_order():
    sample = read_ecg531_csv(FIXTURE)
    assert sample.values.shape == (531,)
    frame = pd.read_csv(FIXTURE, nrows=0)
    assert [column for column in frame.columns if column != "ecg_id"] == list(FEATURE_COLUMNS)


def test_named_features_match_csv_order():
    frame = pd.read_csv(FIXTURE)
    row = frame.iloc[0]
    features = {column: row[column] for column in FEATURE_COLUMNS}
    from_csv = read_ecg531_csv(FIXTURE)
    from_names = read_feature_mapping(features)
    np.testing.assert_array_equal(from_csv.values, from_names.values)


def test_unknown_feature_name_is_rejected():
    frame = pd.read_csv(FIXTURE)
    features = {column: frame.iloc[0][column] for column in FEATURE_COLUMNS}
    features["not_a_feature"] = 1
    with pytest.raises(ECGSchemaError, match="Неизвестные"):
        read_feature_mapping(features)


def test_product_api_does_not_serve_the_531_ensemble():
    frame = pd.read_csv(FIXTURE)
    features = {column: float(frame.iloc[0][column]) for column in FEATURE_COLUMNS}
    with TestClient(app) as client:
        response = client.post("/api/ecg/predict", json={"features": features})
        partial = client.post("/api/ecg/partial", json={"features": {"RR_Mean_Global": 800}})
    assert response.status_code == 404
    assert partial.status_code == 404


def test_raw_extractor_does_not_emit_features():
    extractor = RawECGExtractor()
    with pytest.raises(NotImplementedError):
        extractor.extract(b"raw")
    status = extractor.status()
    assert status["status"] == RAW_TO_531_STATUS
    assert status["features"] is None
    with TestClient(app) as client:
        response = client.post("/api/ecg/raw")
    assert response.status_code == 501
    assert response.json()["features"] is None


def test_image_is_not_masked_or_reencoded_and_pdf_is_rejected(monkeypatch):
    original = b"\x89PNG\r\n\x1a\nfake"
    returned, mime = prepare_image(original, "trace.png", "image/png")
    assert returned is original
    assert mime == "image/png"
    with pytest.raises(UnsupportedImage):
        prepare_image(b"%PDF-1.4", "sheet.pdf", "application/pdf")
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with TestClient(app) as client:
        response = client.post(
            "/api/ecg/image",
            files={"file": ("trace.png", original, "image/png")},
        )
    assert response.status_code == 503
    assert "LLM_API_KEY" in response.json()["detail"]
