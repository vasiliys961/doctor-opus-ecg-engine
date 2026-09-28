from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.app import app
from backend.vision import UnsupportedImage, prepare_image
from ecg_engine.ensemble import get_ensemble, normalize, predict_features, service_payload
from ecg_engine.feature_columns import FEATURE_COLUMNS
from ecg_engine.raw_extractor import RAW_TO_531_STATUS, RawECGExtractor
from ecg_engine.schema import ECGSchemaError, read_ecg531_csv, read_feature_mapping

FIXTURE = "tests/fixtures/another_ecg_features.csv"
MODEL_DIR = "models/ecg_ensemble"


def test_pipeline_contract():
    sample = read_ecg531_csv(FIXTURE)
    assert sample.values.shape == (531,)
    frame = pd.read_csv(FIXTURE, nrows=0)
    assert [column for column in frame.columns if column != "ecg_id"] == list(FEATURE_COLUMNS)
    ensemble = get_ensemble(MODEL_DIR)
    assert set(ensemble.models) == {"MLP", "CNN", "ResNet"}
    normalized = normalize(sample.values, ensemble.mean, ensemble.std)
    expected = (sample.values - ensemble.mean.reshape(-1)) / ensemble.std.reshape(-1)
    np.testing.assert_allclose(normalized, expected)
    prediction = ensemble.predict(sample)
    assert prediction.ensemble.shape == (24,)
    assert prediction.heads["MLP"].shape == (24,)
    assert prediction.heads["CNN"].shape == (24,)
    assert prediction.heads["ResNet"].shape == (24,)
    assert np.isfinite(prediction.ensemble).all()
    manual = np.mean(
        [prediction.heads["MLP"], prediction.heads["CNN"], prediction.heads["ResNet"]],
        axis=0,
    )
    np.testing.assert_allclose(prediction.ensemble, manual)


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


def test_predict_api_returns_24_labeled_scores():
    frame = pd.read_csv(FIXTURE)
    features = {column: float(frame.iloc[0][column]) for column in FEATURE_COLUMNS}
    payload = service_payload(predict_features(features))
    with TestClient(app) as client:
        response = client.post("/api/ecg/predict", json={"features": features})
    assert response.status_code == 200
    body = response.json()
    assert body["feature_count"] == 531
    assert body["model_version"] == payload["model_version"]
    assert [item["scp_code"] for item in body["predictions"]] == [
        item["scp_code"] for item in payload["predictions"]
    ]
    assert len(body["models"]["mlp"]) == 24
    assert len(body["models"]["cnn"]) == 24
    assert len(body["models"]["resnet1d"]) == 24
    assert len(body["models"]["ensemble"]) == 24
    assert body["predictions"][0]["label_en"]
    assert body["predictions"][0]["label_ru"]
    np.testing.assert_allclose(body["models"]["ensemble"], payload["models"]["ensemble"])


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


def test_image_is_not_masked_or_reencoded_and_pdf_is_rejected():
    original = b"\x89PNG\r\n\x1a\nfake"
    returned, mime = prepare_image(original, "trace.png", "image/png")
    assert returned is original
    assert mime == "image/png"
    with pytest.raises(UnsupportedImage):
        prepare_image(b"%PDF-1.4", "sheet.pdf", "application/pdf")
    with TestClient(app) as client:
        response = client.post(
            "/api/ecg/image",
            files={"file": ("trace.png", original, "image/png")},
        )
    assert response.status_code == 503
    assert "LLM_API_KEY" in response.json()["detail"]
