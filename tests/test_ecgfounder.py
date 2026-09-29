from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app import app
from ecg_engine.ecgfounder import SignalError, arrange, load_labels, to_model_input
from ecg_engine.ecgfounder_net import build_ecgfounder

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def test_labels_match_published_head():
    assert len(load_labels()) == 150
    assert load_labels()[0] == "ABNORMAL ECG"
    assert load_labels()[-1] == "MULTIFOCAL ATRIAL TACHYCARDIA"


def test_median_beat_is_rejected():
    signal = np.zeros((12, 600))
    with pytest.raises(SignalError, match="600"):
        to_model_input(signal, 500)


def test_lead_order_is_canonical_not_file_order():
    swapped = ["I", "II", "III", "aVR", "aVF", "aVL", "V1", "V2", "V3", "V4", "V5", "V6"]
    signal = np.arange(12, dtype=float)[:, None] * np.ones((1, 20))
    arranged = arrange(signal, swapped)
    assert arranged.shape == (12, 20)
    np.testing.assert_allclose(arranged[4], signal[5])
    np.testing.assert_allclose(arranged[5], signal[4])


def test_ten_seconds_at_1000_hz_becomes_5000_samples():
    prepared = to_model_input(np.ones((12, 10000)), 1000)
    assert prepared.shape == (12, 5000)
    assert abs(float(prepared.mean())) < 1e-6


def test_csv_route_rejects_short_recording():
    header = ",".join(LEADS)
    row = ",".join(["0"] * 12)
    payload = "\n".join([header, row])
    with TestClient(app) as client:
        response = client.post(
            "/api/ecg/signal/csv",
            files={"file": ("trace.csv", payload, "text/csv")},
            data={"sampling_rate": "500"},
        )
    assert response.status_code == 400
    assert "10 секунд" in response.json()["detail"]


def test_signal_route_returns_founder_scores_without_a_language_model(monkeypatch):
    monkeypatch.setattr(
        "backend.app.predict_signal",
        lambda signal, lead_names, sampling_rate: {
            "model": "ECGFounder",
            "scores": [{"label": "SINUS RHYTHM", "score": 0.91}],
            "note": "проверка маршрута",
        },
    )
    header = ",".join(LEADS)
    row = ",".join(["0"] * 12)
    payload = "\n".join([header] + [row] * 5000)
    with TestClient(app) as client:
        response = client.post(
            "/api/ecg/signal/csv",
            files={"file": ("trace.csv", payload, "text/csv")},
            data={"sampling_rate": "500"},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "ECGFounder"
    assert body["scores"][0]["score"] == 0.91
    assert body["measurements"]["available"] is False
    assert "gemini_reading" not in body


def test_example_route_reads_the_sample_csv(monkeypatch, tmp_path):
    sample = tmp_path / "sample.csv"
    header = ",".join(LEADS)
    row = ",".join(["0"] * 12)
    sample.write_text("\n".join([header] + [row] * 5000), encoding="utf-8")
    monkeypatch.setattr("backend.app.EXAMPLE_CSV", sample)
    monkeypatch.setattr(
        "backend.app.predict_signal",
        lambda signal, lead_names, sampling_rate: {
            "model": "ECGFounder",
            "scores": [{"label": "SINUS RHYTHM", "score": 0.5}],
            "note": "проверка примера",
        },
    )
    with TestClient(app) as client:
        ready = client.post("/api/ecg/signal/example", json={"sampling_rate": 500})
    assert ready.status_code == 200
    assert ready.json()["example"].startswith("PTB-XL 00001")
    assert ready.json()["scores"][0]["label"] == "SINUS RHYTHM"
    assert len(ready.json()["trace"]) == 12
    assert ready.json()["trace"][0]["lead"] == "I"
    monkeypatch.setattr("backend.app.EXAMPLE_CSV", tmp_path / "absent.csv")
    with TestClient(app) as client:
        missing = client.post("/api/ecg/signal/example", json={"sampling_rate": 500})
    assert missing.status_code == 404


def test_network_accepts_founder_shape():
    import torch

    model = build_ecgfounder(150)
    model.eval()
    with torch.no_grad():
        output = model(torch.zeros(1, 12, 5000))
    assert tuple(output.shape) == (1, 150)
