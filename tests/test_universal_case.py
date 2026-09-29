from __future__ import annotations

from fastapi.testclient import TestClient

from backend.app import app
from backend.vision import ANALYZER_MODEL, EYES_MODEL, analyze_case


def test_models_are_gemini_eyes_and_opus_analyzer():
    assert EYES_MODEL == "google/gemini-3.8-flash"
    assert ANALYZER_MODEL == "anthropic/claude-opus-5.5"


def test_empty_case_is_rejected():
    with TestClient(app) as client:
        response = client.post("/api/ecg/analyze", data={"notes": "  ", "clinical_context": ""})
    assert response.status_code == 400
    assert "изображение" in response.json()["detail"]


def test_text_goes_eyes_then_analyzer_without_ensemble(monkeypatch):
    calls = []

    def fake_completion(model, content):
        calls.append((model, content if isinstance(content, str) else content[0]["text"]))
        if model == EYES_MODEL:
            return '{"schema_version":"ecg.vision.v1","measurements":{"heart_rate":70},"findings":[]}'
        return "**Impression:**\nритм по тексту"

    monkeypatch.setattr("backend.vision._completion", fake_completion)
    result = analyze_case(notes="ЧСС 70 уд/мин", clinical_context="без боли")
    assert result["input_kind"] == "text"
    assert result["ensemble_used"] is False
    assert result["eyes_model"] == EYES_MODEL
    assert result["analyzer_model"] == ANALYZER_MODEL
    assert [model for model, _text in calls] == [EYES_MODEL, ANALYZER_MODEL]
    assert "ЧСС 70" in calls[0][1]
    assert "heart_rate" in calls[1][1]
    assert "Что видно" in calls[1][1]
    assert result["extraction"]["measurements"]["heart_rate"] == 70
    assert "ритм по тексту" in result["interpretation"]


def test_protocol_is_a_cheap_standard_form(monkeypatch):
    calls = []

    def fake_completion(model, content):
        calls.append((model, content))
        return (
            "Ритм\nФибрилляция желудочков.\n"
            "Дифференциальный диагноз\nАртефакт записи.\n"
            "Рекомендации\nНачать реанимацию и дефибрилляцию.\n"
            "Заключение\nФибрилляция желудочков."
        )

    monkeypatch.setattr("backend.vision._completion", fake_completion)
    with TestClient(app) as client:
        empty = client.post("/api/ecg/protocol", json={"interpretation": "  "})
        ready = client.post(
            "/api/ecg/protocol",
            json={"interpretation": "Ритм: фибрилляция желудочков.", "extraction": {"findings": ["VF"]}},
        )
    body = ready.json()["protocol"]
    assert empty.status_code == 400
    assert ready.status_code == 200
    assert calls[0][0] == EYES_MODEL
    assert "Ритм" in calls[0][1]
    assert "Миннесота" in calls[0][1]
    assert "Minnesota Code" in calls[0][1]
    assert "Интервалы" in calls[0][1]
    assert "Патологические зубцы" in calls[0][1]
    assert "Дифференциальный диагноз" not in body
    assert "Артефакт" not in body
    assert "реанимац" not in body.lower()
    assert "Фибрилляция желудочков." in body


def test_signal_conclusion_uses_gemini_protocol_form(monkeypatch):
    calls = []

    def fake_completion(model, content):
        calls.append((model, content))
        return (
            "Ритм\nСинусовый ритм, 64 уд/мин.\n"
            "Заключение\nСинусовый ритм. ECGFounder: SINUS RHYTHM 0.999.\n"
            "Дифференциальный диагноз\nАртефакт.\n"
            "Рекомендации\nНачать реанимацию."
        )

    monkeypatch.setattr("backend.vision._completion", fake_completion)
    payload = {
        "measurements": {
            "available": True,
            "description": "ЧСС 64 уд/мин.",
            "lines": [{"name": "ЧСС", "value": "64 уд/мин"}],
            "leads": [],
            "amplitudes": [],
        },
        "scores": [{"label": "SINUS RHYTHM", "score": 0.999}],
    }
    with TestClient(app) as client:
        empty = client.post("/api/ecg/signal/conclusion", json={"measurements": {"available": False}, "scores": []})
        ready = client.post("/api/ecg/signal/conclusion", json=payload)
    body = ready.json()["protocol"]
    assert empty.status_code == 400
    assert ready.status_code == 200
    assert calls[0][0] == EYES_MODEL
    assert "SINUS RHYTHM" in calls[0][1]
    assert "64 уд/мин" in calls[0][1]
    assert "Миннесота" in calls[0][1]
    assert "8-3-1" in calls[0][1]
    assert "Важные соотношения" in calls[0][1]
    assert "Депрессии и элевации" in calls[0][1]
    assert "Ритм" in body
    assert "SINUS RHYTHM" in body
    assert "Дифференциальный диагноз" not in body
    assert "реанимац" not in body.lower()


def test_analyze_api_reports_missing_key(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with TestClient(app) as client:
        response = client.post("/api/ecg/analyze", data={"notes": "QRS 90 мс"})
    assert response.status_code == 503
    assert "LLM_API_KEY" in response.json()["detail"]
