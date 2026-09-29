from __future__ import annotations

import json
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app import app
from ecg_engine.ecgfounder import LEADS
from ecg_engine.raw_extractor import RAW_TO_531_STATUS
from ecg_engine.recording_table import RecordingTableError, to_csv

ROOT = Path(__file__).resolve().parents[1]
CABLE_JS = ROOT / "frontend" / "cable.js"


def _columns() -> list[list[float]]:
    return [[float(lead) - 0.115 + index / 10 for index in range(4)] for lead in range(12)]


def test_table_matches_the_csv_contract():
    text = to_csv(_columns())
    header, *rows = [line for line in text.splitlines() if line]
    assert header.split(",") == list(LEADS)
    assert len(rows) == 4
    parsed = [[float(cell) for cell in row.split(",")] for row in rows]
    np.testing.assert_allclose(np.array(parsed).T, _columns())


def test_table_rejects_bytes_and_a_short_grid():
    with pytest.raises(RecordingTableError, match="не байты"):
        to_csv(b"\x55\xaa\x00")
    with pytest.raises(RecordingTableError, match="12 отведений"):
        to_csv(_columns()[:11])
    broken = _columns()
    broken[3] = broken[3][:-1]
    with pytest.raises(RecordingTableError, match="одно и то же число"):
        to_csv(broken)
    nan_columns = _columns()
    nan_columns[0][0] = math.nan
    with pytest.raises(RecordingTableError, match="NaN"):
        to_csv(nan_columns)


def test_assembled_table_uses_the_existing_csv_route(monkeypatch):
    monkeypatch.setattr(
        "backend.app.predict_signal",
        lambda signal, lead_names, sampling_rate: {
            "model": "ECGFounder",
            "scores": [{"label": "SINUS RHYTHM", "score": 0.2}],
            "note": "проверка кабельной таблицы",
        },
    )
    payload = to_csv([[0.0] * 5000 for _ in LEADS])
    with TestClient(app) as client:
        response = client.post(
            "/api/ecg/signal/csv",
            files={"file": ("cable.csv", payload, "text/csv")},
            data={"sampling_rate": "500"},
        )
    assert response.status_code == 200
    assert response.json()["model"] == "ECGFounder"
    assert RAW_TO_531_STATUS == "NOT_PROVEN"


def test_page_keeps_the_cable_closed_until_a_model_is_named():
    script = CABLE_JS.read_text(encoding="utf-8")
    assert "profiles.set(profile.model" not in script
    assert script.count("profiles.set(") == 1
    for literal in ("115200", "9600", "57600", "usbVendorId", "0x"):
        assert literal not in script
    with TestClient(app) as client:
        page = client.get("/")
        cable = client.get("/cable.js")
        guide = client.get("/ecg-connect.html")
    assert page.status_code == 200
    assert 'id="score-cable"' not in page.text
    assert 'id="panel-signal"' not in page.text
    assert 'src="/cable.js"' not in page.text
    assert "Разобрать" in page.text
    assert guide.status_code == 200
    assert "Аппарат по кабелю" in guide.text
    assert "cu.Bluetooth-Incoming-Port" in guide.text
    assert "не нажимайте" in guide.text
    assert "BTL-08" in guide.text
    assert "Поли-Спектр" in guide.text
    assert cable.status_code == 200
    assert cable.headers["cache-control"] == "no-store"
    assert "Модель аппарата не названа" in cable.text
    assert "BTL-08" in cable.text
    assert "Поли-Спектр" in cable.text


def test_browser_table_matches_python_and_registers_no_device():
    columns = _columns()
    expression = (
        "JSON.stringify({names: globalThis.cable.profileNames(),"
        f"csv: globalThis.cable.samplesToCsv({json.dumps(columns)})}})"
    )
    raw = _eval_cable(expression)
    body = json.loads(raw)
    assert body["names"] == []
    listed = _eval_cable("JSON.stringify(globalThis.cable.devices())")
    devices = json.loads(listed)
    assert devices
    assert {item["action"] for item in devices} == {"file"}
    assert any(item["title"] == "BTL-08" for item in devices)
    parsed = [[float(cell) for cell in row.split(",")] for row in body["csv"].splitlines()[1:]]
    np.testing.assert_allclose(np.array(parsed).T, columns)
    assert body["csv"].splitlines()[0].split(",") == list(LEADS)


def _eval_cable(expression: str) -> str:
    source = CABLE_JS.read_text(encoding="utf-8")
    node = shutil.which("node")
    if node:
        program = source + f"\nconsole.log({expression});\n"
        completed = subprocess.run([node, "-e", program], check=True, capture_output=True, text=True)
        return completed.stdout.strip()
    program = source + "\n" + expression + "\n"
    completed = subprocess.run(
        ["osascript", "-l", "JavaScript", "-e", program],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()
