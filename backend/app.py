"""HTTP-сервис ECG engine. Не импортирует production Doctor Opus."""

from __future__ import annotations

import csv
import io
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from backend.vision import (
    EmptyCase,
    UnsupportedImage,
    VisionUnavailable,
    analyze_case,
    analyze_ecg_image,
    form_protocol,
    form_signal_conclusion,
)
from ecg_engine.delineation import measure_tracing
from ecg_engine.ecgfounder import SignalError, WeightsMissing, arrange, predict_signal
from ecg_engine.feature_compatibility import FeatureCompatibilityEngine, json_ready
from ecg_engine.preprocessing import PreprocessingError
from ecg_engine.raw_extractor import RawECGExtractor

ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend" / "index.html"
CABLE = ROOT / "frontend" / "cable.js"
CONNECT_GUIDE = ROOT / "frontend" / "ecg-connect.html"
EXAMPLE_CSV = ROOT / "examples2" / "00001_норма.csv"

app = FastAPI(title="Doctor Opus ECG Engine", version="0.2.0")


class ProtocolBody(BaseModel):
    interpretation: str
    extraction: dict | None = None


class SignalConclusionBody(BaseModel):
    measurements: dict
    scores: list


class ExampleBody(BaseModel):
    sampling_rate: float = 500


class SignalBody(BaseModel):
    signal: list
    sampling_rate: float
    lead_names: list[str]


class RawFeaturesBody(BaseModel):
    signal: list
    sampling_rate: float
    lead_names: list[str]


def _signal_error(exc: Exception) -> HTTPException:
    if isinstance(exc, SignalError):
        return HTTPException(status_code=400, detail=str(exc))
    if isinstance(exc, WeightsMissing):
        return HTTPException(status_code=503, detail=str(exc))
    raise exc


def _trace_preview(arranged: object, buckets: int = 500) -> list[dict[str, object]]:
    import numpy as np

    from ecg_engine.ecgfounder import LEADS

    matrix = np.asarray(arranged, dtype=float)
    width = int(matrix.shape[1])
    edges = np.linspace(0, width, buckets + 1, dtype=int)
    rows: list[dict[str, object]] = []
    for index, name in enumerate(LEADS):
        lead = matrix[index]
        samples: list[float] = []
        for start, stop in zip(edges[:-1], edges[1:]):
            chunk = lead[int(start) : int(stop)]
            if chunk.size == 0:
                continue
            samples.append(round(float(chunk.min()), 4))
            samples.append(round(float(chunk.max()), 4))
        rows.append({"lead": name, "samples": samples})
    return rows


def _digital_report(signal: object, lead_names: list[str], sampling_rate: float) -> dict[str, object]:
    arranged = arrange(signal, lead_names)
    result = predict_signal(signal, lead_names, sampling_rate)
    result["measurements"] = measure_tracing(arranged, sampling_rate)
    result["trace"] = _trace_preview(arranged)
    return result


@app.post("/api/ecg/signal")
def score_signal(body: SignalBody):
    try:
        return _digital_report(body.signal, body.lead_names, body.sampling_rate)
    except (SignalError, WeightsMissing) as exc:
        raise _signal_error(exc) from exc


def _columns_from_csv(text: str) -> tuple[list[str], list[list[float]]]:
    rows = list(csv.reader(io.StringIO(text)))
    if len(rows) < 2:
        raise HTTPException(status_code=400, detail="В CSV нет строки отсчётов.")
    lead_names = [cell.strip() for cell in rows[0]]
    columns: list[list[float]] = [[] for _ in lead_names]
    try:
        for row in rows[1:]:
            if not any(cell.strip() for cell in row):
                continue
            if len(row) != len(lead_names):
                raise HTTPException(status_code=400, detail="В строке CSV другое число колонок, чем в заголовке.")
            for index, cell in enumerate(row):
                columns[index].append(float(cell))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="В CSV есть нечисловой отсчёт.") from exc
    return lead_names, columns


@app.post("/api/ecg/signal/csv")
async def score_signal_csv(
    file: UploadFile = File(...),
    sampling_rate: float = Form(...),
):
    text = (await file.read()).decode("utf-8-sig")
    lead_names, columns = _columns_from_csv(text)
    try:
        return _digital_report(columns, lead_names, sampling_rate)
    except (SignalError, WeightsMissing) as exc:
        raise _signal_error(exc) from exc


@app.post("/api/ecg/signal/example")
def score_example(body: ExampleBody):
    if not EXAMPLE_CSV.is_file():
        raise HTTPException(status_code=404, detail="Пример записи не найден.")
    lead_names, columns = _columns_from_csv(EXAMPLE_CSV.read_text(encoding="utf-8-sig"))
    try:
        report = _digital_report(columns, lead_names, body.sampling_rate)
    except (SignalError, WeightsMissing) as exc:
        raise _signal_error(exc) from exc
    report["example"] = "PTB-XL 00001, открытая 10-секундная запись. Это не подключённый аппарат."
    return report


@app.post("/api/ecg/signal/conclusion")
def signal_conclusion(body: SignalConclusionBody):
    try:
        return form_signal_conclusion(body.measurements, body.scores)
    except EmptyCase as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except VisionUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/")
def index():
    return FileResponse(FRONTEND, headers={"Cache-Control": "no-store"})


@app.get("/cable.js")
def cable_script():
    if not CABLE.is_file():
        raise HTTPException(status_code=404, detail="Скрипт кабеля не найден.")
    return FileResponse(CABLE, media_type="application/javascript", headers={"Cache-Control": "no-store"})


@app.get("/ecg-connect.html")
def connect_guide():
    if not CONNECT_GUIDE.is_file():
        raise HTTPException(status_code=404, detail="Инструкция по подключению не найдена.")
    return FileResponse(CONNECT_GUIDE, media_type="text/html", headers={"Cache-Control": "no-store"})


@app.post("/api/ecg/analyze")
async def analyze(
    file: UploadFile | None = File(None),
    notes: str = Form(""),
    clinical_context: str = Form(""),
):
    payload = await file.read() if file is not None else None
    if payload == b"":
        payload = None
    try:
        return analyze_case(
            image=payload,
            filename=file.filename if file is not None else "",
            mime_type=file.content_type if file is not None else "",
            notes=notes,
            clinical_context=clinical_context,
        )
    except EmptyCase as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except UnsupportedImage as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc
    except VisionUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/api/ecg/protocol")
def protocol(body: ProtocolBody):
    try:
        return form_protocol(body.interpretation, body.extraction)
    except EmptyCase as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except VisionUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/api/ecg/image")
async def analyze_image(
    file: UploadFile = File(...),
    clinical_context: str = Form(""),
):
    payload = await file.read()
    try:
        return analyze_ecg_image(payload, file.filename or "", file.content_type or "", clinical_context)
    except UnsupportedImage as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc
    except VisionUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/api/ecg/raw")
def raw_ecg():
    return JSONResponse(status_code=501, content=RawECGExtractor().status())


@app.post("/api/ecg/raw/features")
def raw_features(body: RawFeaturesBody):
    try:
        payload = FeatureCompatibilityEngine().analyze(body.signal, body.sampling_rate, body.lead_names)
    except PreprocessingError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    public = json_ready(payload)
    public.pop("analysis", None)
    public.pop("preprocessing", None)
    public["preprocessing_recorded"] = True
    if "prediction" in public:
        public["prediction"] = None
    return public
