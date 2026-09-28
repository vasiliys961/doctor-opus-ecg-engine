"""HTTP-сервис ECG engine. Не импортирует production Doctor Opus."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from backend.vision import UnsupportedImage, VisionUnavailable, analyze_ecg_image
from ecg_engine.ensemble import ModelAssetsError, predict_features, service_payload
from ecg_engine.feature_compatibility import FeatureCompatibilityEngine, json_ready
from ecg_engine.preprocessing import PreprocessingError
from ecg_engine.raw_extractor import RawECGExtractor
from ecg_engine.schema import ECGSchemaError

ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend" / "index.html"

app = FastAPI(title="Doctor Opus ECG Engine", version="0.2.0")


class PredictBody(BaseModel):
    features: dict


class RawFeaturesBody(BaseModel):
    signal: list
    sampling_rate: float
    lead_names: list[str]


@app.get("/")
def index():
    return FileResponse(FRONTEND)


@app.post("/api/ecg/predict")
def predict(body: PredictBody):
    try:
        return service_payload(predict_features(body.features))
    except ECGSchemaError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ModelAssetsError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


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
