"""Публичная страница: снимок и текст. Цифровой разбор и веса модели сюда не входят."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from backend.vision import (
    EmptyCase,
    UnsupportedImage,
    VisionUnavailable,
    analyze_case,
    analyze_ecg_image,
    form_protocol,
)

ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend" / "index.html"

app = FastAPI(title="Doctor Opus ECG", version="0.2.0")


class ProtocolBody(BaseModel):
    interpretation: str
    extraction: dict | None = None


@app.get("/")
def index():
    return FileResponse(FRONTEND, headers={"Cache-Control": "no-store"})


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
