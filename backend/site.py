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


async def _strip_frames(frames: list[UploadFile] | None) -> list[tuple[bytes, str, str]]:
    images: list[tuple[bytes, str, str]] = []
    for item in frames or []:
        payload = await item.read()
        if payload:
            images.append((payload, item.filename or "", item.content_type or ""))
    return images


@app.post("/api/ecg/analyze")
async def analyze(
    file: UploadFile | None = File(None),
    frames: list[UploadFile] | None = File(None),
    notes: str = Form(""),
    clinical_context: str = Form(""),
):
    images = await _strip_frames(frames)
    payload = None
    if not images and file is not None:
        payload = await file.read() or None
    try:
        return analyze_case(
            image=payload,
            filename=file.filename if file is not None else "",
            mime_type=file.content_type if file is not None else "",
            images=images or None,
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
