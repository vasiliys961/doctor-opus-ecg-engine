"""ECG image channel, адаптированный из doctor-opus-global.

Источник промптов: lib/prompts.ts (SPECIALIST_CRITERIA.ecg, getDescriptionPrompt)
и lib/diagnostic-report.ts (DIAGNOSTIC_ECG_SYSTEM_PROMPT), коммит 9221b37a.
Радиологический system prompt production-пути сюда не входит.
Изображение не маскируется и не перекодируется.
"""

from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request

OBSERVER_MODEL = os.environ.get("ECG_OBSERVER_MODEL", "google/gemini-3.8-flash")
INTERPRETER_MODEL = os.environ.get("ECG_INTERPRETER_MODEL", "anthropic/claude-sonnet-5")

ECG_REQUIREMENTS = (
    "EXTRACT ALL METRICS WITH MAXIMUM PRECISION: 1. Technical parameters (Voltage, Speed mm/s). "
    "2. Rhythm (regularity, source), HR. 3. Electrical axis (alpha angle in degrees). "
    "4. Intervals in ms: P-wave (amplitude, duration), PR (interval), QRS (complex), QT/QTc (corrected). "
    "5. Segments: ST (elevation/depression in mm relative to TP baseline, morphology). "
    "6. Waves: Q, R progression V1–V6, T. 7. Hypertrophy voltage criteria. "
    "8. Conduction: bundle branch blocks, AV blocks."
)

OBSERVER_PROMPT = f"""You are an ECG vision observer. Describe only what is visible on this ECG image.
Do not diagnose, do not recommend treatment, and do not invent measurements that are not readable.
If a value is not visible, use null.

Technical requirements: {ECG_REQUIREMENTS}

Return JSON only:
{{
  "schema_version": "ecg.vision.v1",
  "image_quality": "good",
  "quality_issues": [],
  "measurements": {{
    "heart_rate": null,
    "rhythm": null,
    "pr_ms": null,
    "qrs_ms": null,
    "qt_ms": null,
    "qtc_ms": null,
    "axis": null,
    "paper_speed_mm_s": null,
    "gain_mm_per_mv": null
  }},
  "findings": []
}}
"""

INTERPRETER_PROMPT = """You convert an existing ECG image extraction into a short diagnostic test report.
Use only facts present in the extraction. If a measurement is null, write "not specified".
Do not invent grid values, calibration, or diagnoses unsupported by the extraction.

Print these English headings exactly:
**Technical Parameters:**
**Findings:**
**Differential Diagnosis:**
**Clinical Correlation Needed:**
**Impression:**
**Recommendations:**

This is decision support, not an autonomous diagnosis.
"""


class VisionUnavailable(RuntimeError):
    """Канал изображения не вызвал модель."""


class UnsupportedImage(ValueError):
    """Формат изображения для этого этапа не принимается."""


def _api_key() -> str:
    return os.environ.get("LLM_API_KEY") or os.environ.get("OPENROUTER_API_KEY") or ""


def _endpoint() -> str:
    return os.environ.get("LLM_BASE_URL", "https://openrouter.ai/api/v1/chat/completions")


def prepare_image(payload: bytes, filename: str, mime_type: str) -> tuple[bytes, str]:
    """Проверяет формат и возвращает исходные байты без перекодирования и без маски краёв."""
    name = (filename or "").lower()
    mime = (mime_type or "").split(";")[0].strip().lower()
    if name.endswith(".pdf") or mime == "application/pdf" or payload.startswith(b"%PDF"):
        raise UnsupportedImage("PDF пока не поддерживается в ECG image channel.")
    if mime not in {"image/jpeg", "image/png", "image/gif", "image/webp"}:
        raise UnsupportedImage("Нужен JPG, PNG, GIF или WEBP. Файл не изменялся.")
    if not payload:
        raise UnsupportedImage("Пустой файл изображения.")
    return payload, mime


def _completion(model: str, content: list | str) -> str:
    key = _api_key()
    if not key:
        raise VisionUnavailable(
            "LLM_API_KEY не задан. Изображение не отправлялось, заключение не создано."
        )
    message_content = content if isinstance(content, list) else content
    body = json.dumps(
        {
            "model": model,
            "temperature": 0.1,
            "messages": [{"role": "user", "content": message_content}],
        }
    ).encode()
    request = urllib.request.Request(
        _endpoint(),
        data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = json.loads(response.read().decode())
    except urllib.error.URLError as exc:
        raise VisionUnavailable(f"Модель изображения недоступна: {exc}") from exc
    try:
        return payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise VisionUnavailable("Ответ модели не содержит текста.") from exc


def _parse_json(text: str) -> tuple[dict | None, str | None]:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None, "Наблюдатель не вернул JSON."
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None, "JSON наблюдателя не разобран."
    if not isinstance(parsed, dict):
        return None, "JSON наблюдателя не является объектом."
    return parsed, None


def analyze_ecg_image(payload: bytes, filename: str, mime_type: str, clinical_context: str = "") -> dict:
    image, mime = prepare_image(payload, filename, mime_type)
    encoded = base64.b64encode(image).decode("ascii")
    observer_text = _completion(
        OBSERVER_MODEL,
        [
            {"type": "text", "text": OBSERVER_PROMPT},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}},
        ],
    )
    extraction, parse_warning = _parse_json(observer_text)
    context = clinical_context.strip()
    interpretation_input = (
        f"{INTERPRETER_PROMPT}\n\nEXTRACTION:\n{observer_text}"
        + (f"\n\nCLINICAL CONTEXT:\n{context}" if context else "")
    )
    interpretation = _completion(INTERPRETER_MODEL, interpretation_input)
    return {
        "image_preserved": True,
        "edge_mask_applied": False,
        "reencoded": False,
        "observer_model": OBSERVER_MODEL,
        "interpreter_model": INTERPRETER_MODEL,
        "extraction": extraction,
        "extraction_warning": parse_warning,
        "observer_text": observer_text,
        "interpretation": interpretation,
    }
