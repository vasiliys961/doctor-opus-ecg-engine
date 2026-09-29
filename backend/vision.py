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
import ssl
import urllib.error
import urllib.request
from pathlib import Path

import certifi


def _load_local_env() -> None:
    path = Path(__file__).resolve().parents[1] / ".env"
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip()
        value = value.strip().strip('"').strip("'")
        if name and name not in os.environ:
            os.environ[name] = value


_load_local_env()

EYES_MODEL = os.environ.get("ECG_OBSERVER_MODEL", "google/gemini-3.8-flash")
ANALYZER_MODEL = os.environ.get("ECG_INTERPRETER_MODEL", "anthropic/claude-opus-5.5")
OBSERVER_MODEL = EYES_MODEL
INTERPRETER_MODEL = ANALYZER_MODEL

ECG_DOCTOR = (
    "You have the competence of a physician who reports routine ECGs. "
    "You use the ordinary protocol wording and you keep standard abbreviations: "
    "ЧСС, уд/мин, ЭОС, PQ, QRS, ST, QT, QTc, БПНПГ, БЛНПГ, ГЛЖ, ФП. "
    "Do not expand an abbreviation into a textbook phrase such as «блокада ножки пучка Гиса» "
    "or «частота сердечных сокращений составляет … ударов в минуту». "
    "Do not list mutually exclusive statements as if all of them were present. "
    "Weigh each classifier score against the measured intervals and keep the statement that fits. "
    "Do not write treatment, resuscitation, drugs, or a differential essay."
)

ECG_REQUIREMENTS = (
    "EXTRACT ALL METRICS WITH MAXIMUM PRECISION: 1. Technical parameters (Voltage, Speed mm/s). "
    "2. Rhythm (regularity, source), HR. 3. Electrical axis (alpha angle in degrees). "
    "4. Intervals in ms: P-wave (amplitude, duration), PR (interval), QRS (complex), QT/QTc (corrected). "
    "5. Segments: ST (elevation/depression in mm relative to TP baseline, morphology). "
    "6. Waves: Q, R progression V1–V6, T. 7. Hypertrophy voltage criteria. "
    "8. Conduction: bundle branch blocks, AV blocks."
)

OBSERVER_PROMPT = f"""You read this ECG image with the competence of an ECG diagnostician.
Describe only what is visible. Do not invent a measurement that is not readable.
If a value is not visible, use null.
Name findings with the usual abbreviations when the picture shows them: БПНПГ, БЛНПГ, ГЛЖ, ФП.
{ECG_DOCTOR}

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

MINNESOTA_CODES = (
    "The description standard is the Minnesota Code: classify the tracing, do not narrate it. "
    "Lead groups, when a site is needed: переднебоковые I, aVL, V6; нижние II, III, aVF; передние V1–V5. "
    "After Заключение print the title Миннесота and one short line. "
    "You may emit only these codes, and only when the stated measurement matches: "
    "2-1 if the QRS axis is from −30° through −90°; "
    "2-2 if the QRS axis is from +120° through +180°; "
    "an axis from −29° through +119° has no axis code; "
    "7-1-1 complete LBBB with QRS at least 120 ms; "
    "7-2-1 complete RBBB with QRS at least 120 ms; "
    "7-3 incomplete RBBB with QRS under 120 ms; "
    "7-6 incomplete LBBB with QRS under 120 ms; "
    "7-4 if QRS is at least 120 ms and there is no bundle-branch label; "
    "8-3-1 atrial fibrillation when P is unstable and RR is uneven; "
    "8-1-2 only when a wide premature beat is counted; "
    "8-7 sinus rhythm under 50 per minute. "
    "Separate codes with a comma. If none apply, write «кодов нет». Do not invent any other number. "
    "Do not emit a code that depends on wave height."
)

MINNESOTA_READING = (
    MINNESOTA_CODES
    + " Use only rhythm, axis, and durations written in the source. Do not invent a millimetre amplitude."
)

MINNESOTA = (
    MINNESOTA_CODES
    + " A classifier score is not a code. CSV amplitudes are not millimetres."
)

INTERPRETER_PROMPT = """You convert an existing ECG extraction into a short report for a clinician.
Use only facts present in the extraction. Do not invent grid values, calibration, or diagnoses.
Write in Russian. No JSON. No Markdown. No asterisks. No English headings.
Omit every measurement that is missing. Do not write "не указано" or "not specified".

Put each of these titles on its own line, and only if that section has something to say:
Запись
Что видно
С чем сравнивать
Что проверить у пациента
Заключение
Что делать

Under a title, one fact per line, each line starting with "- ".
In Заключение, say once that this is decision support and not an autonomous diagnosis.
"""

PROTOCOL_FIELDS = (
    "Ритм",
    "ЧСС",
    "Интервалы",
    "Зубцы",
    "Важные соотношения",
    "Депрессии и элевации",
    "Патологические зубцы",
    "Заключение",
    "Миннесота",
)

BLANK_GUIDE = """Print these titles, each on its own line, in this order:
Ритм
ЧСС
Интервалы
Зубцы
Важные соотношения
Депрессии и элевации
Патологические зубцы
Заключение
Миннесота

Everything before Заключение is a short preamble. One line under each title. No leading dash.
Do not write the patient's name, age, or sex.
Ритм is like "синусовый". ЧСС is like "64 уд/мин".
Интервалы is one line, like "P 126 мс, PQ 132 мс, QRS 118 мс, QT 396 мс, QTc 463 мс".
Зубцы is one line on shape, like "P есть, QRS узкий, T положительный".
Важные соотношения is one line of the usual indices the source already has: ЭОС, Соколов–Лайон, Корнелл, Льюис, Губнер, R/S V1, R/S V5, угол QRS–T. Example: "ЭОС 3°, Соколов–Лайон +1.209, Корнелл +0.840, Льюис +0.350, Губнер +0.900, R/S V1 0.13, R/S V5 2.40, QRS–T 21°". Skip an index that was not calculated. These values are not millimetres, so do not compare them with a textbook cutoff and do not add hypertrophy from the index alone.
Депрессии и элевации is one line of ST, like "II +0.020, J+80 мс". If the source shows none, write "нет".
Патологические зубцы is one line for a pathological Q, QS, or another abnormal wave the source describes. If none, write "нет".
Do not invent a number to fill a line. Omit a missing number, and write "нет" when the whole line has no finding.
Заключение comes after the preamble. It is the line a doctor would sign, not a repeat of the preamble.
"""

PROTOCOL_PROMPT = f"""Rewrite the shown ECG reading as a standard ECG description form.
{ECG_DOCTOR}
The source is already written. Do not look at an image. Do not invent numbers or findings.
Write in Russian. No JSON. No Markdown. No asterisks. No English headings.
This is a description blank, not a consultation and not a treatment plan.

{BLANK_GUIDE}
{MINNESOTA_READING}
Do not write differential diagnosis, artifact discussion, comparison with other recordings, clinical correlation, recommendations, resuscitation, defibrillation, drugs, or calls for a team.
"""


class VisionUnavailable(RuntimeError):
    """Канал изображения не вызвал модель."""


class UnsupportedImage(ValueError):
    """Формат изображения для этого этапа не принимается."""


class EmptyCase(ValueError):
    """Нет ни снимка, ни текста."""


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
        context = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(request, timeout=120, context=context) as response:
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


TEXT_EYES_PROMPT = f"""You read an ECG note with the competence of an ECG diagnostician. You have no image.
Copy only measurements and findings that are explicitly written.
Do not invent numbers. If a value is absent, use null.
Keep the note's abbreviations; do not expand БПНПГ, БЛНПГ, ГЛЖ, ФП, ЧСС, QRS, ST, QT.
{ECG_DOCTOR}

Technical requirements, only when the note states them: {ECG_REQUIREMENTS}

Return JSON only:
{{
  "schema_version": "ecg.vision.v1",
  "image_quality": null,
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


STRIP_EYES_NOTE = (
    "These pictures are ordered frames of one paper ECG strip. "
    "The camera moved along the paper. Neighbouring frames overlap. "
    "Read them as one recording. "
    "Do not treat the frames as separate patients or separate ECGs. "
    "The time between frames is camera time, not ECG time. "
    "Do not invent a measurement that is readable on none of the frames."
)

MAX_STRIP_FRAMES = 6
MAX_FRAME_BYTES = 1_500_000


def _eyes_content(image_urls: list[str], notes: str) -> list | str:
    if image_urls:
        text = OBSERVER_PROMPT
        if len(image_urls) > 1:
            text = f"{STRIP_EYES_NOTE}\n\n{text}"
        if notes:
            text += (
                "\n\nSUPPLIED TEXT is not the picture. "
                "Do not copy it into measurements unless the same value is visible on the image.\n"
                f"SUPPLIED TEXT:\n{notes}"
            )
        content: list = [{"type": "text", "text": text}]
        content.extend({"type": "image_url", "image_url": {"url": url}} for url in image_urls)
        return content
    return f"{TEXT_EYES_PROMPT}\n\nNOTE:\n{notes}"


def _analyzer_input(observer_text: str, notes: str, clinical_context: str) -> str:
    parts = [
        INTERPRETER_PROMPT,
        "The extraction comes from the eyes model. Use it. Do not repeat the JSON.",
        "SUPPLIED TEXT, when present, is what the user typed. It is not a measurement you saw.",
        f"\nEXTRACTION:\n{observer_text}",
    ]
    if notes:
        parts.append(f"\nSUPPLIED TEXT:\n{notes}")
    if clinical_context:
        parts.append(f"\nCLINICAL CONTEXT:\n{clinical_context}")
    return "\n".join(parts)


def analyze_case(
    *,
    image: bytes | None = None,
    filename: str = "",
    mime_type: str = "",
    images: list[tuple[bytes, str, str]] | None = None,
    notes: str = "",
    clinical_context: str = "",
) -> dict:
    """Глаза — Gemini, анализатор — Opus. Ансамбль 531 не вызывается."""
    supplied = notes.strip()
    context = clinical_context.strip()
    payloads = [(payload, name, mime) for payload, name, mime in (images or []) if payload]
    if not payloads and image:
        payloads = [(image, filename, mime_type)]
    if len(payloads) > MAX_STRIP_FRAMES:
        raise UnsupportedImage("Для ленты нужно не больше шести кадров.")
    image_urls: list[str] = []
    for payload, name, mime in payloads:
        raw, ready = prepare_image(payload, name, mime)
        if len(raw) > MAX_FRAME_BYTES:
            raise UnsupportedImage("Кадр ленты слишком большой.")
        image_urls.append(f"data:{ready};base64,{base64.b64encode(raw).decode('ascii')}")
    image_preserved = bool(image_urls)
    if not image_urls and not supplied:
        raise EmptyCase("Нужно изображение ЭКГ или текст: измерения, описание, заключение аппарата.")
    if len(image_urls) > 1 and supplied:
        kind = "strip+text"
    elif len(image_urls) > 1:
        kind = "strip"
    elif image_urls and supplied:
        kind = "image+text"
    elif image_urls:
        kind = "image"
    else:
        kind = "text"
    observer_text = _completion(EYES_MODEL, _eyes_content(image_urls, supplied))
    extraction, parse_warning = _parse_json(observer_text)
    interpretation = _completion(ANALYZER_MODEL, _analyzer_input(observer_text, supplied, context))
    return {
        "input_kind": kind,
        "ensemble_used": False,
        "image_preserved": image_preserved,
        "edge_mask_applied": False,
        "reencoded": False,
        "eyes_model": EYES_MODEL,
        "analyzer_model": ANALYZER_MODEL,
        "observer_model": EYES_MODEL,
        "interpreter_model": ANALYZER_MODEL,
        "extraction": extraction,
        "extraction_warning": parse_warning,
        "observer_text": observer_text,
        "interpretation": interpretation,
    }


def _protocol_form(text: str) -> str:
    """Оставляет только поля бланка. Лишние разделы модели в протокол не попадают."""
    allowed = {title.casefold(): title for title in PROTOCOL_FIELDS}
    kept: list[str] = []
    accept = False
    for raw in text.replace("**", "").replace("*", "").splitlines():
        line = raw.strip()
        bare = line[1:].strip() if line.startswith("-") else line
        key = bare.casefold()
        matched = next((title for name, title in allowed.items() if key == name or key.startswith(name + ":")), None)
        if matched:
            accept = True
            kept.append(matched)
            same_line = bare.split(":", 1)[1].strip() if ":" in bare else ""
            if same_line:
                kept.append(same_line)
                accept = False
            continue
        if accept and bare:
            kept.append(bare)
            accept = False
    return "\n".join(kept)


SIGNAL_CONCLUSION_PROMPT = f"""Fill a standard ECG description form from two prepared blocks about one digital recording.
{ECG_DOCTOR}
MEASUREMENTS are intervals and amplitudes already calculated. Put those numbers in the matching fields. Do not replace them.
ECGFOUNDER lines are sigmoid scores of a research classifier. They are not a calibrated diagnosis and not a percent of accuracy.
You decide which scores belong in the conclusion the way an ECG diagnostician would.
Write in Russian. No JSON. No Markdown. No asterisks. No English headings.
This is a formal ECG protocol, not a consultation and not a treatment plan.

{BLANK_GUIDE}
Do not write "составляет", "равен", "регистрируется", "продолжительность", "частота сердечных сокращений" or "ударов в минуту".
Keep abbreviations as written: ЧСС, уд/мин, PQ, QRS, ST, QT, QTc, БПНПГ, БЛНПГ, ГЛЖ, ФП. Do not expand them into "пучок Гиса" or a full phrase.
Заключение is one short line, not a catalogue of every score.
Ignore scores below 0.5.
If two scores exclude each other, keep the one that matches the measurements.
Sinus rhythm and atrial fibrillation: unstable P and uneven RR means ФП.
Incomplete and complete bundle branch block: keep the one that matches the QRS duration.
Normal ECG and abnormal ECG: do not write both.
Use Russian abbreviations: неполная БПНПГ, БПНПГ, БЛНПГ, ГЛЖ, ФП, перегородочный инфаркт.
{MINNESOTA}
Do not write differential diagnosis, artifact discussion, comparison with other recordings, clinical correlation, recommendations, resuscitation, defibrillation, drugs, or calls for a team.
"""


def _signal_brief(measurements: dict, scores: list) -> str:
    text_lines = [
        f"{item.get('name')}: {item.get('value')}"
        for item in measurements.get("lines") or []
        if item.get("name") and item.get("value")
    ]
    lead_lines = []
    for row in measurements.get("leads") or []:
        lead_lines.append(
            f"{row.get('lead')}: ST {row.get('st')}, Q {row.get('q_ms')} мс, QT {row.get('qt_ms')} мс, качество {row.get('quality')}"
        )
    height_lines = []
    for row in measurements.get("amplitudes") or []:
        height_lines.append(
            f"{row.get('lead')}: P {row.get('p')}, Q {row.get('q')}, R {row.get('r')}, S {row.get('s')}, T {row.get('t')}"
        )
    ranked = []
    for item in scores:
        label = item.get("label")
        score = item.get("score")
        if not label or score is None:
            continue
        ranked.append(f"{label}: {float(score):.3f}")
    parts = [
        "MEASUREMENTS",
        str(measurements.get("description") or ""),
        "\n".join(text_lines),
        "LEADS",
        "\n".join(lead_lines),
        "AMPLITUDES",
        "\n".join(height_lines),
        "ECGFOUNDER",
        "\n".join(ranked[:20]),
    ]
    return "\n".join(part for part in parts if part.strip())


def form_signal_conclusion(measurements: dict | None, scores: list | None) -> dict:
    """Формальный бланк по цифровой кривой пишет Gemini, как протокол первого модуля."""
    if not measurements or not measurements.get("available"):
        raise EmptyCase("Сначала нужна разметка кривой.")
    if not scores:
        raise EmptyCase("Сначала нужны оценки ECGFounder.")
    brief = _signal_brief(measurements, scores)
    drafted = _completion(EYES_MODEL, f"{SIGNAL_CONCLUSION_PROMPT}\n\n{brief}")
    return {"protocol": _protocol_form(drafted), "protocol_model": EYES_MODEL}


def form_protocol(interpretation: str, extraction: dict | None = None) -> dict:
    """Второй шаг: бланк протокола пишет дешёвая модель. Снимок повторно не отправляется."""
    shown = interpretation.strip()
    if not shown:
        raise EmptyCase("Сначала нужно заключение на экране.")
    source = shown
    if extraction:
        source += "\n\nEXTRACTION:\n" + json.dumps(extraction, ensure_ascii=False)
    drafted = _completion(EYES_MODEL, f"{PROTOCOL_PROMPT}\n\nSHOWN CONCLUSION:\n{source}")
    return {"protocol": _protocol_form(drafted), "protocol_model": EYES_MODEL}


def analyze_ecg_image(payload: bytes, filename: str, mime_type: str, clinical_context: str = "") -> dict:
    return analyze_case(
        image=payload,
        filename=filename,
        mime_type=mime_type,
        clinical_context=clinical_context,
    )
