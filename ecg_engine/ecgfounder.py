"""Инференс ECGFounder на цифровой 12-канальной записи.

Вход совпадает с ptbxl_eval.py авторов: около 10 секунд, затем 5000 отсчётов,
общий z-score, сигмоида головы на 150 классов. Median beat и снимок сюда не входят.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ecg_engine.ecgfounder_net import build_ecgfounder

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = ROOT / "models" / "ecgfounder" / "12_lead_ECGFounder.pth"
TASKS_PATH = Path(__file__).with_name("ecgfounder_tasks.txt")
LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
_LEAD_KEY = {name.upper(): name for name in LEADS}
TARGET_SAMPLES = 5000
MIN_SECONDS = 8.0
MAX_SECONDS = 12.0


class SignalError(ValueError):
    """Цифровая запись не подходит ECGFounder."""


class WeightsMissing(FileNotFoundError):
    """Файл весов ECGFounder не найден."""


def lead_name(raw: str) -> str:
    key = raw.strip().upper().replace(" ", "")
    if key not in _LEAD_KEY:
        raise SignalError(f"Неизвестное отведение {raw!r}. Нужны {', '.join(LEADS)}.")
    return _LEAD_KEY[key]


def load_labels() -> tuple[str, ...]:
    labels = tuple(line.strip() for line in TASKS_PATH.read_text(encoding="utf-8").splitlines() if line.strip())
    if len(labels) != 150:
        raise RuntimeError(f"Ожидалось 150 меток ECGFounder, получено {len(labels)}.")
    return labels


def arrange(signal: object, lead_names: list[str]) -> np.ndarray:
    names = [lead_name(name) for name in lead_names]
    if len(names) != len(set(names)):
        raise SignalError("Отведения повторяются.")
    missing = [name for name in LEADS if name not in names]
    if missing or len(names) != 12:
        raise SignalError(f"Нужны все 12 отведений. Нет: {', '.join(missing)}.")
    array = np.asarray(signal, dtype=np.float64)
    if array.ndim != 2:
        raise SignalError("Сигнал должен быть таблицей отведение × время.")
    if array.shape[0] == 12:
        by_lead = array
    elif array.shape[1] == 12:
        by_lead = array.T
    else:
        raise SignalError(f"Ожидается 12 отведений, получена форма {array.shape}.")
    if not np.isfinite(array).all():
        raise SignalError("В сигнале есть NaN или Inf.")
    order = [names.index(name) for name in LEADS]
    return by_lead[order]


def to_model_input(leads_by_time: np.ndarray, sampling_rate: float) -> np.ndarray:
    if not np.isfinite(sampling_rate) or sampling_rate <= 0:
        raise SignalError("Частота дискретизации должна быть положительным числом.")
    samples = leads_by_time.shape[1]
    duration = samples / float(sampling_rate)
    if not (MIN_SECONDS <= duration <= MAX_SECONDS):
        raise SignalError(
            f"ECGFounder принимает запись около 10 секунд. "
            f"Получено {duration:.2f} с ({samples} отсчётов при {sampling_rate:g} Гц). "
            "Median beat из 600 отсчётов сюда не входит."
        )
    old_x = np.linspace(0.0, duration, num=samples, endpoint=True)
    new_x = np.linspace(0.0, duration, num=TARGET_SAMPLES, endpoint=True)
    resized = np.vstack([np.interp(new_x, old_x, leads_by_time[index]) for index in range(12)])
    prepared = (resized - resized.mean()) / (resized.std() + 1e-8)
    return prepared.astype(np.float32)


def weights_path() -> Path:
    override = os.environ.get("ECGFOUNDER_WEIGHTS", "").strip()
    return Path(override) if override else DEFAULT_WEIGHTS


@lru_cache(maxsize=1)
def get_model() -> torch.nn.Module:
    path = weights_path()
    if not path.is_file():
        raise WeightsMissing(
            "Веса ECGFounder не найдены: "
            f"{path}. Ожидается файл 12_lead_ECGFounder.pth."
        )
    model = build_ecgfounder(150)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    missing, unexpected = model.load_state_dict(state, strict=False)
    dense_missing = [name for name in missing if name.startswith("dense.")]
    if dense_missing:
        raise RuntimeError(f"Голова ECGFounder не загрузилась: {dense_missing[:4]}")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def predict_signal(signal: object, lead_names: list[str], sampling_rate: float, top: int = 20) -> dict[str, object]:
    prepared = to_model_input(arrange(signal, lead_names), sampling_rate)
    model = get_model()
    with torch.no_grad():
        logits = model(torch.from_numpy(prepared).unsqueeze(0))
        scores = F.sigmoid(logits).cpu().numpy()[0]
    labels = load_labels()
    if scores.shape != (len(labels),):
        raise RuntimeError(f"ECGFounder вернула {scores.shape[0]} оценок, меток {len(labels)}.")
    order = np.argsort(-scores)
    limit = max(1, min(int(top), len(labels)))
    ranked = [
        {"label": labels[int(index)], "score": float(scores[int(index)])}
        for index in order[:limit]
    ]
    return {
        "model": "ECGFounder",
        "weights": "PKUDigitalHealth/ECGFounder 12_lead_ECGFounder.pth",
        "paper": "NEJM AI 2025, Li et al.",
        "input_samples": TARGET_SAMPLES,
        "input_sampling_rate_hz": 500,
        "lead_order": list(LEADS),
        "class_count": len(labels),
        "scores": ranked,
        "note": "Сигмоида исследовательской головы на 150 классов. Это не калиброванный диагноз.",
    }
