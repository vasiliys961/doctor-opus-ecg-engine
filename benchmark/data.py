"""Чтение records500. Порядок отведений не переставляется молча."""

from __future__ import annotations

from pathlib import Path

import numpy as np

CANONICAL_LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
_NAME_ALIASES = {"AVR": "aVR", "AVL": "aVL", "AVF": "aVF"}

OFFICIAL_ECGDELI_SHA256 = "84143735682f727201cc7f2825c047f98898f9756c6a21729af993b516221556"


class WaveformError(ValueError):
    """Сигнал нельзя подать в raw-модель."""


def canonical_lead(name: str) -> str:
    token = name.strip()
    return _NAME_ALIASES.get(token.upper(), token)


def read_header(path: Path) -> tuple[int, int, int, list[str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    record, leads, sampling, samples = lines[0].split()[:4]
    names = [canonical_lead(line.split()[-1]) for line in lines[1 : 1 + int(leads)]]
    return int(leads), int(sampling), int(samples), names


def read_records500(header_path: Path) -> np.ndarray:
    """Возвращает милливольты формы [12, N]. Чужой порядок отведений — ошибка."""
    n_leads, _sampling, n_samples, names = read_header(header_path)
    if tuple(names) != CANONICAL_LEADS:
        raise WaveformError(
            f"Порядок отведений {names} не равен каноническому {list(CANONICAL_LEADS)}. "
            "Сигнал не переставляется."
        )
    dat_path = header_path.with_suffix(".dat")
    raw = np.fromfile(dat_path, dtype="<i2")
    if raw.size != n_samples * n_leads:
        raise WaveformError(f"Размер {dat_path.name} не совпадает с заголовком.")
    # WFDB format 16: отведения чередуются внутри каждого отсчёта.
    adc = raw.reshape(n_samples, n_leads).T
    # PTB-XL records500: gain 1000 ADC/mV, baseline 0.
    return adc.astype(np.float64) / 1000.0


def validate_model_array(signal: np.ndarray) -> None:
    if signal.ndim == 2:
        expected = (12, 5000)
        if signal.shape != expected:
            raise WaveformError(f"Ожидалась форма {expected}, получено {signal.shape}.")
    elif signal.ndim == 3:
        if signal.shape[1:] != (12, 5000):
            raise WaveformError(f"Ожидалась форма [B, 12, 5000], получено {signal.shape}.")
    else:
        raise WaveformError(f"Сигнал должен быть 2D или 3D, получено ndim={signal.ndim}.")
    if not np.isfinite(signal).all():
        raise WaveformError("В сигнале есть NaN или Inf.")


def train_lead_statistics(signals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """signals: [N, 12, T]. Среднее и std только по переданной выборке."""
    mean = signals.mean(axis=(0, 2))
    std = signals.std(axis=(0, 2))
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float64), std.astype(np.float64)


def apply_lead_normalization(signal: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (signal - mean[:, None]) / std[:, None]
