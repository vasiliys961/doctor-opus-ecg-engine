"""Проверки входа benchmark. Не подменяет битые поля."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from benchmark.data import CANONICAL_LEADS, WaveformError, read_header, validate_model_array
from benchmark.label_mapping import parse_scp_codes


class DatasetValidationError(ValueError):
    """Запись не проходит контракт benchmark."""


def validate_record(
    signal: np.ndarray,
    sampling_rate: int,
    lead_names: list[str],
    ecg_id: str,
    patient_id: str,
    scp_codes: object,
) -> None:
    if not str(ecg_id).strip() or not str(patient_id).strip():
        raise DatasetValidationError("ecg_id и patient_id обязательны.")
    if sampling_rate != 500:
        raise DatasetValidationError(f"Частота {sampling_rate} не равна 500. Ресэмплинг в baseline выключен.")
    if tuple(lead_names) != CANONICAL_LEADS:
        raise DatasetValidationError("Порядок отведений не канонический. Перестановка не выполняется.")
    try:
        validate_model_array(signal)
        parse_scp_codes(scp_codes)
    except (WaveformError, ValueError) as exc:
        raise DatasetValidationError(str(exc)) from exc
    if signal.shape[-1] != 5000:
        raise DatasetValidationError("Длительность не равна 10 с при 500 Гц.")


def validate_header_file(path: Path) -> None:
    n_leads, sampling, samples, names = read_header(path)
    if n_leads != 12 or sampling != 500 or samples != 5000:
        raise DatasetValidationError(f"Заголовок {path.name} не соответствует 12 x 500 Гц x 10 с.")
    if tuple(names) != CANONICAL_LEADS:
        raise DatasetValidationError(f"Отведения {names} не канонические.")
