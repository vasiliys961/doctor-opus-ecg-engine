"""Проверки слоя совместимости. Ансамбль сюда не вызывается."""

from __future__ import annotations

import math

import numpy as np

from ecg_engine.canonical_schema import FEATURE_COLUMNS_531
from ecg_engine.raw_extractor import RAW_TO_531_STATUS


class CompatibilityError(ValueError):
    """Вектор нельзя отдать существующему ансамблю."""


def assert_bundle_shape(names: list[str], values: list[object], mask: list[int], statuses: list[str]) -> None:
    if names != list(FEATURE_COLUMNS_531):
        raise CompatibilityError("Имена признаков разошлись с canonical schema.")
    if not (len(values) == len(mask) == len(statuses) == 531):
        raise CompatibilityError("Длины вектора, маски и статусов должны быть 531.")
    if RAW_TO_531_STATUS != "NOT_PROVEN":
        raise CompatibilityError("RAW_TO_531_STATUS изменён.")


def ensemble_allowed(values: list[object], statuses: list[str]) -> bool:
    """Существующий ансамбль принимает только полный конечный reference-вектор."""
    if any(status != "EXACT_PROVEN" for status in statuses):
        return False
    for value in values:
        if value is None or (isinstance(value, float) and not math.isfinite(value)):
            return False
    return True


def reject_incomplete_for_ensemble(values: np.ndarray) -> None:
    if not np.isfinite(values).all():
        raise CompatibilityError(
            "COMPATIBILITY_ONLY. Существующий ансамбль не запускается на векторе с NaN."
        )
