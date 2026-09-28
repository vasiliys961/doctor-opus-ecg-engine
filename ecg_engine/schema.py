"""Контракт входа ECG531Input.

Порядок колонок — часть модели. Модуль не сортирует столбцы, не выбрасывает
неизвестные и не заполняет пропуски.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ecg_engine.feature_columns import FEATURE_COLUMNS
from ecg_engine.version import FEATURE_SCHEMA_VERSION, INPUT_SCHEMA, MODEL_VERSION

IDENTIFIER_COLUMN = "ecg_id"
FEATURE_COUNT = 531


class ECGSchemaError(ValueError):
    """CSV не соответствует зафиксированной схеме 531."""


@dataclass(frozen=True)
class ECG531Input:
    """Одна строка признаков в каноническом порядке, до нормализации."""

    values: np.ndarray
    row_index: int
    rows_in_file: int
    model_version: str = MODEL_VERSION
    feature_schema_version: str = FEATURE_SCHEMA_VERSION
    input_schema: str = INPUT_SCHEMA

    def __post_init__(self) -> None:
        if self.values.shape != (FEATURE_COUNT,):
            raise ECGSchemaError(
                f"Внутренний вектор должен иметь форму ({FEATURE_COUNT},), получено {self.values.shape}."
            )
        if not np.isfinite(self.values).all():
            raise ECGSchemaError("Вектор признаков содержит NaN или Inf.")


def _feature_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column != IDENTIFIER_COLUMN]


def _explain_column_mismatch(actual: list[str]) -> str:
    if len(actual) != FEATURE_COUNT:
        return (
            f"Ожидается ровно {FEATURE_COUNT} колонок признаков "
            f"(колонка {IDENTIFIER_COLUMN!r}, если она есть, не считается). "
            f"Получено {len(actual)}."
        )
    for index, (found, expected) in enumerate(zip(actual, FEATURE_COLUMNS)):
        if found != expected:
            return (
                f"Порядок колонок расходится со схемой на позиции {index}: "
                f"в файле {found!r}, в схеме {expected!r}. "
                "Колонки не сортируются и не переставляются."
            )
    return "Набор колонок не совпадает со схемой."


def _finite_value(column: str, raw: object) -> float:
    if raw is None or (isinstance(raw, float) and math.isnan(raw)) or pd.isna(raw):
        raise ECGSchemaError(
            f"В колонке {column!r} пропуск. NaN не заполняется средним обучающей выборки."
        )
    try:
        number = float(raw)
    except (TypeError, ValueError) as exc:
        raise ECGSchemaError(f"В колонке {column!r} нечисловое значение {raw!r}.") from exc
    if not math.isfinite(number):
        raise ECGSchemaError(f"В колонке {column!r} значение Inf. Оно не заменяется конечным числом.")
    return number


def read_ecg531_csv(csv_path: str) -> ECG531Input:
    """Читает первую строку CSV и проверяет её по каноническому порядку колонок."""
    try:
        frame = pd.read_csv(csv_path)
    except Exception as exc:
        raise ECGSchemaError(f"Не удалось прочитать CSV {csv_path}: {exc}") from exc

    if frame.empty:
        raise ECGSchemaError("CSV не содержит строк данных.")

    actual = _feature_columns(frame)
    if actual != list(FEATURE_COLUMNS):
        raise ECGSchemaError(_explain_column_mismatch(actual))

    row = frame.iloc[0]
    values = np.asarray(
        [_finite_value(column, row[column]) for column in FEATURE_COLUMNS],
        dtype=np.float64,
    )
    return ECG531Input(values=values, row_index=0, rows_in_file=int(frame.shape[0]))
