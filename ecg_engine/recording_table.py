"""Таблица 12 отведений для уже существующего POST /api/ecg/signal/csv.

Кабель и файл сходятся в одном CSV: первая строка — I, II, III, aVR, aVL, aVF, V1–V6,
дальше по отсчёту в строке. Байты аппарата сюда не входят. Модель не названа,
поэтому протокол потока не задан.
"""

from __future__ import annotations

import csv
import io
import math
from collections.abc import Sequence

from ecg_engine.ecgfounder import LEADS


class RecordingTableError(ValueError):
    """Столбцы отсчётов нельзя записать таблицей цифровой записи."""


def to_csv(columns: object) -> str:
    rows = _rows(columns)
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(LEADS)
    writer.writerows(rows)
    return buffer.getvalue()


def _rows(columns: object) -> list[list[str]]:
    if isinstance(columns, (str, bytes, bytearray)) or not isinstance(columns, Sequence):
        raise RecordingTableError("Нужны 12 столбцов отсчётов, не байты аппарата.")
    if len(columns) != len(LEADS):
        raise RecordingTableError("Нужны 12 отведений в порядке I, II, III, aVR, aVL, aVF, V1–V6.")
    parsed: list[list[float]] = []
    width: int | None = None
    for column in columns:
        if isinstance(column, (str, bytes, bytearray)) or not isinstance(column, Sequence):
            raise RecordingTableError("Каждое отведение — столбец чисел.")
        if width is None:
            width = len(column)
            if width == 0:
                raise RecordingTableError("В записи нет отсчётов.")
        elif len(column) != width:
            raise RecordingTableError("В каждом отведении должно быть одно и то же число отсчётов.")
        parsed.append([_sample(value) for value in column])
    assert width is not None
    return [[_cell(parsed[lead][time]) for lead in range(len(LEADS))] for time in range(width)]


def _sample(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RecordingTableError("Отсчёт должен быть числом.")
    number = float(value)
    if not math.isfinite(number):
        raise RecordingTableError("В отсчёте есть NaN или Inf.")
    return number


def _cell(number: float) -> str:
    return format(number, ".8g")
