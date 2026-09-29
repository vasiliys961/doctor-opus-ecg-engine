"""Проверка опубликованной таблицы 531. Ансамбль, который её считал, удалён."""

from __future__ import annotations

import hashlib
from pathlib import Path

from benchmark.data import OFFICIAL_ECGDELI_SHA256
from ecg_engine.feature_columns import FEATURE_COLUMNS


class FeatureTableError(RuntimeError):
    """Опубликованная таблица 531 непригодна."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ensemble_removed() -> None:
    raise RuntimeError("Ансамбль 531 удалён вместе с весами. Цифровую ЭКГ считает ECGFounder.")


def regression_max_abs(fixture: Path, model_dir: Path) -> float:
    del fixture, model_dir
    _ensemble_removed()
    return 0.0


def column_order_status(columns: list[str]) -> str:
    """Порядок берётся из canonical schema. Колонки не сортируются."""
    if "ecg_id" not in columns:
        return "MISSING_ECG_ID"
    features = [name for name in columns if name != "ecg_id"]
    known = set(FEATURE_COLUMNS)
    if any(name not in known for name in features) or len(features) != len(FEATURE_COLUMNS):
        missing = [name for name in FEATURE_COLUMNS if name not in columns]
        if missing:
            return "MISSING_531_COLUMN"
        return "OTHER"
    if features != list(FEATURE_COLUMNS):
        return "COLUMN_ORDER_MISMATCH"
    return "OK"


def _header_columns(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as handle:
        return handle.readline().rstrip("\n").split(",")


def _row_count(path: Path) -> int:
    with path.open("rb") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def classify_feature_table(path: Path) -> dict[str, object]:
    """Причина, по которой published 531 нельзя отдать в ансамбль. Схему не меняет."""
    if not path.is_file():
        return {
            "accepted": False,
            "reason": "NO_FEATURE_FILE",
            "detail": f"Нет {path}. Published 531 не заменяется reconstructed-вектором.",
            "rows": None,
            "feature_columns": None,
            "column_order": None,
            "sha256": None,
        }
    digest = sha256_file(path)
    columns = _header_columns(path)
    order = column_order_status(columns)
    feature_count = sum(name != "ecg_id" for name in columns)
    rows = _row_count(path)
    checksum_ok = digest == OFFICIAL_ECGDELI_SHA256
    if not checksum_ok:
        reason = "OTHER"
        detail = (
            "checksum mismatch with PTB-XL+ 1.0.1 ecgdeli_features.csv. "
            f"found {digest}, expected {OFFICIAL_ECGDELI_SHA256}. "
            "Файл в ансамбль не передаётся, 531 признаки заново не считаются."
        )
    elif rows <= 0:
        reason = "OTHER"
        detail = "В таблице признаков нет строк."
    elif order != "OK":
        reason = order
        detail = f"Заголовок не совпал с canonical schema: {order}."
    else:
        reason = "OK"
        detail = "published 531 table accepted"
    return {
        "accepted": reason == "OK",
        "reason": reason,
        "detail": detail,
        "rows": rows,
        "feature_columns": feature_count,
        "column_order": order,
        "sha256": digest,
    }


def require_official_feature_table(path: Path) -> None:
    report = classify_feature_table(path)
    if not report["accepted"]:
        raise FeatureTableError(str(report["detail"]))


def sanity_known_record(fixture: Path, model_dir: Path) -> dict[str, object]:
    del fixture, model_dir
    _ensemble_removed()
    return {}


def predict_published_table(table: Path, ecg_ids: list[str], model_dir: Path, destination: Path) -> None:
    del table, ecg_ids, model_dir, destination
    _ensemble_removed()
