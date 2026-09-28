"""Оценка уже обученного ансамбля на опубликованном 531-векторе. Веса не меняются."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from benchmark.data import OFFICIAL_ECGDELI_SHA256
from benchmark.label_mapping import TARGET_CODES
from ecg_engine.ensemble import predict_csv, predict_features
from ecg_engine.feature_columns import FEATURE_COLUMNS


class FeatureTableError(RuntimeError):
    """Опубликованная таблица 531 непригодна."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def regression_max_abs(fixture: Path, model_dir: Path) -> float:
    """Именованный вход benchmark и CSV-вход текущего inference на одной строке."""
    from_csv = predict_csv(str(fixture), model_dir).ensemble
    frame = pd.read_csv(fixture)
    row = frame.iloc[0]
    features = {name: row[name] for name in FEATURE_COLUMNS}
    from_names = predict_features(features, model_dir).ensemble
    return float(np.max(np.abs(from_csv - from_names)))


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
    """Один известный ECG. Это не статистический benchmark."""
    frame = pd.read_csv(fixture)
    row = frame.iloc[0]
    values = np.array([float(row[name]) for name in FEATURE_COLUMNS], dtype=np.float64)
    if values.shape != (531,):
        raise FeatureTableError(f"Sanity shape {values.shape}, ожидалось (531,).")
    if not np.isfinite(values).all():
        raise FeatureTableError("Sanity vector содержит NaN или Inf.")
    probabilities = predict_features({name: row[name] for name in FEATURE_COLUMNS}, model_dir).ensemble
    if probabilities.shape != (24,):
        raise FeatureTableError(f"Sanity prediction shape {probabilities.shape}.")
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0) or np.any(probabilities > 1):
        raise FeatureTableError("Sanity probabilities вне [0, 1].")
    return {
        "role": "sanity_not_benchmark",
        "ecg_id": str(row["ecg_id"]) if "ecg_id" in frame.columns else None,
        "shape": 531,
        "prediction_shape": 24,
        "probability_range": [float(probabilities.min()), float(probabilities.max())],
    }


def predict_published_table(table: Path, ecg_ids: list[str], model_dir: Path, destination: Path) -> None:
    require_official_feature_table(table)
    frame = pd.read_csv(table)
    if "ecg_id" not in frame.columns:
        raise FeatureTableError("В таблице признаков нет ecg_id.")
    missing = [name for name in FEATURE_COLUMNS if name not in frame.columns]
    if missing:
        raise FeatureTableError(f"В таблице нет колонок схемы, первая: {missing[0]}.")
    indexed = frame.set_index(frame["ecg_id"].map(lambda value: str(int(value))))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ecg_id", *[f"{code}_prob" for code in TARGET_CODES]])
        for ecg_id in ecg_ids:
            if ecg_id not in indexed.index:
                raise FeatureTableError(f"Для ecg_id {ecg_id} нет опубликованной строки 531.")
            row = indexed.loc[ecg_id]
            features = {name: row[name] for name in FEATURE_COLUMNS}
            probabilities = predict_features(features, model_dir).ensemble
            writer.writerow([ecg_id, *[float(value) for value in probabilities]])
