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


def require_official_feature_table(path: Path) -> None:
    if not path.is_file():
        raise FeatureTableError(
            f"Нет {path}. Ожидается PTB-XL+ 1.0.1 features/ecgdeli_features.csv, "
            f"sha256 {OFFICIAL_ECGDELI_SHA256}. Таблица не скачивается автоматически."
        )
    found = sha256_file(path)
    if found != OFFICIAL_ECGDELI_SHA256:
        raise FeatureTableError(
            "Контрольная сумма ecgdeli_features.csv не совпала с официальной. "
            "Файл не используется, 531 признаки заново не считаются."
        )


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
