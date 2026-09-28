"""Загрузка весов и инференс ансамбля.

Усреднение совпадает с ecg_web_up: равное среднее сигмоид трёх голов.
Пропуск в признаках здесь не заполняется. Исходный скрипт подставлял train_mean;
для конечной полной строки формула z-score совпадает.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from ecg_engine.networks import ECG1DCNN, create_mlp_model, resnet18_1d
from ecg_engine.schema import ECG531Input, ECGSchemaError, read_ecg531_csv
from ecg_engine.scp import TOP_24_CODES
from ecg_engine.version import (
    FEATURE_SCHEMA_VERSION,
    INPUT_SCHEMA,
    MODEL_VERSION,
    PREPROCESSING_VERSION,
    SOURCE_COMMIT,
)

HEAD_FILES = {
    "MLP": "ecg_model.pth",
    "CNN": "ecg_1dcnn_best.pth",
    "ResNet": "ecg_resnet1d_features_best.pth",
}
REQUIRED_HEADS = ("MLP", "CNN", "ResNet")


class ModelAssetsError(FileNotFoundError):
    """Нет весов или файлов нормализации."""


@dataclass(frozen=True)
class EnsemblePrediction:
    ensemble: np.ndarray
    heads: dict[str, np.ndarray]
    row_index: int
    rows_in_file: int

    def as_json(self) -> dict:
        return {
            "model_version": MODEL_VERSION,
            "source_commit": SOURCE_COMMIT,
            "input_schema": INPUT_SCHEMA,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "preprocessing_version": PREPROCESSING_VERSION,
            "row_index": self.row_index,
            "rows_in_file": self.rows_in_file,
            "predictions": _scores(self.ensemble),
            "heads": {name: _scores(values) for name, values in self.heads.items()},
        }


def default_model_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "models" / "ecg_ensemble"


def _scores(vector: np.ndarray) -> dict[str, float]:
    if vector.shape != (len(TOP_24_CODES),):
        raise RuntimeError(f"Ожидался вектор из {len(TOP_24_CODES)} выходов, получено {vector.shape}.")
    return {code: float(vector[index]) for index, code in enumerate(TOP_24_CODES)}


def _state_dict(checkpoint: object) -> dict:
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
            raw = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint and isinstance(checkpoint["model_state_dict"], dict):
            raw = checkpoint["model_state_dict"]
        else:
            raw = checkpoint
    else:
        raise ModelAssetsError("Чекпоинт не является словарём весов.")
    return {key[4:] if key.startswith("net.") else key: value for key, value in raw.items()}


def _load_normalization(model_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    mean_path = model_dir / "ecg_train_mean.npy"
    std_path = model_dir / "ecg_train_std.npy"
    if not mean_path.is_file() or not std_path.is_file():
        raise ModelAssetsError(f"Нет файлов нормализации в {model_dir}.")
    mean = np.load(mean_path)
    std = np.load(std_path)
    if mean.shape != (1, 531) or std.shape != (1, 531):
        raise ModelAssetsError(f"Ожидались mean/std формы (1, 531), получено {mean.shape} и {std.shape}.")
    if np.any(std == 0):
        raise ModelAssetsError("В ecg_train_std.npy есть нули. Файл нормализации расходится с обучением.")
    return mean, std


def normalize(values: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Z-score обученной выборки. values имеет форму (531,)."""
    if not np.isfinite(values).all():
        raise ECGSchemaError("Нормализация получила NaN или Inf. Пропуски не заполняются.")
    normalized = (values.reshape(1, -1) - mean) / std
    return normalized.reshape(-1)


def _build(name: str) -> torch.nn.Module:
    if name == "MLP":
        return create_mlp_model(input_dim=531, num_classes=24)
    if name == "CNN":
        return ECG1DCNN(num_classes=24, input_channels=1, input_length=531)
    if name == "ResNet":
        return resnet18_1d(num_classes=24, input_channels=1, input_length=531)
    raise ModelAssetsError(f"Неизвестная голова ансамбля: {name}")


class Ensemble:
    def __init__(self, model_dir: Path | None = None):
        self.model_dir = Path(model_dir) if model_dir else default_model_dir()
        self.device = torch.device("cpu")
        self.mean, self.std = _load_normalization(self.model_dir)
        self.models = self._load_models()

    def _load_models(self) -> dict[str, torch.nn.Module]:
        models: dict[str, torch.nn.Module] = {}
        missing = []
        for name, filename in HEAD_FILES.items():
            path = self.model_dir / filename
            if not path.is_file():
                missing.append(str(path))
                continue
            model = _build(name).to(self.device)
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            model.load_state_dict(_state_dict(checkpoint))
            model.eval()
            models[name] = model
        if missing or tuple(models) != REQUIRED_HEADS:
            absent = ", ".join(missing) or ", ".join(set(REQUIRED_HEADS) - set(models))
            raise ModelAssetsError(f"Ансамбль требует все три головы. Не загружено: {absent}.")
        return models

    def predict(self, sample: ECG531Input) -> EnsemblePrediction:
        normalized = normalize(sample.values, self.mean, self.std)
        dense = torch.tensor(normalized.reshape(1, -1), dtype=torch.float32, device=self.device)
        signal = dense.unsqueeze(1)
        heads: dict[str, np.ndarray] = {}
        with torch.no_grad():
            for name in REQUIRED_HEADS:
                logits = self.models[name](dense if name == "MLP" else signal)
                heads[name] = torch.sigmoid(logits).cpu().numpy()[0]
        stacked = np.stack([heads[name] for name in REQUIRED_HEADS], axis=0)
        ensemble = np.mean(stacked, axis=0)
        return EnsemblePrediction(
            ensemble=ensemble,
            heads=heads,
            row_index=sample.row_index,
            rows_in_file=sample.rows_in_file,
        )


@lru_cache(maxsize=4)
def get_ensemble(model_dir: str) -> Ensemble:
    return Ensemble(Path(model_dir))


def predict_csv(csv_path: str, model_dir: str | os.PathLike | None = None) -> EnsemblePrediction:
    directory = str(Path(model_dir) if model_dir else default_model_dir())
    return get_ensemble(directory).predict(read_ecg531_csv(csv_path))
