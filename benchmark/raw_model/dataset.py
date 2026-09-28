"""Синтетический и файловый вход raw-модели. Метки в тензор сигнала не входят."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from benchmark.data import apply_lead_normalization, read_records500, validate_model_array
from benchmark.label_mapping import TARGET_CODES


class WaveformDataset(Dataset):
    def __init__(self, signals: np.ndarray, labels: np.ndarray, mean: np.ndarray, std: np.ndarray) -> None:
        if signals.ndim != 3 or signals.shape[1:] != (12, 5000):
            raise ValueError(f"Сигналы должны иметь форму [N, 12, 5000], получено {signals.shape}.")
        self.signals = signals.astype(np.float32)
        self.labels = labels.astype(np.float32)
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)

    def __len__(self) -> int:
        return int(self.signals.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        normalized = apply_lead_normalization(self.signals[index], self.mean, self.std)
        validate_model_array(normalized)
        return torch.from_numpy(normalized.astype(np.float32)), torch.from_numpy(self.labels[index])


def synthetic_split(records: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Детерминированный subset для smoke. Это не PTB-XL."""
    rng = np.random.default_rng(seed)
    signals = rng.normal(0.0, 0.2, size=(records, 12, 5000)).astype(np.float32)
    labels = np.zeros((records, len(TARGET_CODES)), dtype=np.float32)
    labels[:, 0] = (np.arange(records) % 2).astype(np.float32)
    labels[:, 1] = (np.arange(records) % 3 == 0).astype(np.float32)
    cut_train = max(records // 2, 2)
    cut_val = cut_train + max(records // 4, 1)
    parts = np.array(["train"] * cut_train + ["val"] * (cut_val - cut_train) + ["test"] * (records - cut_val))
    return signals, labels, parts


class LazyWaveformDataset(Dataset):
    """Читает одну запись в __getitem__. Весь корпус в RAM не собирается."""

    def __init__(self, headers: list[Path], labels: np.ndarray, mean: np.ndarray, std: np.ndarray) -> None:
        if len(headers) != len(labels):
            raise ValueError("Число заголовков и число целей не совпали.")
        self.headers = list(headers)
        self.labels = np.asarray(labels, dtype=np.float32)
        self.mean = np.asarray(mean, dtype=np.float64)
        self.std = np.asarray(std, dtype=np.float64)

    def __len__(self) -> int:
        return len(self.headers)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        signal = read_records500(self.headers[index])
        if signal.shape != (12, 5000):
            raise ValueError(f"Ожидалась форма [12, 5000], получено {signal.shape}. Resampling не выполняется.")
        normalized = apply_lead_normalization(signal, self.mean, self.std)
        validate_model_array(normalized)
        return (
            torch.from_numpy(normalized.astype(np.float32)),
            torch.from_numpy(self.labels[index]),
        )


def index_waveforms(records_root: Path) -> dict[str, Path]:
    found: dict[str, Path] = {}
    if not records_root.is_dir():
        return found
    for header in records_root.rglob("*_hr.hea"):
        stem = header.name.split("_", 1)[0]
        if not stem.isdigit():
            continue
        found[str(int(stem))] = header
    return found


def stream_lead_statistics(headers: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    """Один проход по файлам. Суммы копятся, матрица корпуса не хранится."""
    total = np.zeros(12, dtype=np.float64)
    total_sq = np.zeros(12, dtype=np.float64)
    count = 0
    for header in headers:
        signal = read_records500(header)
        if signal.shape != (12, 5000):
            raise ValueError(f"{header} имеет форму {signal.shape}.")
        total += signal.sum(axis=1)
        total_sq += np.square(signal).sum(axis=1)
        count += signal.shape[1]
    if count == 0:
        raise ValueError("Нет волн для train-статистик.")
    mean = total / count
    variance = np.maximum(total_sq / count - np.square(mean), 0.0)
    std = np.sqrt(variance)
    std = np.where(std < 1e-6, 1.0, std)
    return mean, std


def load_waveform_matrix(headers: list[Path]) -> np.ndarray:
    arrays = [read_records500(path) for path in headers]
    stacked = np.stack(arrays, axis=0)
    if stacked.shape[1:] != (12, 5000):
        raise ValueError(f"Собранная матрица имеет форму {stacked.shape}.")
    return stacked
