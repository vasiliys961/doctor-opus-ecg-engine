"""Test-оценка raw-модели. Пороги берутся только из validation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from benchmark.label_mapping import TARGET_CODES
from benchmark.metrics import (
    binary_auprc,
    binary_auroc,
    binary_counts,
    expected_calibration_error,
    macro_mean,
    micro_f1,
    select_f1_threshold,
    threshold_grid,
)
from benchmark.raw_model.dataset import WaveformDataset
from benchmark.raw_model.model import RawECGCNN
from benchmark.raw_model.train import pick_device


def _loader(signals: np.ndarray, labels: np.ndarray, mean: np.ndarray, std: np.ndarray) -> DataLoader:
    return DataLoader(WaveformDataset(signals, labels, mean, std), batch_size=8, shuffle=False, num_workers=0)


def collect(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    truths: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for waveform, labels in loader:
            logits = model(waveform.to(device))
            scores.append(torch.sigmoid(logits).cpu().numpy())
            truths.append(labels.numpy())
    return np.concatenate(truths), np.concatenate(scores)


def freeze_thresholds(y_true: np.ndarray, y_prob: np.ndarray, config: dict) -> list[float]:
    grid = threshold_grid(
        float(config["threshold_selection"]["grid_min"]),
        float(config["threshold_selection"]["grid_max"]),
        float(config["threshold_selection"]["grid_step"]),
    )
    return [select_f1_threshold(y_true[:, index], y_prob[:, index], grid) for index in range(y_true.shape[1])]


def metric_rows(model_name: str, y_true: np.ndarray, y_prob: np.ndarray, thresholds: list[float]) -> list[dict[str, object]]:
    rows = []
    for index, code in enumerate(TARGET_CODES):
        counts = binary_counts(y_true[:, index], y_prob[:, index], thresholds[index])
        positive = int(y_true[:, index].sum())
        negative = int(len(y_true) - positive)
        rows.append(
            {
                "model": model_name,
                "label": code,
                "n_positive": positive,
                "n_negative": negative,
                "prevalence": positive / len(y_true) if len(y_true) else float("nan"),
                "auroc": binary_auroc(y_true[:, index], y_prob[:, index]),
                "auprc": binary_auprc(y_true[:, index], y_prob[:, index]),
                "f1": counts["f1"],
                "precision": counts["precision"],
                "recall": counts["recall"],
                "specificity": counts["specificity"],
                "sensitivity": counts["sensitivity"],
                "brier": counts["brier"],
                "ece": expected_calibration_error(y_true[:, index], y_prob[:, index]),
                "threshold": counts["threshold"],
                "TP": counts["TP"],
                "TN": counts["TN"],
                "FP": counts["FP"],
                "FN": counts["FN"],
            }
        )
    return rows


def summary_from_rows(model_name: str, rows: list[dict[str, object]], y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, object]:
    def column(name: str) -> list[float]:
        return [float(row[name]) for row in rows]

    return {
        "model": model_name,
        "macro_auroc": macro_mean(column("auroc")),
        "macro_auprc": macro_mean(column("auprc")),
        "macro_f1": macro_mean(column("f1")),
        "macro_precision": macro_mean(column("precision")),
        "macro_recall": macro_mean(column("recall")),
        "macro_specificity": macro_mean(column("specificity")),
        "macro_sensitivity": macro_mean(column("sensitivity")),
        "micro_f1": micro_f1(y_true, y_pred),
        "macro_brier": macro_mean(column("brier")),
        "macro_ece": macro_mean(column("ece")),
    }


def load_model(checkpoint: Path, device: torch.device) -> RawECGCNN:
    model = RawECGCNN(num_classes=len(TARGET_CODES))
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model
