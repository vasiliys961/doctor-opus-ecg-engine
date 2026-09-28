"""Сверка голов и ансамбля с немодифицированным ecg_web_up."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from ecg_engine.ensemble import predict_csv

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "fixtures" / "another_ecg_features.csv"
ORIGINAL_ROOT = ROOT / "_audit" / "ecg_web_up"
ORIGINAL_MODELS = ORIGINAL_ROOT / "models"
NEW_MODELS = ROOT / "models" / "ecg_ensemble"
REPORT = Path(__file__).resolve().parent / "baseline_comparison.json"


def _load_original():
    module_path = ORIGINAL_ROOT / "analysis_scripts" / "predict_csv.py"
    if not module_path.is_file():
        pytest.skip(f"Нет локального снимка ecg_web_up: {module_path}")
    spec = importlib.util.spec_from_file_location("ecg_web_up_predict_csv", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _original_outputs(module):
    device = torch.device("cpu")
    mean = np.load(ORIGINAL_MODELS / "ecg_train_mean.npy")
    std = np.load(ORIGINAL_MODELS / "ecg_train_std.npy")
    frame = pd.read_csv(FIXTURE)
    columns = [column for column in frame.columns if column != "ecg_id"]
    values = frame[columns].values
    if np.isnan(values).any() or np.isinf(values).any():
        values = np.where(np.isnan(values), mean, values)
        values = np.nan_to_num(values, nan=0.0, posinf=1e6, neginf=-1e6)
    values = (values - mean) / std
    values = np.nan_to_num(values, nan=0.0, posinf=1e6, neginf=-1e6)
    models = module.load_ensemble_models(device, str(ORIGINAL_MODELS))
    dense = torch.tensor(values, dtype=torch.float32)
    signal = torch.tensor(values[:, np.newaxis, :], dtype=torch.float32)
    heads = {}
    with torch.no_grad():
        heads["MLP"] = torch.sigmoid(models["MLP"](dense)).cpu().numpy()[0]
        heads["CNN"] = torch.sigmoid(models["CNN"](signal)).cpu().numpy()[0]
        heads["ResNet"] = torch.sigmoid(models["ResNet"](signal)).cpu().numpy()[0]
    ensemble = module.get_ensemble_predictions(values, values[:, np.newaxis, :], models, device)[0]
    return heads, ensemble


def _diff(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    delta = np.abs(reference.astype(np.float64) - candidate.astype(np.float64))
    return {
        "max_absolute_difference": float(delta.max()),
        "mean_absolute_difference": float(delta.mean()),
    }


def test_heads_and_ensemble_match_ecg_web_up():
    module = _load_original()
    reference_heads, reference_ensemble = _original_outputs(module)
    candidate = predict_csv(str(FIXTURE), NEW_MODELS)
    report = {
        "fixture": "tests/fixtures/another_ecg_features.csv",
        "ecg_id": "513",
        "original": "ecg_web_up@bfe7c1738baca97f9f501ede25f1716f2aa94cf6",
        "candidate_model_dir": "models/ecg_ensemble",
        "heads": {
            name: _diff(reference_heads[name], candidate.heads[name])
            for name in ("MLP", "CNN", "ResNet")
        },
        "ensemble": _diff(reference_ensemble, candidate.ensemble),
    }
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for name, stats in report["heads"].items():
        assert stats["max_absolute_difference"] < 1e-5, name
    assert report["ensemble"]["max_absolute_difference"] < 1e-5
