from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from benchmark.config import ROOT, load_config, resolve
from benchmark.coverage import refuse_incomplete_for_ensemble
from benchmark.reference import regression_max_abs
from ecg_engine.ensemble import predict_csv
from ecg_engine.raw_extractor import RAW_TO_531_STATUS


def test_named_and_csv_inference_match():
    config = load_config()
    delta = regression_max_abs(resolve(config["paths"]["reference_fixture"]), resolve(config["paths"]["ensemble_dir"]))
    assert delta == 0.0
    assert RAW_TO_531_STATUS == "NOT_PROVEN"


def test_incomplete_mask_is_not_sent_to_ensemble():
    with pytest.raises(RuntimeError, match="ensemble"):
        refuse_incomplete_for_ensemble([1, 0, 1])


def test_matches_original_ecg_web_up_when_snapshot_exists():
    original = ROOT / "_audit" / "ecg_web_up" / "analysis_scripts" / "predict_csv.py"
    if not original.is_file():
        pytest.skip("Нет локального снимка ecg_web_up")
    spec = importlib.util.spec_from_file_location("ecg_web_up_predict_csv", original)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    fixture = resolve(load_config()["paths"]["reference_fixture"])
    models_dir = ROOT / "_audit" / "ecg_web_up" / "models"
    device = torch.device("cpu")
    mean = np.load(models_dir / "ecg_train_mean.npy")
    std = np.load(models_dir / "ecg_train_std.npy")
    frame = pd.read_csv(fixture)
    columns = [column for column in frame.columns if column != "ecg_id"]
    values = frame[columns].values.astype(np.float64)
    values = (values - mean) / std
    models = module.load_ensemble_models(device, str(models_dir))
    dense = torch.tensor(values, dtype=torch.float32)
    signal = torch.tensor(values[:, np.newaxis, :], dtype=torch.float32)
    with torch.no_grad():
        reference = module.get_ensemble_predictions(values, values[:, np.newaxis, :], models, device)[0]
    candidate = predict_csv(str(fixture), resolve(load_config()["paths"]["ensemble_dir"])).ensemble
    assert float(np.max(np.abs(reference - candidate))) == 0.0
