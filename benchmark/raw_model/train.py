"""Обучение raw CNN. Test в выбор модели и в пороги не входит."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from benchmark.data import train_lead_statistics
from benchmark.label_mapping import TARGET_CODES, validate_binary_targets
from benchmark.metrics import binary_auprc, binary_auroc, macro_mean
from benchmark.raw_model.dataset import LazyWaveformDataset, WaveformDataset, stream_lead_statistics
from benchmark.raw_model.model import RawECGCNN


def pick_device(requested: str | None = None) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    if requested == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise RuntimeError("MPS requested but not available")
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ensure_benchmark_device(requested: str | None = None) -> torch.device:
    """Ошибка Apple MPS переводит только benchmark на CPU. Production ensemble не меняется."""
    device = pick_device(requested)
    if device.type != "mps":
        return device
    try:
        probe = RawECGCNN().to(device)
        loss = probe(torch.zeros(1, 12, 64, device=device)).sum()
        loss.backward()
        return device
    except RuntimeError as exc:
        print(f"MPS error, benchmark fallback to CPU: {exc}")
        return torch.device("cpu")


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _loader(signals: np.ndarray, labels: np.ndarray, mean: np.ndarray, std: np.ndarray, batch_size: int) -> DataLoader:
    dataset = WaveformDataset(signals, labels, mean, std)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)


def _loss_function(config: dict, train_labels: np.ndarray) -> nn.Module:
    if not config.get("class_weighting"):
        return nn.BCEWithLogitsLoss()
    weights = []
    for column in range(train_labels.shape[1]):
        positive = float(train_labels[:, column].sum())
        negative = float(len(train_labels) - positive)
        weights.append(1.0 if positive == 0 else negative / positive)
    return nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weights, dtype=torch.float32))


def probabilities(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    truths: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    with torch.no_grad():
        for waveform, labels in loader:
            logits = model(waveform.to(device))
            scores.append(torch.sigmoid(logits).cpu().numpy())
            truths.append(labels.numpy())
    return np.concatenate(truths), np.concatenate(scores)


def macro_auroc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return macro_mean([binary_auroc(y_true[:, index], y_prob[:, index]) for index in range(y_true.shape[1])])


def _fit_loaders(
    train_loader: DataLoader,
    val_loader: DataLoader,
    train_labels: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    config: dict,
    output_dir: Path,
    device: torch.device,
) -> dict[str, object]:
    model = RawECGCNN(num_classes=len(TARGET_CODES)).to(device)
    criterion = _loss_function(config, train_labels).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "training_history.csv"
    best_score = -1.0
    best_epoch = 0
    patience = int(config["early_stopping"]["patience"])
    stale = 0
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["epoch", "train_loss", "val_loss", "val_macro_auroc", "val_macro_auprc", "learning_rate"],
        )
        writer.writeheader()
        for epoch in range(1, int(config["epochs"]) + 1):
            model.train()
            total = 0.0
            seen = 0
            for waveform, target in train_loader:
                waveform = waveform.to(device)
                target = target.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(waveform), target)
                loss.backward()
                optimizer.step()
                total += float(loss.item()) * len(target)
                seen += len(target)
            model.eval()
            val_loss = 0.0
            val_seen = 0
            with torch.no_grad():
                for waveform, target in val_loader:
                    waveform = waveform.to(device)
                    target = target.to(device)
                    loss = criterion(model(waveform), target)
                    val_loss += float(loss.item()) * len(target)
                    val_seen += len(target)
            truth, score = probabilities(model, val_loader, device)
            val_auroc = macro_auroc(truth, score)
            val_auprc = macro_mean(
                [binary_auprc(truth[:, index], score[:, index]) for index in range(truth.shape[1])]
            )
            writer.writerow(
                {
                    "epoch": epoch,
                    "train_loss": total / max(seen, 1),
                    "val_loss": val_loss / max(val_seen, 1),
                    "val_macro_auroc": val_auroc,
                    "val_macro_auprc": val_auprc,
                    "learning_rate": config["learning_rate"],
                }
            )
            if np.isfinite(val_auroc) and val_auroc > best_score:
                best_score = float(val_auroc)
                best_epoch = epoch
                stale = 0
                torch.save(model.state_dict(), output_dir / "best_model.pth")
            else:
                stale += 1
                if stale >= patience:
                    break
    metadata = {
        "architecture": "RawECGCNN",
        "input": [12, 5000],
        "outputs": list(TARGET_CODES),
        "normalization": "per-lead mean/std on train only",
        "train_mean": mean.tolist(),
        "train_std": std.tolist(),
        "seed": config["seed"],
        "device": str(device),
        "best_epoch": best_epoch,
        "best_val_macro_auroc": best_score,
        "augmentation": False,
        "loss": config["loss"],
        "optimizer": config["optimizer"],
    }
    (output_dir / "model_config.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    history = (output_dir / "training_history.csv").read_text(encoding="utf-8").strip().splitlines()
    checkpoint = output_dir / "best_model.pth"
    if len(history) < 2 or not checkpoint.is_file() or int(metadata["best_epoch"]) < 1:
        raise RuntimeError("Обучение не прогрессировало: нет checkpoint, history или выбранной модели.")
    return metadata


def _fit_with_fallback(
    train_loader: DataLoader,
    val_loader: DataLoader,
    train_labels: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    config: dict,
    output_dir: Path,
    device: torch.device,
) -> dict[str, object]:
    try:
        return _fit_loaders(train_loader, val_loader, train_labels, mean, std, config, output_dir, device)
    except RuntimeError as exc:
        if device.type != "mps":
            raise
        print(f"MPS error, benchmark fallback to CPU: {exc}")
        cpu = torch.device("cpu")
        return _fit_loaders(train_loader, val_loader, train_labels, mean, std, config, output_dir, cpu)


def fit_raw_model(
    signals: np.ndarray,
    labels: np.ndarray,
    parts: np.ndarray,
    config: dict,
    output_dir: Path,
) -> dict[str, object]:
    if config.get("augmentation"):
        raise RuntimeError("Augmentation в основном benchmark выключена.")
    device = ensure_benchmark_device(config.get("device"))
    set_seed(int(config["seed"]))
    train_mask = parts == "train"
    val_mask = parts == "val"
    validate_binary_targets(labels[train_mask], "train")
    validate_binary_targets(labels[val_mask], "val")
    mean, std = train_lead_statistics(signals[train_mask])
    train_loader = _loader(signals[train_mask], labels[train_mask], mean, std, int(config["batch_size"]))
    val_loader = _loader(signals[val_mask], labels[val_mask], mean, std, int(config["batch_size"]))
    return _fit_with_fallback(
        train_loader,
        val_loader,
        labels[train_mask],
        mean,
        std,
        config,
        output_dir,
        device,
    )


def fit_lazy_records(
    train_headers: list[Path],
    train_labels: np.ndarray,
    val_headers: list[Path],
    val_labels: np.ndarray,
    config: dict,
    output_dir: Path,
) -> dict[str, object]:
    """Обучение по файлам records500. В RAM лежит один batch, не весь корпус."""
    if config.get("augmentation"):
        raise RuntimeError("Augmentation в основном benchmark выключена.")
    device = ensure_benchmark_device(config.get("device"))
    set_seed(int(config["seed"]))
    validate_binary_targets(train_labels, "train")
    validate_binary_targets(val_labels, "val")
    mean, std = stream_lead_statistics(train_headers)
    batch_size = int(config["batch_size"])
    train_loader = DataLoader(
        LazyWaveformDataset(train_headers, train_labels, mean, std),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    val_loader = DataLoader(
        LazyWaveformDataset(val_headers, val_labels, mean, std),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    return _fit_with_fallback(train_loader, val_loader, train_labels, mean, std, config, output_dir, device)
