"""Метрики multilabel. Порог для F1 выбирается только на validation."""

from __future__ import annotations

import numpy as np

def _ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        # Средний ранг для совпавших значений, ранги с 1.
        average = (start + 1 + stop) / 2.0
        ranks[order[start:stop]] = average
        start = stop
    return ranks


def binary_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    truth = np.asarray(y_true).astype(int)
    score = np.asarray(y_score, dtype=np.float64)
    n_pos = int(truth.sum())
    n_neg = int(len(truth) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    sum_pos = _ranks(score)[truth == 1].sum()
    return float((sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def binary_auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Average precision: среднее precision в позициях положительных объектов."""
    truth = np.asarray(y_true).astype(int)
    score = np.asarray(y_score, dtype=np.float64)
    n_pos = int(truth.sum())
    if n_pos == 0 or n_pos == len(truth):
        return float("nan")
    order = np.argsort(-score, kind="mergesort")
    hits = 0
    total = 0.0
    for position, index in enumerate(order, start=1):
        if truth[index] == 1:
            hits += 1
            total += hits / position
    return float(total / n_pos)


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    truth = np.asarray(y_true).astype(int)
    pred = np.asarray(y_pred).astype(int)
    tp = int(((truth == 1) & (pred == 1)).sum())
    tn = int(((truth == 0) & (pred == 0)).sum())
    fp = int(((truth == 0) & (pred == 1)).sum())
    fn = int(((truth == 1) & (pred == 0)).sum())
    return {"TP": tp, "TN": tn, "FP": fp, "FN": fn}


def _ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return float("nan")
    return numerator / denominator


def binary_counts(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, float]:
    truth = np.asarray(y_true).astype(int)
    prob = np.asarray(y_prob, dtype=np.float64)
    pred = (prob >= threshold).astype(int)
    counts = confusion(truth, pred)
    precision = _ratio(counts["TP"], counts["TP"] + counts["FP"])
    recall = _ratio(counts["TP"], counts["TP"] + counts["FN"])
    specificity = _ratio(counts["TN"], counts["TN"] + counts["FP"])
    if np.isnan(precision) or np.isnan(recall) or precision + recall == 0:
        f1 = float("nan")
    else:
        f1 = 2 * precision * recall / (precision + recall)
    brier = float(np.mean((prob - truth) ** 2))
    return {
        **counts,
        "precision": precision,
        "recall": recall,
        "sensitivity": recall,
        "specificity": specificity,
        "f1": f1,
        "brier": brier,
        "threshold": float(threshold),
    }


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 10) -> float:
    truth = np.asarray(y_true).astype(int)
    prob = np.asarray(y_prob, dtype=np.float64)
    if len(truth) == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        if right == 1.0:
            mask = (prob >= left) & (prob <= right)
        else:
            mask = (prob >= left) & (prob < right)
        if not np.any(mask):
            continue
        accuracy = float(truth[mask].mean())
        confidence = float(prob[mask].mean())
        total += abs(accuracy - confidence) * (mask.sum() / len(truth))
    return float(total)


def threshold_grid(minimum: float = 0.01, maximum: float = 0.99, step: float = 0.01) -> np.ndarray:
    return np.round(np.arange(minimum, maximum + step / 2, step), 10)


def select_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray, grid: np.ndarray | None = None) -> float:
    """Порог с максимальным F1 на validation. Ничья берёт меньший порог."""
    candidates = threshold_grid() if grid is None else np.asarray(grid, dtype=np.float64)
    best_threshold = float(candidates[0])
    best_f1 = -1.0
    for threshold in candidates:
        score = binary_counts(y_true, y_prob, float(threshold))["f1"]
        if np.isnan(score):
            continue
        if score > best_f1:
            best_f1 = float(score)
            best_threshold = float(threshold)
    return best_threshold


def macro_mean(values: list[float]) -> float:
    finite = [value for value in values if not np.isnan(value)]
    if not finite:
        return float("nan")
    return float(np.mean(finite))


def micro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    counts = confusion(y_true.reshape(-1), y_pred.reshape(-1))
    precision = _ratio(counts["TP"], counts["TP"] + counts["FP"])
    recall = _ratio(counts["TP"], counts["TP"] + counts["FN"])
    if np.isnan(precision) or np.isnan(recall) or precision + recall == 0:
        return float("nan")
    return float(2 * precision * recall / (precision + recall))
