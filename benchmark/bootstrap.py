"""Bootstrap 95% CI. Единица пересэмплирования — patient_id, не отдельная ЭКГ."""

from __future__ import annotations

import numpy as np

from benchmark.metrics import binary_auroc, binary_auprc, macro_mean


def _groups(patient_ids: np.ndarray) -> dict[str, np.ndarray]:
    grouped: dict[str, list[int]] = {}
    for index, patient in enumerate(patient_ids):
        grouped.setdefault(str(patient), []).append(index)
    return {key: np.asarray(value, dtype=int) for key, value in grouped.items()}


def bootstrap_macro(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    patient_ids: np.ndarray,
    metric: str = "auroc",
    iterations: int = 1000,
    seed: int = 42,
) -> dict[str, float]:
    truth = np.asarray(y_true, dtype=np.float64)
    prob = np.asarray(y_prob, dtype=np.float64)
    groups = _groups(np.asarray(patient_ids))
    keys = np.asarray(list(groups))
    rng = np.random.default_rng(seed)
    scorer = binary_auroc if metric == "auroc" else binary_auprc
    samples: list[float] = []
    for _ in range(iterations):
        chosen = rng.choice(keys, size=len(keys), replace=True)
        index = np.concatenate([groups[key] for key in chosen])
        values = [
            scorer(truth[index, column], prob[index, column])
            for column in range(truth.shape[1])
        ]
        samples.append(macro_mean(values))
    array = np.asarray(samples, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if len(finite) == 0:
        return {"metric": metric, "mean": float("nan"), "lower_95": float("nan"), "upper_95": float("nan")}
    return {
        "metric": f"macro_{metric}",
        "mean": float(np.mean(finite)),
        "lower_95": float(np.percentile(finite, 2.5)),
        "upper_95": float(np.percentile(finite, 97.5)),
    }
