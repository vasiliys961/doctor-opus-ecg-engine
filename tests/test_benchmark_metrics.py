from __future__ import annotations

import numpy as np

from benchmark.bootstrap import bootstrap_macro
from benchmark.metrics import binary_auprc, binary_auroc, binary_counts, select_f1_threshold


def test_known_scores_match_hand_calculation():
    truth = np.array([1, 0, 1, 0])
    score = np.array([0.6, 0.7, 0.2, 0.1])
    assert binary_auroc(truth, score) == 0.5
    assert abs(binary_auprc(truth, score) - (0.5 + 2 / 3) / 2) < 1e-12
    counts = binary_counts(truth, score, 0.5)
    assert counts["TP"] == 1 and counts["FP"] == 1 and counts["FN"] == 1 and counts["TN"] == 1
    assert counts["precision"] == 0.5
    assert counts["recall"] == 0.5
    assert counts["specificity"] == 0.5
    assert counts["f1"] == 0.5
    perfect_truth = np.array([0, 0, 1, 1])
    perfect_score = np.array([0.1, 0.4, 0.8, 0.9])
    assert binary_auroc(perfect_truth, perfect_score) == 1.0


def test_bootstrap_interval_is_patient_level():
    truth = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    score = np.array([[0.9, 0.1], [0.8, 0.2], [0.2, 0.9], [0.1, 0.8]])
    # Дополняем до 24 столбцов нулями и единицами, чтобы macro не был пустым.
    extra_true = np.zeros((4, 22))
    extra_score = np.zeros((4, 22))
    extra_true[:, 0] = 1
    extra_score[:, 0] = 1
    result = bootstrap_macro(
        np.concatenate([truth, extra_true], axis=1),
        np.concatenate([score, extra_score], axis=1),
        np.array(["a", "a", "b", "b"]),
        iterations=30,
        seed=42,
    )
    assert result["lower_95"] <= result["mean"] <= result["upper_95"]


def test_threshold_is_chosen_on_the_given_split_only():
    truth = np.array([0, 1, 0, 1])
    score = np.array([0.2, 0.8, 0.4, 0.9])
    chosen = select_f1_threshold(truth, score, np.array([0.5, 0.85]))
    assert chosen == 0.5

