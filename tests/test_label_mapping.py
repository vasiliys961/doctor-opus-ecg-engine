from __future__ import annotations

import numpy as np

from benchmark.label_mapping import TARGET_CODES, load_mapping, scp_to_targets
from ecg_engine.scp import TOP_24_CODES


def test_targets_match_ensemble_order():
    assert TARGET_CODES == TOP_24_CODES
    assert len(TARGET_CODES) == 24
    table = load_mapping()
    assert list(table) == list(TARGET_CODES)
    assert table["SR"] == ["SR"]


def test_likelihood_zero_still_counts_as_present():
    targets = scp_to_targets({"SR": 0.0, "NORM": 100.0, "NOT_A_TARGET": 100.0})
    assert targets.shape == (24,)
    assert targets[TARGET_CODES.index("SR")] == 1.0
    assert targets[TARGET_CODES.index("NORM")] == 1.0
    assert targets[TARGET_CODES.index("AFIB")] == 0.0
    assert np.sum(targets) == 2
