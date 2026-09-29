from __future__ import annotations

import pytest

from benchmark.coverage import refuse_incomplete_for_ensemble


def test_incomplete_mask_is_not_sent_to_ensemble():
    with pytest.raises(RuntimeError, match="ensemble"):
        refuse_incomplete_for_ensemble([1, 0, 1])
