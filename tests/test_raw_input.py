from __future__ import annotations

import numpy as np
import pytest
import torch

from benchmark.data import CANONICAL_LEADS, WaveformError, validate_model_array
from benchmark.data_validation import DatasetValidationError, validate_record
from benchmark.raw_model.model import RawECGCNN


def test_canonical_shape_is_accepted_and_other_shapes_fail():
    validate_model_array(np.zeros((12, 5000)))
    validate_model_array(np.zeros((2, 12, 5000)))
    with pytest.raises(WaveformError):
        validate_model_array(np.zeros((5000, 12)))
    with pytest.raises(WaveformError):
        validate_model_array(np.full((12, 5000), np.nan))


def test_lead_order_is_not_reordered():
    signal = np.zeros((12, 5000))
    names = list(CANONICAL_LEADS)
    names[0], names[1] = names[1], names[0]
    with pytest.raises(DatasetValidationError):
        validate_record(signal, 500, names, "1", "2", {"NORM": 100.0})


def test_forward_and_backward_on_one_batch():
    model = RawECGCNN()
    batch = torch.zeros(2, 12, 5000)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(model(batch), torch.zeros(2, 24))
    loss.backward()
    assert model.head.weight.grad is not None
