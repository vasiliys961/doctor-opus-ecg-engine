"""Inference для обученного ансамбля 531 → 24 SCP.

Веса и формула ансамбля перенесены из ecg_web_up без переобучения.
"""

from ecg_engine.ensemble import predict_csv
from ecg_engine.schema import ECG531Input, ECGSchemaError
from ecg_engine.version import MODEL_VERSION

__all__ = ["ECG531Input", "ECGSchemaError", "MODEL_VERSION", "predict_csv"]
