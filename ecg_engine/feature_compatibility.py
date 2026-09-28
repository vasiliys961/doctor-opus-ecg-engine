"""Raw ECG → 531 с маской происхождения. Это не копия генератора PTB-XL+."""

from __future__ import annotations

import math

import numpy as np

from ecg_engine.canonical_schema import FEATURE_COLUMNS_531
from ecg_engine.feature_registry import REGISTRY
from ecg_engine.preprocessing import arrange_leads
from ecg_engine.provenance import AnalysisBundle, FeatureProvenance
from ecg_engine.raw_extractor import RAW_TO_531_STATUS
from ecg_engine.validation import assert_bundle_shape, ensemble_allowed


class ECGInferenceAdapter:
    """Место для будущей модели. В этом шаге она не обучается."""

    def predict(self, feature_vector: object, feature_mask: object) -> None:
        raise NotImplementedError("ADAPTER_NOT_TRAINED")


class FeatureCompatibilityEngine:
    mode = "RECONSTRUCTED_RAW_MODE"

    def analyze(self, raw_signal: object, sampling_rate: float, lead_names: list[str]) -> dict[str, object]:
        _ordered, preprocessing = arrange_leads(raw_signal, lead_names, sampling_rate)
        values: list[float | None] = []
        mask: list[int] = []
        statuses: list[str] = []
        provenance: list[FeatureProvenance] = []
        for row in REGISTRY:
            status = str(row["status"])
            # Делинеации нет, поэтому число не создаётся. Статус семьи при этом сохраняется.
            values.append(None)
            mask.append(0)
            statuses.append(status)
            provenance.append(
                FeatureProvenance(
                    feature_name=str(row["feature"]),
                    family=str(row["family"]),
                    status=status,
                    method=str(row["method"]),
                    source=str(row["source"]),
                    exact_proven=bool(row["exact_proven"]),
                    confidence=str(row["confidence"]),
                    error=None,
                    notes=str(row["notes"]),
                )
            )
        names = [item.feature_name for item in provenance]
        assert_bundle_shape(names, values, mask, statuses)
        if ensemble_allowed(values, statuses):
            raise RuntimeError("Сырой режим не должен открывать существующий ансамбль.")
        bundle = AnalysisBundle(
            feature_vector=values,
            feature_names=FEATURE_COLUMNS_531,
            provenance=tuple(provenance),
            global_status=RAW_TO_531_STATUS,
        )
        counts = _counts(statuses)
        return {
            "status": self.mode,
            "raw_to_531_status": RAW_TO_531_STATUS,
            "compatibility": "COMPATIBILITY_ONLY",
            "feature_count": 531,
            "feature_mask": mask,
            "feature_status": statuses,
            "features": values,
            "provenance": [item.as_dict() for item in provenance],
            "counts": counts,
            "preprocessing": preprocessing.as_dict(),
            "analysis": bundle.as_dict(),
            "prediction": None,
        }


def _counts(statuses: list[str]) -> dict[str, int]:
    return {
        "reconstructed": sum(status == "RECONSTRUCTED" for status in statuses),
        "not_executed": sum(status == "NOT_EXECUTED" for status in statuses),
        "unknown": sum(status == "UNKNOWN" for status in statuses),
        "calculated": 0,
    }


def json_ready(payload: dict[str, object]) -> dict[str, object]:
    """NaN в JSON недопустим. None остаётся null и не означает ноль."""

    def convert(value: object) -> object:
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, dict):
            return {key: convert(item) for key, item in value.items()}
        if isinstance(value, list):
            return [convert(item) for item in value]
        if isinstance(value, tuple):
            return [convert(item) for item in value]
        if isinstance(value, np.generic):
            return convert(value.item())
        return value

    return convert(payload)  # type: ignore[return-value]
