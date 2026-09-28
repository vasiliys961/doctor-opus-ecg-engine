"""Происхождение одной колонки и пакета анализа. Не является сертификатом PTB-XL+."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class FeatureProvenance:
    feature_name: str
    family: str
    status: str
    method: str
    source: str
    exact_proven: bool
    confidence: str
    error: float | None
    notes: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class AnalysisBundle:
    feature_vector: list[float | None]
    feature_names: tuple[str, ...]
    provenance: tuple[FeatureProvenance, ...]
    global_status: str

    def as_dict(self) -> dict[str, object]:
        return {
            "feature_vector": self.feature_vector,
            "feature_names": list(self.feature_names),
            "provenance": [item.as_dict() for item in self.provenance],
            "global_status": self.global_status,
        }
