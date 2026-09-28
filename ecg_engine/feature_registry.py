"""Статус каждой из 531 колонок по forensic-разбору. Это не признание тождества с PTB-XL+."""

from __future__ import annotations

from ecg_engine.canonical_schema import FEATURE_COLUMNS_531

_LEADS = (
    "III",
    "II",
    "aVF",
    "aVL",
    "aVR",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
    "Global",
    "I",
)

_INTERVALS = frozenset(
    {
        "PQ_Int",
        "PR_Int",
        "QRS_Dur",
        "QT_Int",
        "P_DurFull",
        "T_DurFull",
        "P_Dur",
        "T_Dur",
        "RR_Mean",
        "QT_IntFramingham",
    }
)
_AMPLITUDES = frozenset({"P_Amp", "Q_Amp", "R_Amp", "S_Amp", "T_Amp"})


def split_feature(name: str) -> tuple[str, str, str]:
    if name.startswith("HA__"):
        stat = "value"
        for suffix, label in (("_iqr", "iqr"), ("_count", "count")):
            if name.endswith(suffix):
                return "HA", "Global", label
        return "HA", "Global", stat
    stat = "value"
    base = name
    for suffix, label in (("_iqr", "iqr"), ("_count", "count")):
        if name.endswith(suffix):
            stat = label
            base = name[: -len(suffix)]
            break
    for lead in _LEADS:
        token = f"_{lead}"
        if base.endswith(token):
            return base[: -len(token)], lead, stat
    raise KeyError(name)


def _meta(family: str) -> dict[str, str | bool]:
    if family in _INTERVALS:
        return {
            "source": "ECGDeli 1.1 ExtractIntervalFeaturesFromFPT and published fiducials",
            "status": "RECONSTRUCTED",
            "method": "ExtractIntervalFeaturesFromFPT",
            "confidence": "HIGH",
            "exact_proven": False,
            "notes": (
                "Algorithm reconstructed and validated against published PTB-XL+ row 00513. "
                "238 cells matched at <=1e-6 in forensic reconstruction."
            ),
        }
    if family in _AMPLITUDES:
        return {
            "source": "ECGDeli 1.1 ExtractAmplitudeFeaturesFromFPT",
            "status": "RECONSTRUCTED",
            "method": "ExtractAmplitudeFeaturesFromFPT",
            "confidence": "MEDIUM",
            "exact_proven": False,
            "notes": (
                "Public ECGDeli implementation identified. "
                "Original author preprocessing and signal-conditioning pipeline not fully recovered."
            ),
        }
    if family == "P_Morph":
        return {
            "source": "ECGDeli 1.1 Get_P_Morphology",
            "status": "NOT_EXECUTED",
            "method": "Get_P_Morphology",
            "confidence": "UNKNOWN",
            "exact_proven": False,
            "notes": (
                "Function identified in ECGDeli. "
                "Required inputs are signal, sampling frequency and FPT. "
                "Function was not executed because MATLAB/Octave/MATLAB Runtime were unavailable."
            ),
        }
    if family == "QT_IntCorr":
        return {
            "source": "lead-wise QT and padded lead RR through the public Framingham expression",
            "status": "RECONSTRUCTED",
            "method": "Framingham per beat, then median and IQR",
            "confidence": "HIGH",
            "exact_proven": False,
            "notes": (
                "Formula reconstructed from published ECGDeli/PTB-XL+ evidence. "
                "Numerical residual remains below approximately 0.0055 ms for record 00513 "
                "but strict 1e-6 equivalence was not achieved."
            ),
        }
    if family == "ST_Elev":
        return {
            "source": "feature dictionary name STc_X; no public ECGDeli 1.1 function",
            "status": "UNKNOWN",
            "method": "none",
            "confidence": "UNKNOWN",
            "exact_proven": False,
            "notes": (
                "STc_X is named in the feature dictionary. "
                "The original algorithm was not found. "
                "ECGDeli 1.1 does not expose a corresponding public feature function. "
                "Attempted raw and fiducial reconstructions did not reproduce the published values."
            ),
        }
    if family == "HA":
        return {
            "source": "feature dictionary name elHA; no public ECGDeli 1.1 function",
            "status": "UNKNOWN",
            "method": "none",
            "confidence": "UNKNOWN",
            "exact_proven": False,
            "notes": (
                "elHA implementation was not recovered. "
                "An exact mapping from raw ECG to the published class was not established. "
                "PTB-XL metadata does not provide heart_axis for record 00513."
            ),
        }
    raise KeyError(family)


def feature_registry() -> tuple[dict[str, object], ...]:
    rows = []
    for feature in FEATURE_COLUMNS_531:
        family, _lead, _stat = split_feature(feature)
        meta = _meta(family)
        rows.append({"feature": feature, "family": family, **meta})
    if len(rows) != 531:
        raise RuntimeError(f"registry length {len(rows)}")
    return tuple(rows)


REGISTRY = feature_registry()
REGISTRY_BY_NAME = {row["feature"]: row for row in REGISTRY}
