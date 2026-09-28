"""Покрытие compatibility layer. Неполный вектор в ансамбль не отправляется."""

from __future__ import annotations

import csv
from pathlib import Path

from ecg_engine.feature_registry import REGISTRY

FAMILY_GROUPS = (
    ("interval", {"PQ_Int", "PR_Int", "QRS_Dur", "QT_Int", "P_DurFull", "T_DurFull", "P_Dur", "T_Dur"}),
    ("RR", {"RR_Mean"}),
    ("Framingham", {"QT_IntFramingham"}),
    ("amplitude", {"P_Amp", "Q_Amp", "R_Amp", "S_Amp", "T_Amp"}),
    ("P_Morph", {"P_Morph"}),
    ("QT_IntCorr", {"QT_IntCorr"}),
    ("ST_Elev", {"ST_Elev"}),
    ("HA__Global", {"HA"}),
)


class IncompleteFeatureVectorError(RuntimeError):
    """Существующий ансамбль этот вектор не принимает."""


def refuse_incomplete_for_ensemble(mask: list[int]) -> None:
    if not mask or any(value != 1 for value in mask):
        raise IncompleteFeatureVectorError(
            "Неполный 531-вектор не передаётся в существующий ensemble. Inference не inventится."
        )


def coverage_rows() -> list[dict[str, object]]:
    rows = []
    for family, names in FAMILY_GROUPS:
        selected = [item for item in REGISTRY if item["family"] in names]
        status = sorted({str(item["status"]) for item in selected})
        rows.append(
            {
                "family": family,
                "columns": len(selected),
                "status": "|".join(status),
                "exact_proven": "false",
                "runtime_calculated": 0,
                "coverage_percent_of_531": round(100.0 * len(selected) / 531.0, 4),
                "known_method": sum(item["status"] == "RECONSTRUCTED" for item in selected),
                "unknown_or_not_executed": sum(item["status"] != "RECONSTRUCTED" for item in selected),
            }
        )
    return rows


COVERAGE_STATEMENT = (
    "456 features have a documented reconstruction method/path, "
    "but runtime reconstruction of the full 531-vector is not established."
)


def coverage_accounting() -> dict[str, int]:
    """456 — это documented method, не успешно посчитанный runtime-вектор."""
    return {
        "runtime": 0,
        "method_available": sum(item["status"] == "RECONSTRUCTED" for item in REGISTRY),
        "not_executed": sum(item["status"] == "NOT_EXECUTED" for item in REGISTRY),
        "unknown": sum(item["status"] == "UNKNOWN" for item in REGISTRY),
    }


def detailed_coverage_rows(*, published_reference_available: bool) -> list[dict[str, str]]:
    published = "true" if published_reference_available else "false"
    rows = []
    for item in REGISTRY:
        status = str(item["status"])
        if status == "RECONSTRUCTED":
            method_status = "documented"
            execution_status = "not_established"
        elif status == "NOT_EXECUTED":
            method_status = "identified_not_executed"
            execution_status = "not_executed"
        else:
            method_status = "absent"
            execution_status = "unknown"
        rows.append(
            {
                "feature": str(item["feature"]),
                "family": str(item["family"]),
                "runtime_status": "not_available",
                "method_status": method_status,
                "execution_status": execution_status,
                "published_reference_available": published,
            }
        )
    return rows


def write_detailed_coverage(path: Path, *, published_reference_available: bool) -> None:
    rows = detailed_coverage_rows(published_reference_available=published_reference_available)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_coverage(path: Path) -> None:
    rows = coverage_rows()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
