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


def write_coverage(path: Path) -> None:
    rows = coverage_rows()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
