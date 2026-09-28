"""Сводка STEP 13. Числа 00513 берутся из forensic-таблиц и не дорисовываются."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from ecg_engine.canonical_schema import FEATURE_INDEX_531
from ecg_engine.feature_registry import REGISTRY

ROOT = Path(__file__).resolve().parents[2]
FORENSIC = ROOT / "experiments" / "raw_to_531" / "step12_feature_status.csv"
STATUS_OUT = ROOT / "experiments" / "raw_to_531" / "step13_feature_status.csv"
SUMMARY_OUT = ROOT / "experiments" / "raw_to_531" / "step13_summary.json"

_WITH_VALUE = {"EXACT", "CALCULABLE_BUT_DIFFERENT", "REQUIRES_PROCESSED_SIGNAL"}


def _blank(value: str) -> str:
    return "" if value is None else value


def write() -> None:
    forensic = {}
    with FORENSIC.open(newline="") as handle:
        for row in csv.DictReader(handle):
            forensic[row["feature"]] = row
    fields = [
        "feature",
        "index",
        "family",
        "status",
        "method",
        "source",
        "exact_proven",
        "confidence",
        "value_00513",
        "reference_value_00513",
        "absolute_error_00513",
        "notes",
    ]
    with STATUS_OUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in REGISTRY:
            prior = forensic[str(row["feature"])]
            produced = prior["reconstruction_status"] in _WITH_VALUE and prior["reconstructed_value"] != ""
            writer.writerow(
                {
                    "feature": row["feature"],
                    "index": FEATURE_INDEX_531[str(row["feature"])],
                    "family": row["family"],
                    "status": row["status"],
                    "method": row["method"],
                    "source": row["source"],
                    "exact_proven": "false",
                    "confidence": row["confidence"],
                    "value_00513": prior["reconstructed_value"] if produced else "",
                    "reference_value_00513": prior["published_value"],
                    "absolute_error_00513": prior["abs_error"] if produced else "",
                    "notes": row["notes"],
                }
            )
    summary = {
        "raw_to_531_status": "NOT_PROVEN",
        "reference_ensemble_status": "VALID",
        "total_features": 531,
        "families": {
            "interval_rr_framingham": 240,
            "amplitude": 180,
            "p_morph": 36,
            "qt_intcorr": 36,
            "st_elev": 36,
            "ha_global": 3,
        },
        "raw_inference": {
            "supported": False,
            "reason": "Full exact 531-feature reconstruction has not been established.",
        },
    }
    SUMMARY_OUT.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    write()
