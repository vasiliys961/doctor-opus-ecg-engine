"""Patient-level split. Официальные strat_fold PTB-XL, без пересечения пациентов."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from benchmark.config import ROOT, load_config
from benchmark.label_mapping import TARGET_CODES, parse_scp_codes, scp_to_targets

FORENSIC_ECG_IDS = frozenset({"513", "00513"})


class SplitLeakageError(RuntimeError):
    """Один пациент или один ecg_id попал в два набора."""


def canonical_id(value: object) -> str:
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]
    return text


def assert_disjoint(frames: dict[str, list[dict[str, str]]]) -> None:
    patients = {name: {row["patient_id"] for row in rows} for name, rows in frames.items()}
    records = {name: {row["ecg_id"] for row in rows} for name, rows in frames.items()}
    names = list(frames)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            shared_patients = patients[left] & patients[right]
            shared_records = records[left] & records[right]
            if shared_patients or shared_records:
                raise SplitLeakageError(
                    f"{left} и {right} пересекаются: "
                    f"пациентов {len(shared_patients)}, записей {len(shared_records)}."
                )


def assign_fold(fold: int, config: dict) -> str | None:
    split = config["split"]
    if fold in split["train_folds"]:
        return "train"
    if fold in split["val_folds"]:
        return "val"
    if fold in split["test_folds"]:
        return "test"
    return None


def build_frames(database: Path, config: dict | None = None) -> tuple[dict[str, list[dict[str, str]]], dict]:
    used = config or load_config()
    frames: dict[str, list[dict[str, str]]] = {"train": [], "val": [], "test": []}
    positives = {name: [0] * len(TARGET_CODES) for name in frames}
    with database.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            ecg_id = canonical_id(row["ecg_id"])
            if ecg_id in FORENSIC_ECG_IDS:
                continue
            part = assign_fold(int(row["strat_fold"]), used)
            if part is None:
                continue
            frames[part].append(
                {
                    "ecg_id": ecg_id,
                    "patient_id": canonical_id(row["patient_id"]),
                    "split": part,
                }
            )
            targets = scp_to_targets(parse_scp_codes(row["scp_codes"]))
            for index, value in enumerate(targets):
                positives[part][index] += int(value)
    assert_disjoint(frames)
    summary = {
        "strategy": used["split"]["strategy"],
        "seed": used["seed"],
        "dataset_version": "PTB-XL 1.0.3",
        "forensic_records_excluded": ["00513"],
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "splits": {},
    }
    for name, rows in frames.items():
        summary["splits"][name] = {
            "ecg_count": len(rows),
            "patient_count": len({row["patient_id"] for row in rows}),
            "positive_labels": {code: positives[name][index] for index, code in enumerate(TARGET_CODES)},
        }
    return frames, summary


def write_frames(frames: dict[str, list[dict[str, str]]], summary: dict, directory: Path | None = None) -> None:
    out = directory or (ROOT / "benchmark" / "splits")
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in frames.items():
        target = out / f"{name}.csv"
        with target.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["ecg_id", "patient_id", "split"])
            writer.writeheader()
            writer.writerows(rows)
    (ROOT / "benchmark" / "split_manifest.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )


def load_frames(directory: Path | None = None) -> dict[str, list[dict[str, str]]]:
    folder = directory or (ROOT / "benchmark" / "splits")
    frames: dict[str, list[dict[str, str]]] = {}
    for name in ("train", "val", "test"):
        with (folder / f"{name}.csv").open(newline="", encoding="utf-8") as handle:
            frames[name] = list(csv.DictReader(handle))
    assert_disjoint(frames)
    return frames
