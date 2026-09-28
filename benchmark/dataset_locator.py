"""Поиск локального PTB-XL. Датасет отсюда не скачивается."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from benchmark.config import ROOT, resolve
from benchmark.data import OFFICIAL_ECGDELI_SHA256
from benchmark.raw_model.dataset import index_waveforms
from benchmark.reference import classify_feature_table

DISCOVERY_METADATA = (Path("/tmp/ptbxl_database.csv"),)
DISCOVERY_RECORDS = (ROOT / "research" / "ecgdeli_reproduction" / "data" / "ptbxl" / "records500",)
DISCOVERY_FEATURES = (Path("/tmp/ptbxl_plus/ecgdeli_features.csv"),)


@dataclass(frozen=True)
class LocatedPath:
    path: Path
    state: str
    source: str


@dataclass(frozen=True)
class DatasetLocation:
    root: LocatedPath | None
    metadata: LocatedPath
    records: LocatedPath
    features: LocatedPath
    feature_report: dict[str, object]


def _first_existing(candidates: tuple[Path, ...]) -> Path | None:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _pick(cli: str | None, env: str | None, configured: object, fallback: Path) -> tuple[Path, str]:
    if cli:
        return Path(cli), "cli"
    if env:
        return Path(env), "environment"
    if isinstance(configured, str) and configured.strip():
        return Path(configured), "config"
    return fallback, "config"


def resolve_dataset(
    config: dict,
    *,
    cli_root: str | None = None,
    cli_features: str | None = None,
    cli_database: str | None = None,
    search: bool = True,
) -> DatasetLocation:
    """Приоритет пути: CLI, затем переменная окружения, затем config.yaml."""
    data = config.get("data") or {}
    paths = config["paths"]
    env_root = os.environ.get("PTBXL_ROOT") or None
    env_features = os.environ.get("ECGDELI_FEATURES") or None
    explicit_root = bool(cli_root or env_root or data.get("ptbxl_root"))
    explicit_features = bool(cli_features or env_features or data.get("ecgdeli_features"))
    root_path, root_source = _pick(cli_root, env_root, data.get("ptbxl_root"), Path())
    root = LocatedPath(root_path, "FOUND" if root_path.is_dir() else "MISSING", root_source) if explicit_root else None

    if cli_database:
        metadata_path, metadata_source = Path(cli_database), "cli"
    elif root is not None:
        metadata_path, metadata_source = root.path / "ptbxl_database.csv", root.source
    else:
        metadata_path, metadata_source = resolve(paths["ptbxl_database"]), "config"
    if search and not explicit_root and not cli_database and not metadata_path.is_file():
        discovered = _first_existing(DISCOVERY_METADATA)
        if discovered is not None:
            metadata_path, metadata_source = discovered, "search"

    if root is not None:
        records_path, records_source = root.path / "records500", root.source
    else:
        records_path, records_source = resolve(paths["records500"]), "config"
    if search and not explicit_root and not records_path.is_dir():
        discovered = _first_existing(DISCOVERY_RECORDS)
        if discovered is not None:
            records_path, records_source = discovered, "search"

    feature_fallback = resolve(paths["ecgdeli_features"])
    if explicit_features:
        features_path, features_source = _pick(cli_features, env_features, data.get("ecgdeli_features"), feature_fallback)
    elif root is not None:
        nested = (
            root.path / "features" / "ecgdeli_features.csv",
            root.path / "ecgdeli_features.csv",
        )
        found = _first_existing(nested)
        features_path, features_source = (found, root.source) if found is not None else (nested[0], root.source)
    else:
        features_path, features_source = feature_fallback, "config"
    if search and not explicit_features and root is None and not features_path.is_file():
        discovered = _first_existing(DISCOVERY_FEATURES)
        if discovered is not None:
            features_path, features_source = discovered, "search"

    return DatasetLocation(
        root=root,
        metadata=LocatedPath(metadata_path, "FOUND" if metadata_path.is_file() else "MISSING", metadata_source),
        records=LocatedPath(records_path, "FOUND" if records_path.is_dir() else "MISSING", records_source),
        features=LocatedPath(features_path, "FOUND" if features_path.is_file() else "MISSING", features_source),
        feature_report=classify_feature_table(features_path),
    )


def count_lines(path: Path) -> int | None:
    if not path.is_file():
        return None
    with path.open("rb") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def count_waveforms(path: Path) -> int:
    if not path.is_dir():
        return 0
    return sum(1 for _ in path.rglob("*_hr.hea"))


def classify_raw_corpus(records_root: Path, frames: dict[str, list[dict[str, str]]]) -> dict[str, object]:
    if not records_root.is_dir():
        return {
            "reason": "NO_RAW_DATA",
            "headers": 0,
            "usable": 0,
            "detail": f"Нет каталога {records_root}. Raw training не симулируется.",
            "usable_rows": {"train": [], "val": [], "test": []},
        }
    index = index_waveforms(records_root)
    usable_rows = {
        name: [row for row in rows if row["ecg_id"] in index]
        for name, rows in frames.items()
    }
    usable = sum(len(rows) for rows in usable_rows.values())
    if usable == 0:
        detail = (
            f"В {records_root} найдено заголовков {len(index)}. "
            "Ни один не входит в train/val/test. "
            "Запись 00513, если она единственная, исключена из benchmark split и не заменяет корпус."
        )
        return {
            "reason": "NO_RAW_DATA",
            "headers": len(index),
            "usable": 0,
            "detail": detail,
            "usable_rows": usable_rows,
            "index": index,
        }
    return {
        "reason": "OK",
        "headers": len(index),
        "usable": usable,
        "detail": "records500 joined to the patient split",
        "usable_rows": usable_rows,
        "index": index,
    }


def dataset_status(location: DatasetLocation, *, raw_usable: int) -> str:
    metadata_ok = location.metadata.state == "FOUND"
    records_ok = location.records.state == "FOUND" and raw_usable > 0
    features_ok = bool(location.feature_report.get("accepted"))
    if metadata_ok and records_ok and features_ok:
        return "READY"
    if not metadata_ok and not records_ok and not features_ok and location.features.state == "MISSING" and location.records.state == "MISSING":
        return "BLOCKED_BY_MISSING_DATASET"
    if location.feature_report.get("accepted") is False or raw_usable == 0 or location.records.state == "MISSING":
        return "BLOCKED_BY_MISSING_DATASET"
    return "BLOCKED_BY_MISSING_DATASET"


def format_check(location: DatasetLocation) -> str:
    metadata_rows = count_lines(location.metadata.path)
    raw_records = count_waveforms(location.records.path)
    report = location.feature_report
    feature_rows = report.get("rows")
    feature_columns = report.get("feature_columns")
    root_state = "MISSING" if location.root is None or location.root.state == "MISSING" else "FOUND"
    lines = [
        "DATASET_STATUS:",
        f"metadata = {location.metadata.state}",
        f"records500 = {location.records.state}",
        f"ecgdeli_features = {location.features.state}",
        f"PTBXL_ROOT:",
        root_state,
        "ptbxl_database.csv:",
        location.metadata.state,
        f"path: {location.metadata.path}",
        f"source: {location.metadata.source}",
        "records500:",
        location.records.state,
        f"path: {location.records.path}",
        f"source: {location.records.source}",
        "ecgdeli_features.csv:",
        location.features.state,
        f"path: {location.features.path}",
        f"source: {location.features.source}",
        f"accepted_for_benchmark: {'YES' if report.get('accepted') else 'NO'}",
        f"reason: {report.get('reason')}",
        f"detail: {report.get('detail')}",
        f"official_sha256: {OFFICIAL_ECGDELI_SHA256}",
        "metadata rows:",
        str(metadata_rows if metadata_rows is not None else "MISSING"),
        "raw ECG records:",
        str(raw_records),
        "531 feature rows:",
        str(feature_rows if feature_rows is not None else "MISSING"),
        "531 feature columns:",
        str(feature_columns if feature_columns is not None else "MISSING"),
    ]
    return "\n".join(lines)
