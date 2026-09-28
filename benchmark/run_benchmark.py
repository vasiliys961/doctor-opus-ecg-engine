"""Единая точка benchmark. Не скачивает датасет и не подменяет published 531."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from benchmark.config import ROOT, load_config, resolve
from benchmark.coverage import COVERAGE_STATEMENT, coverage_accounting, coverage_rows, write_detailed_coverage
from benchmark.data import OFFICIAL_ECGDELI_SHA256
from benchmark.dataset_locator import (
    classify_raw_corpus,
    count_waveforms,
    dataset_status,
    format_check,
    resolve_dataset,
)
from benchmark.label_mapping import scp_to_targets, validate_binary_targets
from benchmark.raw_model.dataset import index_waveforms, synthetic_split
from benchmark.raw_model.train import fit_lazy_records, fit_raw_model
from benchmark.reference import regression_max_abs, sanity_known_record
from benchmark.split import canonical_id, load_frames

RESULTS = ROOT / "benchmark" / "results"
STEP = RESULTS / "step15_1"


def _location(config: dict, args: argparse.Namespace):
    return resolve_dataset(
        config,
        cli_root=args.ptbxl_root,
        cli_features=args.ecgdeli_features,
        cli_database=args.ptbxl_database,
    )


def _status_line(status: str) -> int:
    print(f"BENCHMARK STATUS: {status}")
    return 0 if status != "CODE_FAILURE" else 1


def prepare(config: dict, database: Path) -> int:
    if not database.is_file():
        print(f"Нет {database}.")
        print("Положите PTB-XL 1.0.3 ptbxl_database.csv по этому пути.")
        print("Команда с подсказкой: python benchmark/fetch_dataset.py")
        print("BENCHMARK STATUS: BLOCKED_BY_MISSING_DATASET")
        return 0
    from benchmark.split import build_frames, write_frames

    frames, summary = build_frames(database, config)
    write_frames(frames, summary)
    print(
        "split "
        + " ".join(f"{name}={summary['splits'][name]['ecg_count']}" for name in ("train", "val", "test"))
    )
    return 0


def dry_run(config: dict, location) -> int:
    report = location.feature_report
    checks = {
        "ptbxl_database": location.metadata.state == "FOUND",
        "records500": location.records.state == "FOUND",
        "ecgdeli_features": bool(report.get("accepted")),
        "ensemble_weights": (resolve(config["paths"]["ensemble_dir"]) / "ecg_model.pth").is_file(),
        "splits": (ROOT / "benchmark" / "splits" / "test.csv").is_file(),
        "label_count": len(config["split"]["train_folds"]) > 0,
        "sampling_rate": config["sampling_rate"] == 500,
        "channels": config["channels"] == 12,
        "augmentation": config["augmentation"] is False,
    }
    for name, ok in checks.items():
        print(f"{name}: {'OK' if ok else 'MISSING'}")
    print("lead_order: I II III aVR aVL aVF V1 V2 V3 V4 V5 V6")
    print("обучение не запускалось")
    if all(checks.values()):
        return _status_line("READY")
    return _status_line("BLOCKED_BY_MISSING_DATASET")


def check_data(location, frames) -> int:
    raw = classify_raw_corpus(location.records.path, frames)
    print(format_check(location))
    print(f"usable raw ECG in split: {raw['usable']}")
    status = dataset_status(location, raw_usable=int(raw["usable"]))
    _write_json(STEP / "dataset_status.json", {"status": status, "raw": _public_raw(raw), "features": location.feature_report})
    return _status_line(status)


def _public_raw(raw: dict) -> dict:
    return {key: value for key, value in raw.items() if key not in {"index", "usable_rows"}}


def evaluate_reference(config: dict, location) -> int:
    fixture = resolve(config["paths"]["reference_fixture"])
    model_dir = resolve(config["paths"]["ensemble_dir"])
    try:
        delta = regression_max_abs(fixture, model_dir)
    except Exception as exc:
        _write_failure("reference", "MODEL_IMPORT_ERROR", exc)
        return _status_line("CODE_FAILURE")
    print(f"reference regression max_abs_difference={delta}")
    if delta >= 1e-5:
        print("STOP: regression существующего inference не сошёлся.")
        return _status_line("CODE_FAILURE")
    try:
        sanity = sanity_known_record(fixture, model_dir)
    except Exception as exc:
        _write_failure("reference", "PREDICTION_ERROR", exc)
        return _status_line("CODE_FAILURE")
    print(f"sanity ecg_id={sanity['ecg_id']} shape={sanity['shape']} prediction={sanity['prediction_shape']}")
    report = location.feature_report
    if not report.get("accepted"):
        print(f"reference reason: {report.get('reason')}")
        print(report.get("detail"))
        print("Оценка reference 531 на test split не запущена.")
        _write_json(STEP / "reference_debug.json", {"regression_max_abs": delta, "sanity": sanity, "table": report})
        return _status_line("BLOCKED_BY_MISSING_DATASET")
    frames = load_frames()
    from benchmark.reference import predict_published_table

    destination = STEP / "reference_531_predictions.csv"
    try:
        predict_published_table(
            location.features.path,
            [row["ecg_id"] for row in frames["test"]],
            model_dir,
            destination,
        )
    except Exception as exc:
        _write_failure("reference", "PREDICTION_ERROR", exc)
        return _status_line("CODE_FAILURE")
    print(f"reference predictions {destination}")
    return _status_line("COMPLETED")


def debug_reference(config: dict, location) -> int:
    code = evaluate_reference(config, location)
    report = location.feature_report
    print(f"REFERENCE_REASON: {report.get('reason')}")
    print(f"REFERENCE_DETAIL: {report.get('detail')}")
    return code


def smoke(config: dict, device: str | None) -> int:
    smoke_config = dict(config)
    smoke_config["epochs"] = int(config["smoke"]["epochs"])
    smoke_config["batch_size"] = int(config["smoke"]["batch_size"])
    smoke_config["device"] = device
    signals, labels, parts = synthetic_split(int(config["smoke"]["records"]), int(config["seed"]))
    metadata = fit_raw_model(signals, labels, parts, smoke_config, RESULTS / "smoke")
    summary = {
        "kind": "synthetic_smoke",
        "ptbxl_waveforms_used": False,
        "forward_backward_evaluation": True,
        "device": metadata["device"],
        "best_epoch": metadata["best_epoch"],
        "best_val_macro_auroc": metadata["best_val_macro_auroc"],
    }
    (RESULTS / "smoke" / "smoke_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("smoke PASS synthetic subset; это не метрика PTB-XL")
    return 0


def _write_raw_error(text: str) -> None:
    STEP.mkdir(parents=True, exist_ok=True)
    (STEP / "raw_training_error.txt").write_text(text, encoding="utf-8")


def _write_failure(kind: str, reason: str, exc: BaseException) -> None:
    text = f"classification: {reason}\nkind: {kind}\n\n{traceback.format_exc()}"
    if kind == "raw":
        _write_raw_error(text)
    else:
        STEP.mkdir(parents=True, exist_ok=True)
        (STEP / f"{kind}_error.txt").write_text(text, encoding="utf-8")
    print(text)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def _labels(database: Path, rows: list[dict[str, str]], split_name: str) -> np.ndarray:
    frame = pd.read_csv(database, usecols=["ecg_id", "scp_codes"])
    lookup = {canonical_id(row.ecg_id): row.scp_codes for row in frame.itertuples(index=False)}
    matrix = []
    for row in rows:
        if row["ecg_id"] not in lookup:
            raise KeyError(row["ecg_id"])
        matrix.append(scp_to_targets(lookup[row["ecg_id"]]))
    labels = np.stack(matrix).astype(np.float64)
    validate_binary_targets(labels, split_name)
    return labels


def _cap(rows: list[dict[str, str]], limit: int) -> list[dict[str, str]]:
    return rows[:limit]


def train_raw(config: dict, location, *, device: str | None, small: bool) -> int:
    frames = load_frames()
    raw = classify_raw_corpus(location.records.path, frames)
    print(f"records500 headers: {raw['headers']}")
    print(f"usable raw ECG in split: {raw['usable']}")
    print(raw["detail"])
    if raw["reason"] != "OK":
        _write_raw_error(
            "classification: NO_RAW_DATA\n"
            "traceback: none\n"
            "training was not started\n\n"
            f"{raw['detail']}\n"
        )
        _write_json(STEP / "raw_debug.json", _public_raw(raw))
        print("raw training = BLOCKED")
        print("raw evaluation = BLOCKED")
        return _status_line("BLOCKED_BY_MISSING_DATASET")
    try:
        index = index_waveforms(location.records.path)
        train_rows = list(raw["usable_rows"]["train"])
        val_rows = list(raw["usable_rows"]["val"])
        if small:
            train_rows = _cap(train_rows, 128)
            val_rows = _cap(val_rows, 64)
            config = dict(config)
            config["epochs"] = int(config["smoke"]["epochs"])
            config["batch_size"] = int(config["smoke"]["batch_size"])
        if len(train_rows) + len(val_rows) > 256 and small:
            raise RuntimeError("smoke cap exceeded 256 records")
        config = dict(config)
        config["device"] = device
        train_labels = _labels(location.metadata.path, train_rows, "train")
        val_labels = _labels(location.metadata.path, val_rows, "val")
        output = STEP / "raw_smoke" if small else STEP / "raw_model"
        metadata = fit_lazy_records(
            [index[row["ecg_id"]] for row in train_rows],
            train_labels,
            [index[row["ecg_id"]] for row in val_rows],
            val_labels,
            config,
            output,
        )
    except Exception as exc:
        reason = "LABEL_ERROR" if isinstance(exc, (KeyError, ValueError)) else "UNKNOWN"
        _write_failure("raw", reason, exc)
        return _status_line("CODE_FAILURE")
    print(f"raw training device={metadata['device']} best_epoch={metadata['best_epoch']}")
    return _status_line("COMPLETED")


def debug_raw(config: dict, location, device: str | None) -> int:
    code = train_raw(config, location, device=device, small=True)
    print("RAW_REASON: см. benchmark/results/step15_1/raw_training_error.txt или raw_debug.json")
    return code


def write_report(config: dict, location) -> int:
    frames = load_frames()
    raw = classify_raw_corpus(location.records.path, frames)
    accepted = bool(location.feature_report.get("accepted"))
    accounting = coverage_accounting()
    for path in (STEP / "feature_coverage_detailed.csv", RESULTS / "feature_coverage_detailed.csv"):
        write_detailed_coverage(path, published_reference_available=accepted)
    status = dataset_status(location, raw_usable=int(raw["usable"]))
    summary = {
        "step": "15.1",
        "benchmark_status": status,
        "dataset": {
            "name": "PTB-XL",
            "version": "1.0.3",
            "feature_dataset": "PTB-XL+ 1.0.1",
            "metadata": location.metadata.state,
            "metadata_path": str(location.metadata.path),
            "records500": location.records.state,
            "records500_path": str(location.records.path),
            "raw_headers": count_waveforms(location.records.path),
            "raw_usable_in_split": raw["usable"],
            "ecgdeli_features": location.features.state,
            "ecgdeli_accepted": accepted,
            "official_feature_sha256": OFFICIAL_ECGDELI_SHA256,
            "feature_reason": location.feature_report.get("reason"),
            "feature_detail": location.feature_report.get("detail"),
        },
        "split": {
            name: {
                "ecg_count": len(rows),
                "patient_count": len({row["patient_id"] for row in rows}),
            }
            for name, rows in frames.items()
        },
        "models": {
            "reference_531": {
                "retrained": False,
                "test_evaluation": "BLOCKED" if not accepted else "READY",
                "reason": location.feature_report.get("reason"),
            },
            "raw_cnn": {
                "architecture": "RawECGCNN",
                "full_training": "BLOCKED" if raw["reason"] != "OK" else "READY",
                "reason": raw["reason"],
            },
        },
        "metrics": {"reference_531": None, "raw_cnn": None},
        "coverage": {
            "statement": COVERAGE_STATEMENT,
            "runtime": accounting["runtime"],
            "method_available": accounting["method_available"],
            "not_executed": accounting["not_executed"],
            "unknown": accounting["unknown"],
            "families": coverage_rows(),
        },
        "reproducibility": {
            "seed": config["seed"],
            "augmentation": False,
            "raw_vector_to_existing_ensemble": False,
            "raw_to_531_status": "NOT_PROVEN",
        },
    }
    _write_json(STEP / "benchmark_summary.json", summary)
    print(COVERAGE_STATEMENT)
    print(f"отчёт {STEP / 'benchmark_summary.json'}")
    return _status_line(status)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PTB-XL benchmark")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--train-raw", action="store_true")
    parser.add_argument("--evaluate-reference", action="store_true")
    parser.add_argument("--evaluate-raw", action="store_true")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--check-data", action="store_true")
    parser.add_argument("--debug-reference", action="store_true")
    parser.add_argument("--debug-raw", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default=None)
    parser.add_argument("--ptbxl-database", default=None)
    parser.add_argument("--ptbxl-root", default=None)
    parser.add_argument("--ecgdeli-features", default=None)
    args = parser.parse_args(argv)
    config = load_config()
    if args.device:
        config = dict(config)
        config["device"] = args.device
    location = _location(config, args)
    frames = load_frames()
    selected = any(
        [
            args.prepare,
            args.train_raw,
            args.evaluate_reference,
            args.evaluate_raw,
            args.report,
            args.all,
            args.dry_run,
            args.smoke_test,
            args.check_data,
            args.debug_reference,
            args.debug_raw,
        ]
    )
    if not selected:
        parser.print_help()
        return 2
    code = 0
    if args.check_data or args.all:
        code = max(code, check_data(location, frames))
    if args.dry_run or args.all:
        code = max(code, dry_run(config, location))
    if args.prepare or args.all:
        code = max(code, prepare(config, location.metadata.path))
    if args.debug_reference:
        code = max(code, debug_reference(config, location))
    elif args.evaluate_reference or args.all:
        code = max(code, evaluate_reference(config, location))
    if args.smoke_test or args.all:
        code = max(code, smoke(config, args.device))
    if args.debug_raw:
        code = max(code, debug_raw(config, location, args.device))
    elif args.train_raw or args.all:
        code = max(code, train_raw(config, location, device=args.device, small=args.smoke or args.all))
    if args.evaluate_raw or args.all:
        raw = classify_raw_corpus(location.records.path, frames)
        if raw["reason"] != "OK" or not (STEP / "raw_model" / "best_model.pth").is_file():
            print("Оценка raw-модели на test PTB-XL не запущена.")
            print("raw evaluation = BLOCKED")
            code = max(code, _status_line("BLOCKED_BY_MISSING_DATASET"))
        else:
            print("Checkpoint есть, отдельная test-оценка подключается после полного обучения.")
            code = max(code, _status_line("READY"))
    if args.report or args.all:
        code = max(code, write_report(config, location))
    return code


if __name__ == "__main__":
    sys.exit(main())
