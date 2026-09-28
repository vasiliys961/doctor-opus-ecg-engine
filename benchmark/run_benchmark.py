"""Единая точка benchmark. Полное обучение не стартует, если нет records500."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from benchmark.config import ROOT, load_config, resolve
from benchmark.coverage import coverage_rows, write_coverage
from benchmark.data import OFFICIAL_ECGDELI_SHA256
from benchmark.raw_model.dataset import synthetic_split
from benchmark.raw_model.train import fit_raw_model
from benchmark.reference import FeatureTableError, regression_max_abs, require_official_feature_table
from benchmark.split import build_frames, load_frames, write_frames

RESULTS = ROOT / "benchmark" / "results"


def _database_path(config: dict, override: str | None) -> Path:
    return Path(override) if override else resolve(config["paths"]["ptbxl_database"])


def prepare(config: dict, database: Path) -> int:
    if not database.is_file():
        print(f"Нет {database}.")
        print("Положите PTB-XL 1.0.3 ptbxl_database.csv по этому пути.")
        print("Команда с подсказкой: python benchmark/fetch_dataset.py")
        return 2
    frames, summary = build_frames(database, config)
    write_frames(frames, summary)
    print(
        "split "
        + " ".join(f"{name}={summary['splits'][name]['ecg_count']}" for name in ("train", "val", "test"))
    )
    return 0


def dry_run(config: dict, database: Path) -> int:
    checks = {
        "ptbxl_database": database.is_file(),
        "records500": resolve(config["paths"]["records500"]).is_dir(),
        "ecgdeli_features": resolve(config["paths"]["ecgdeli_features"]).is_file(),
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
    return 0 if all(checks.values()) else 2


def evaluate_reference(config: dict) -> int:
    fixture = resolve(config["paths"]["reference_fixture"])
    model_dir = resolve(config["paths"]["ensemble_dir"])
    delta = regression_max_abs(fixture, model_dir)
    print(f"reference regression max_abs_difference={delta}")
    if delta >= 1e-5:
        print("STOP STEP 15: regression существующего inference не сошёлся.")
        return 2
    table = resolve(config["paths"]["ecgdeli_features"])
    try:
        require_official_feature_table(table)
    except FeatureTableError as exc:
        print(str(exc))
        print("Оценка reference 531 на test split не запущена: опубликованной таблицы нет.")
        return 2
    frames = load_frames()
    from benchmark.reference import predict_published_table

    predict_published_table(
        table,
        [row["ecg_id"] for row in frames["test"]],
        model_dir,
        RESULTS / "reference_531_predictions.csv",
    )
    print("reference predictions записаны")
    return 0


def smoke(config: dict) -> int:
    smoke_config = dict(config)
    smoke_config["epochs"] = int(config["smoke"]["epochs"])
    smoke_config["batch_size"] = int(config["smoke"]["batch_size"])
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


def train_full(config: dict) -> int:
    records = resolve(config["paths"]["records500"])
    headers = list(records.glob("*/*_hr.hea")) if records.is_dir() else []
    print(f"records500 headers: {len(headers)}")
    print("Полное обучение не запущено: в окружении нет корпуса PTB-XL records500.")
    print("Smoke на синтетике этот корпус не заменяет. Скачивание молча не выполняется.")
    return 2


def write_report(config: dict) -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    write_coverage(RESULTS / "feature_coverage.csv")
    frames = load_frames()
    summary = {
        "dataset": {
            "name": "PTB-XL",
            "version": "1.0.3",
            "feature_dataset": "PTB-XL+ 1.0.1",
            "sampling_rate": config["sampling_rate"],
            "waveforms_present": resolve(config["paths"]["records500"]).is_dir(),
            "official_feature_sha256": OFFICIAL_ECGDELI_SHA256,
        },
        "split": {
            name: {
                "ecg_count": len(rows),
                "patient_count": len({row["patient_id"] for row in rows}),
            }
            for name, rows in frames.items()
        },
        "models": {
            "reference_531": {"retrained": False, "test_evaluation": "NOT_RUN", "reason": "official ecgdeli_features.csv absent or checksum rejected"},
            "raw_cnn": {"architecture": "RawECGCNN", "full_training": "NOT_RUN", "reason": "PTB-XL records500 corpus absent"},
        },
        "metrics": {"reference_531": None, "raw_cnn": None},
        "coverage": coverage_rows(),
        "reproducibility": {
            "seed": config["seed"],
            "augmentation": False,
            "raw_vector_to_existing_ensemble": False,
            "raw_to_531_status": "NOT_PROVEN",
        },
    }
    (RESULTS / "benchmark_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"отчёт {RESULTS / 'benchmark_summary.json'}")
    return 0


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
    parser.add_argument("--ptbxl-database", default=None)
    args = parser.parse_args(argv)
    config = load_config()
    database = _database_path(config, args.ptbxl_database)
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
        ]
    )
    if not selected:
        parser.print_help()
        return 2
    code = 0
    if args.dry_run or args.all:
        code = max(code, dry_run(config, database))
    if args.prepare or args.all:
        code = max(code, prepare(config, database))
    if args.evaluate_reference or args.all:
        code = max(code, evaluate_reference(config))
    if args.smoke_test or args.all:
        code = max(code, smoke(config))
    if args.train_raw or args.all:
        code = max(code, train_full(config))
    if args.evaluate_raw or args.all:
        print("Оценка raw-модели на test PTB-XL не запущена: полного обучения нет.")
        code = max(code, 2)
    if args.report or args.all:
        code = max(code, write_report(config))
    return code


if __name__ == "__main__":
    sys.exit(main())
