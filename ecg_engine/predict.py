"""CLI: python -m ecg_engine.predict path.csv"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ecg_engine.ensemble import ModelAssetsError, predict_csv
from ecg_engine.schema import ECGSchemaError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Инференс ансамбля по CSV с 531 признаком.")
    parser.add_argument("csv", help="CSV: ecg_id (необязательно) и 531 колонка в каноническом порядке")
    parser.add_argument("-o", "--output", help="Куда записать JSON. Без флага печатает в stdout.")
    parser.add_argument("--model-dir", help="Каталог с .pth, ecg_train_mean.npy и ecg_train_std.npy")
    args = parser.parse_args(argv)
    try:
        payload = predict_csv(args.csv, args.model_dir).as_json()
    except (ECGSchemaError, ModelAssetsError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
