"""Подсказка, куда положить PTB-XL. Датасет сам не скачивается."""

from __future__ import annotations

import sys

INSTRUCTIONS = """
PTB-XL 1.0.3 и PTB-XL+ 1.0.1 в репозиторий не входят и этим скриптом не скачиваются.

Ожидаемые пути:
  benchmark/data/ptbxl/ptbxl_database.csv
  benchmark/data/ptbxl/records500/
  benchmark/data/ptbxl_plus/ecgdeli_features.csv

Источники:
  https://physionet.org/content/ptb-xl/1.0.3/
  https://physionet.org/content/ptb-xl-plus/1.0.1/

ecgdeli_features.csv принимается только с sha256
  84143735682f727201cc7f2825c047f98898f9756c6a21729af993b516221556

После размещения:
  python benchmark/run_benchmark.py --dry-run
  python benchmark/run_benchmark.py --prepare --ptbxl-database benchmark/data/ptbxl/ptbxl_database.csv
""".strip()


def main() -> int:
    print(INSTRUCTIONS)
    return 0


if __name__ == "__main__":
    sys.exit(main())
