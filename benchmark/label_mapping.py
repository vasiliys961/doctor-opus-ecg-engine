"""SCP-коды PTB-XL → 24 бинарные цели существующего ансамбля.

Положительный класс — наличие кода среди ключей scp_codes.
Порог likelihood не применяется: у PTB-XL многие ритмы, включая SR, записаны с likelihood 0.0,
и именно наличие ключа соответствует этим 24 выходам.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np

from ecg_engine.scp import TOP_24_CODES

TARGET_CODES: tuple[str, ...] = TOP_24_CODES
MAPPING_PATH = Path(__file__).resolve().parent / "label_mapping.json"
LABEL_RULE = "positive_if_scp_key_present_including_likelihood_0"


def mapping_table() -> dict[str, list[str]]:
    """Каждая из 24 целей совпадает с одним SCP-кодом того же имени."""
    return {code: [code] for code in TARGET_CODES}


def save_mapping(path: Path = MAPPING_PATH) -> None:
    path.write_text(json.dumps(mapping_table(), indent=2) + "\n", encoding="utf-8")


def load_mapping(path: Path = MAPPING_PATH) -> dict[str, list[str]]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if list(loaded) != list(TARGET_CODES):
        raise ValueError("Порядок label_mapping.json разошёлся с 24 выходами ансамбля.")
    return loaded


def parse_scp_codes(raw: object) -> dict[str, float]:
    if isinstance(raw, dict):
        parsed = raw
    else:
        parsed = ast.literal_eval(str(raw))
    if not isinstance(parsed, dict):
        raise ValueError("scp_codes должен быть словарём.")
    return {str(key): float(value) for key, value in parsed.items()}


def scp_to_targets(scp_codes: object, table: dict[str, list[str]] | None = None) -> np.ndarray:
    codes = parse_scp_codes(scp_codes)
    used = table or mapping_table()
    targets = np.zeros(len(TARGET_CODES), dtype=np.float64)
    for index, name in enumerate(TARGET_CODES):
        if any(source in codes for source in used[name]):
            targets[index] = 1.0
    return targets
