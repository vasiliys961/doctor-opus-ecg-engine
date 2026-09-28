"""Единственный порядок 531 колонки. Его задаёт схема, не экстрактор."""

from __future__ import annotations

from ecg_engine.feature_columns import FEATURE_COLUMNS

FEATURE_COLUMNS_531: tuple[str, ...] = FEATURE_COLUMNS
FEATURE_INDEX_531: dict[str, int] = {
    name: index for index, name in enumerate(FEATURE_COLUMNS_531)
}

assert len(FEATURE_COLUMNS_531) == 531
assert len(set(FEATURE_COLUMNS_531)) == 531
assert len(FEATURE_INDEX_531) == 531
