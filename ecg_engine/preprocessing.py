"""Детерминированная раскладка отведений. Фильтр автора сюда не подставляется."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

CANONICAL_LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
_ALIASES = {
    "I": "I",
    "II": "II",
    "III": "III",
    "AVR": "aVR",
    "AVL": "aVL",
    "AVF": "aVF",
    "V1": "V1",
    "V2": "V2",
    "V3": "V3",
    "V4": "V4",
    "V5": "V5",
    "V6": "V6",
}


class PreprocessingError(ValueError):
    """Вход нельзя разложить в 12 канонических отведений."""


@dataclass(frozen=True)
class PreprocessingRecord:
    sampling_rate_input: float
    sampling_rate_processing: float
    filter_config: str
    baseline_config: str
    normalization_config: str
    lead_mapping: dict[str, str]
    n_samples: int

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def canonical_lead_name(name: str) -> str:
    token = str(name).strip()
    found = _ALIASES.get(token.upper())
    if found is None:
        raise PreprocessingError(f"FAIL: неизвестное отведение {name!r}.")
    return found


def arrange_leads(signal: object, lead_names: list[str], sampling_rate: float) -> tuple[np.ndarray, PreprocessingRecord]:
    if not isinstance(sampling_rate, (int, float)) or not np.isfinite(sampling_rate) or sampling_rate <= 0:
        raise PreprocessingError("FAIL: sampling_rate должен быть конечным положительным числом.")
    array = np.asarray(signal, dtype=float)
    if array.ndim != 2:
        raise PreprocessingError("FAIL: signal должен быть двумерным.")
    names = [canonical_lead_name(name) for name in lead_names]
    if len(names) != len(set(names)):
        raise PreprocessingError("FAIL: отведения повторяются.")
    if array.shape[1] == len(names):
        oriented = array
    elif array.shape[0] == len(names) and array.shape[1] != len(names):
        oriented = array.T
    else:
        raise PreprocessingError(
            f"FAIL: форма signal {array.shape} не совпадает с числом отведений {len(names)}."
        )
    missing = [lead for lead in CANONICAL_LEADS if lead not in names]
    if missing:
        raise PreprocessingError(f"FAIL: нет отведений {missing}.")
    extra = [name for name in names if name not in CANONICAL_LEADS]
    if extra:
        raise PreprocessingError(f"FAIL: лишние отведения {extra}.")
    index = {name: position for position, name in enumerate(names)}
    ordered = np.column_stack([oriented[:, index[lead]] for lead in CANONICAL_LEADS])
    if ordered.shape[0] < 1 or not np.isfinite(ordered).all():
        raise PreprocessingError("FAIL: сигнал пустой или содержит NaN/Inf.")
    record = PreprocessingRecord(
        sampling_rate_input=float(sampling_rate),
        sampling_rate_processing=float(sampling_rate),
        filter_config="none",
        baseline_config="none",
        normalization_config="none",
        lead_mapping={lead: lead for lead in CANONICAL_LEADS},
        n_samples=int(ordered.shape[0]),
    )
    return ordered, record
