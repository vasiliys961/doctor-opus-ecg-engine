"""Заготовка входа raw ECG. Признаков не считает."""

from __future__ import annotations

RAW_TO_531_STATUS = "NOT_PROVEN"


class RawECGExtractor:
    """Интерфейс будущего экстрактора. Сейчас возвращать 531 число нельзя."""

    status_code = RAW_TO_531_STATUS

    def extract(self, raw_ecg: object) -> None:
        raise NotImplementedError(
            "RAW_TO_531_STATUS = NOT_PROVEN. "
            "Нет подтверждённой пары raw ECG и CSV 531, поэтому вектор признаков не создаётся."
        )

    def status(self) -> dict:
        return {
            "status": RAW_TO_531_STATUS,
            "implementation": "NotImplemented",
            "features": None,
            "prediction": None,
            "message": (
                "Экстрактор отсутствует. Подтверждённой пары raw 12-lead ECG и соответствующего "
                "CSV из 531 колонок нет. Приблизительный вектор не подставляется."
            ),
        }
