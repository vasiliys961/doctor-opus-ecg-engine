# Doctor Opus ECG Engine

Исследовательский каркас для ECG. Сейчас в нём работает только уже обученный ансамбль из `ecg_web_up`: CSV с 531 колонкой в фиксированном порядке, три головы, среднее сигмоид, 24 кода SCP.

Приложение не является медицинским изделием. Score модели не калибровался как клиническая вероятность.

## Три входа

```text
531 CSV ───────────────→ этот ансамбль
Raw 12-lead ECG → экстрактор 531 ещё не доказан
ECG image ─────────────→ vision из Doctor Opus, ещё не перенесён
```

Состояние raw-пути: `docs/RAW_TO_531_STATUS.md` (`NOT YET PROVEN`).

## Запуск

Нужен Python 3.11.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m ecg_engine.predict tests/fixtures/another_ecg_features.csv
```

Проверка схемы и сверка с исходным `ecg_web_up`:

```bash
pytest
```

Сверка ищет локальный снимок `ecg_web_up` в `_audit/ecg_web_up` и пропускается, если его нет. Числовой отчёт последнего прогона лежит в `tests/regression/baseline_comparison.json`.

## Документы

- `docs/ECG_MODEL_SPEC.md` — как устроена исходная сеть
- `docs/FEATURE_SCHEMA.md` — контракт колонок и нормализации
- `docs/SCP_MAPPING.md` — индексы и расхождения имён
- `docs/DOCTOR_OPUS_ECG_AUDIT.md` — что есть в production ECG и что не переносилось
- `docs/SOURCE_PROVENANCE.md` — откуда скопированы файлы
- `docs/ARCHITECTURE.md` — что уже есть и что сознательно пусто
