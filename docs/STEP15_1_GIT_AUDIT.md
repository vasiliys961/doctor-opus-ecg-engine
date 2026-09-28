# STEP 15.1 — git audit до правок benchmark

HEAD на момент проверки: `3bd9d0042f9f131e84287364c1ca3d05e7885d3f`  
Ветка: `experimental/step13-compatibility-layer`  
Индекс пуст: `git diff --cached --stat` ничего не показал.

`git reset --hard` и `git clean -fd` не выполнялись.

## Изменённые tracked-файлы

Это не код STEP 15. STEP 15 уже в коммите `3bd9d00`. Файлы оставлены как есть и в коммит 15.1 не входят.

| Файл | Класс | Что это |
| --- | --- | --- |
| `README.md`, `backend/README.md`, `frontend/README.md` | документация ранних шагов | описание запуска и границ raw-канала |
| `docs/ARCHITECTURE.md`, `docs/DOCTOR_OPUS_ECG_AUDIT.md`, `docs/ECG_MODEL_SPEC.md`, `docs/SOURCE_PROVENANCE.md` | документация ранних шагов | аудит и спецификация, не benchmark |
| `pyproject.toml`, `requirements.txt` | упаковка API | fastapi/uvicorn/httpx, не benchmark |
| `tests/regression/baseline_comparison.json` | сгенерированный отчёт сверки | два поля `status` и `pass_limit` |
| `tests/regression/test_match_original.py` | код сверки | зависит от неотслеживаемого `tests/compare_with_ecg_web_up.py`; отдельно не коммитится |

## Неотслеживаемые файлы

| Группа | Класс | Коммит 15.1 |
| --- | --- | --- |
| `docs/STEP4`…`STEP14`, forensic maps, `docs/step5_evidence.json` | отчёты forensic-разбора | нет |
| `experiments/raw_to_531/` | расчёты по записи 00513 | нет |
| `research/ecgdeli_reproduction/` README, comparison, logs | журнал воспроизведения | нет |
| `research/.../data` и `output` | данные и вывод, уже в `.gitignore` | нет |
| `tests/compare_with_ecg_web_up.py`, `tests/test_rr_aggregation.py` | вспомогательные тесты вне STEP 15 | нет |

Кэш benchmark и `*.pth` в этот список не попали: каталоги `benchmark/data/` и `benchmark/cache/` игнорируются, веса smoke тоже.

## Вывод

Dirty tree — хвост forensic-работы до STEP 15, а не поломка benchmark. Сбрасывать его нельзя: там неотслеживаемые отчёты и эксперименты. STEP 15.1 добавляет только файлы locator, диагностики и тестов блокеров.
