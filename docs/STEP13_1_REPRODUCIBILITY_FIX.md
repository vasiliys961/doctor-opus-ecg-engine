# STEP 13.1 — воспроизводимый commit

Родитель: `093ebb1` на ветке `experimental/step13-compatibility-layer`.

## Дефект

`backend/app.py` в `093ebb1` импортирует `predict_features` и `service_payload` из `ecg_engine.ensemble`. Этих функций в committed `ensemble.py` не было. `read_feature_mapping`, который им нужен, лежал только в незакоммиченном `schema.py`.

Чистый `git archive 093ebb1` падал так:

```text
ImportError: cannot import name 'predict_features'
```

Рабочее дерево при этом импортировалось, потому что определения остались незакоммиченными после STEP 13. STEP 14 это и зафиксировал.

## Причина

Commit STEP 13 добавил HTTP-сервис целиком. Функции входа в ансамбль по именам колонок уже были написаны в рабочем дереве и не вошли в snapshot. Импорт не заглушался: без этих функций `POST /api/ecg/predict` не к чему привязать.

## Что вошло в исправление

- `ecg_engine/schema.py` — `read_feature_mapping`. Имена раскладываются в канонический порядок. Пропуск и лишняя колонка отвергаются, нулём не заполняются.
- `ecg_engine/ensemble.py` — `predict_features` и `service_payload`. Формула `predict` не менялась: по-прежнему среднее трёх сигмоид. Отказ на NaN после среднего в commit не переносился: для самодостаточности он не нужен и менял бы поведение `predict`.
- `tests/test_feature_compatibility_00513.py` — контракт слоя проверяется и без gitignored-сигнала `00513_hr.dat`. Если файл есть локально, проверяется он.
- `docs/RAW_TO_531_STATUS.md` — вступительные пункты больше не говорят, что сырого HTTP-пути нет. `POST /api/ecg/raw` остаётся 501. `POST /api/ecg/raw/features` описан как слой пустых ячеек. Статус `NOT_PROVEN`.
- `docs/STEP13_COMPATIBILITY_LAYER.md` — `value_00513` подписан как forensic-колонка, не как выход API.

Веса, архитектура, порядок 531, нормализация, 24 метки и усреднение ансамбля не менялись. `doctor-opus-global` не менялся. Сырой вектор в ансамбль не подключён.

## Проверка чистого дерева

Сначала worktree на `093ebb1` только с этими файлами, без остальных dirty-изменений:

```text
import backend.app          PASS
pytest tests                25 passed, 0 failed, 1 skipped
```

Пропущен один тест: committed `tests/regression/test_match_original.py` не находит `_audit/ecg_web_up`, потому что снимок исходного репозитория в git не входит. Это не подмена результата. Та же сверка запущена отдельно на коде этого дерева против `ecg_web_up` `bfe7c17`, CSV `tests/fixtures/another_ecg_features.csv`.

| Голова | max absolute difference |
| --- | ---: |
| MLP | 0.0 |
| CNN | 0.0 |
| ResNet | 0.0 |
| ensemble, 24 выхода | 0.0 |

`RAW_TO_531_STATUS = NOT_PROVEN`.

Числа выше сняты с дерева `093ebb1` плюс только файлы этого исправления, до commit. Hash нового commit в этот файл не вписан: он известен только после записи. Повтор import и pytest на чистом worktree нового commit обязателен и приводится в ответе шага.
