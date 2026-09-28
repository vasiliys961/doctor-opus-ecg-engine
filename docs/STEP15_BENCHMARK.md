# STEP 15 — воспроизводимый benchmark

`RAW_TO_531_STATUS = NOT_PROVEN`

Сырой reconstructed-вектор в существующий ансамбль не передаётся.

## 1. Dataset

PTB-XL 1.0.3 и PTB-XL+ 1.0.1. Волны — `records500`, 500 Гц, 10 с, 12 отведений, 5000 отсчётов. Порядок отведений фиксирован: I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6. Если заголовок в другом порядке, чтение завершается ошибкой и отведения не переставляются. Имена AVR/AVL/AVF только приводятся к aVR/aVL/aVF.

В этом окружении есть метаданные `ptbxl_database.csv` (21799 записей, sha256 `7600de9c1b27d181d850b3c6038a35d7c3ddb6bb33b702e3a20252a6859d216b`). Корпуса `records500` нет: локально только запись 00513. Файл `ecgdeli_features.csv` в `/tmp` не совпал с официальной суммой `84143735682f727201cc7f2825c047f98898f9756c6a21729af993b516221556` и не используется. Датасет скриптом не скачивается. Инструкция: `python benchmark/fetch_dataset.py`.

## 2. Versions

Признаки для пути A — только опубликованный PTB-XL+ 1.0.1 `ecgdeli_features.csv`. Слой совместимости их не пересчитывает.

## 3. Split

Официальные `strat_fold` PTB-XL, не случайные 70/10/20. Пациенты между fold не пересекаются, поэтому одна и та же схема годится для всех моделей.

```text
train = folds 1–8
val   = fold 9
test  = fold 10
```

Запись `ecg_id=513` (`00513`) исключена из всех трёх частей и помечена `forensic_records_excluded`. Остальные записи её пациента в train остаются: исключена запись, не вся семья пациента.

| Часть | ЭКГ | Пациенты |
| --- | ---: | ---: |
| train | 17417 | 15023 |
| val | 2183 | 1942 |
| test | 2198 | 1904 |

Файлы: `benchmark/splits/*.csv`, `benchmark/split_manifest.json`. Пересечение patient_id или ecg_id останавливает benchmark.

## 4. Patient counts

См. таблицу выше. Seed 42 записан в конфиге. Сам split от seed не зависит: это официальные fold.

## 5. ECG counts

21798 записей в split плюс одна forensic-only запись вне split.

## 6. Preprocessing

Путь C: ADC → мВ делением на 1000, как в заголовке PTB-XL gain 1000. Затем per-lead mean/std, посчитанные только на train. Resampling нет. Augmentation выключена. В smoke статистики считаются на синтетическом train и к PTB-XL не относятся.

## 7. Models

A. Существующий ансамбль 531 → MLP, 1D CNN, ResNet1D → среднее трёх сигмоид. Не переобучался. Вход — опубликованные 531 колонки.

B. Compatibility layer STEP 13. Считает покрытие реестра. Неизвестный признак не заменяется нулём, средним или медианой. В ансамбль вектор не идёт. Это не model benchmark.

C. `RawECGCNN`: три Conv1d со stride 2, global pooling, линейный слой на 24 логита. Вход только `[B, 12, 5000]`. `feature_compatibility.py` и `ecgdeli_features.csv` эта модель не импортирует. Метки в вход не входят.

## 8. Training parameters

`benchmark/config.yaml`: AdamW, lr 0.001, weight_decay 0.0001, BCEWithLogitsLoss, class weighting выключен, early stopping по validation macro AUROC, patience 5, до 20 эпох. Порог F1 для каждого класса выбирается на validation сеткой 0.01–0.99 и замораживается. Test для выбора модели и порога не используется.

Полное обучение не запускалось: нет корпуса волн.

## 9. Metrics

Реализованы AUROC (Mann–Whitney со средними рангами), average precision, F1, precision, recall, specificity, sensitivity, micro F1, Brier, ECE по 10 равным бинам. Класс без обоих исходов даёт NaN и в macro не входит; это пишется явно через пропуск NaN.

Чисел PTB-XL для путей A и C нет.

```text
macro AUROC 531 = NOT RUN
macro AUROC RAW = NOT RUN
macro AUPRC 531 = NOT RUN
macro AUPRC RAW = NOT RUN
macro F1 531 = NOT RUN
macro F1 RAW = NOT RUN
```

Smoke на 8 синтетических записях, 2 эпохи, устройство MPS, дал validation macro AUROC 1.0 на этом шуме. Это проверка forward/loss/backward, не результат исследования.

## 10. Thresholds

Код выбора и заморозки порога есть в `benchmark/raw_model/evaluate.py`. На test PTB-XL порог не подбирался, потому что test-оценки нет.

## 11. Confidence intervals

`benchmark/bootstrap.py` пересэмплирует пациентов, 1000 итераций, seed 42, процентили 2.5 и 97.5. На PTB-XL не считался.

## 12. Hardware

Smoke: Apple MPS. Выбор устройства: CUDA, затем MPS, затем CPU.

## 13. Software

Python окружения проекта, torch 2.5.0, numpy 1.26.4, pandas 2.2.3. Отдельные sklearn, wfdb и PyYAML не ставятся. Список: `requirements-benchmark.txt`.

## 14. Random seed

42 для NumPy, PyTorch и bootstrap. DataLoader workers = 0.

## 15. Commands

```text
python benchmark/fetch_dataset.py
python benchmark/run_benchmark.py --dry-run
python benchmark/run_benchmark.py --prepare --ptbxl-database /path/ptbxl_database.csv
python benchmark/run_benchmark.py --smoke-test
python benchmark/run_benchmark.py --evaluate-reference
python benchmark/run_benchmark.py --train-raw
python benchmark/run_benchmark.py --report
```

`--evaluate-reference` сначала сверяет текущий inference на `tests/fixtures/another_ecg_features.csv`. Расхождение имён и CSV-входа: max absolute difference 0.0. Сверка с `ecg_web_up` `bfe7c17` на этой машине тоже 0.0. Дальше команда останавливается, пока официальный CSV признаков не ляжет на ожидаемый путь.

## 16. Limitations

Нет полного `records500` и нет проверенного `ecgdeli_features.csv`, поэтому вопросы «насколько хорош ансамбль на независимом test» и «сколько информации сохраняет raw ECG» этим прогоном не отвечены. Инфраструктура split, метрик, порогов, bootstrap и smoke собрана. Сравнивать модели по smoke AUROC нельзя.

Покрытие compatibility layer — это статус реестра, не число посчитанных на сигнале ячеек. Runtime calculated = 0 из 531. Метод реконструкции отмечен у 456 колонок, `P_Morph` 36 не исполнялись, `ST_Elev` 36 и `HA__Global` 3 неизвестны.

Ground truth для будущего сравнения — SCP-коды PTB-XL: цель положительна, если код есть среди ключей, включая likelihood 0.0. Предсказания ECGDeli и собственной сети ground truth не являются.
