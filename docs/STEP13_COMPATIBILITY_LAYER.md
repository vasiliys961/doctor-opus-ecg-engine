# STEP 13 — слой совместимости 531

`RAW_TO_531_STATUS = NOT_PROVEN`

The published PTB-XL+ 531-feature vector is the reference representation used by the existing ensemble. The raw-ECG compatibility layer reconstructs a substantial subset of this representation from publicly documented ECGDeli algorithms and published fiducial information, but exact reproduction of the complete original 531-feature generation pipeline has not been established.

## A. Architecture

```text
RAW ECG
 ↓
Preprocessing
 ↓
Feature Registry
 ↓
531 Canonical Schema
 ↓
Provenance
 ↓
Mask
```

Два рабочих режима разделены.

```text
REFERENCE_MODE
published 531 CSV → existing ensemble → 24 outputs

RECONSTRUCTED_RAW_MODE
raw 12-lead ECG → compatibility engine → 531 reconstructed features
                                      → provenance + mask
                                      → NOT PROVEN FOR ENSEMBLE
```

`ECGInferenceAdapter.predict` существует как интерфейс и поднимает `ADAPTER_NOT_TRAINED`. Адаптер не обучался.

## B. Feature status

| Family                  | Columns | Status        | Exact proven | Confidence |
| ----------------------- | ------: | ------------- | ------------ | ---------- |
| Intervals/RR/Framingham |     240 | RECONSTRUCTED | partial      | HIGH       |
| Amplitudes              |     180 | RECONSTRUCTED | no           | MEDIUM     |
| P_Morph                 |      36 | NOT_EXECUTED  | no           | UNKNOWN    |
| QT_IntCorr              |      36 | RECONSTRUCTED | no           | HIGH       |
| ST_Elev                 |      36 | UNKNOWN       | no           | UNKNOWN    |
| HA__Global              |       3 | UNKNOWN       | no           | UNKNOWN    |

`partial` у интервалов означает 238 ячеек записи 00513 с абсолютной ошибкой не выше `1e-6`. Это не статус `EXACT_PROVEN` для всех 240 колонок.

## C. Safety

Reconstructed raw ECG features are not claimed to be numerically identical to the original PTB-XL+ feature-generation pipeline.

Неизвестная колонка не заменяется нулём, медианой или соседним признаком. В сыром режиме значение остаётся пустым, маска равна 0.

## D. Ensemble compatibility

```text
Published 531 CSV:
SUPPORTED

Raw ECG → reconstructed 531:
NOT YET APPROVED FOR EXISTING ENSEMBLE
```

`POST /api/ecg/predict` по-прежнему принимает только готовый конечный вектор 531. `POST /api/ecg/raw/features` возвращает статус `RECONSTRUCTED_RAW_MODE`, маску, статусы и provenance и не возвращает прогноз. Прежний `POST /api/ecg/raw` остаётся 501.
