Да, теперь понял задачу точно. **`doctor-opus-global` вообще не трогаем.** Он остаётся production.

Мы создаём **новый самостоятельный репозиторий**, который объединяет две уже существующие технологии:

1. твою обученную ECG-сеть из `ecg_web_up`;
2. ECG-код/логику, уже реализованную в `doctor-opus-global`.

И уже внутри нового приложения экспериментируем с raw ECG → 531 → твоя сеть, image ECG, fusion и новой raw-моделью.

Это гораздо правильнее для текущего этапа. Я бы даже назвал проект условно **Doctor Opus ECG Lab / Doctor Opus ECG Engine**.

# ТЕХНИЧЕСКОЕ ЗАДАНИЕ ДЛЯ GROK

# Новый самостоятельный проект Doctor Opus ECG Engine

## 0. КРИТИЧЕСКОЕ УСЛОВИЕ

Создаётся **НОВЫЙ GitHub-репозиторий**.

Рабочие исходные проекты:

### Проект A — моя обученная ECG нейросеть

`https://github.com/vasiliys961/ecg_web_up`

### Проект B — production Doctor Opus

`https://github.com/vasiliys961/doctor-opus-global`

## ЗАПРЕЩЕНО

Не изменять:

* `doctor-opus-global`;
* production code;
* production API;
* production prompts;
* production model routing;
* production deployment.

`doctor-opus-global` использовать **только как источник кода, архитектурных решений и ECG-функциональности для анализа и переноса необходимых компонентов**.

Также не изменять существующие веса и inference-логику ECG-сети без необходимости.

---

# 1. ЦЕЛЬ НОВОГО ПРОЕКТА

Создать самостоятельное приложение:

# Doctor Opus ECG Engine

Это не просто API и не просто копия `ecg_web_up`.

Это отдельная исследовательско-прикладная ECG AI платформа, которая должна объединить:

```text
                 Doctor Opus ECG Engine
                          │
          ┌───────────────┼────────────────┐
          │               │                │
          ▼               ▼                ▼
       RAW ECG        ECG IMAGE       531 FEATURES
          │               │                │
          ▼               ▼                │
   Feature Engine       ECG Vision          │
          │               │                │
          ▼               │                │
      531 features        │                │
          │               │                │
          └───────────────┼────────────────┘
                          ▼
                MY ECG ENSEMBLE
              MLP + CNN + ResNet
                          │
                          ▼
                    24 SCP codes
                          │
                          ▼
                   ECG Evidence
                          │
                          ▼
                    ECG Reasoning
```

Главная задача первой версии:

> Проверить, можно ли брать обычное цифровое raw 12-lead ECG, самостоятельно получать из него те 531 признака, на которых обучена моя сеть, и затем использовать существующую сеть без переобучения.

Параллельно приложение должно использовать ECG-функциональность, уже реализованную в Doctor Opus.

---

# 2. АРХИТЕКТУРНЫЙ ПРИНЦИП

Не копировать весь Doctor Opus.

Нужно:

```text
ecg_web_up
      +
ECG-related code from doctor-opus-global
      ↓
NEW REPOSITORY
      ↓
Doctor Opus ECG Engine
```

То есть новый проект должен быть **самостоятельным ECG-приложением**, а не fork всего Doctor Opus.

---

# 3. ПЕРЕД НАПИСАНИЕМ КОДА

Сначала провести аудит обоих репозиториев.

## Репозиторий A

Изучить:

`ecg_web_up`

Особенно:

```text
analysis_scripts/predict_csv.py
```

Определить:

* архитектуру MLP;
* архитектуру 1D CNN;
* архитектуру ResNet1D;
* model weights;
* normalization;
* 531 feature schema;
* 24 SCP codes;
* ensemble logic;
* preprocessing;
* thresholding;
* prediction output.

---

## Репозиторий B

Изучить:

`doctor-opus-global`

Но анализировать **только ECG-related функциональность**.

Найти:

* ECG page;
* ECG upload;
* ECG image handling;
* ECG analysis flow;
* ECG prompts;
* image preprocessing;
* ECG-specific UI;
* ECG-specific components;
* ECG result rendering;
* feedback;
* digital caliper;
* ruler;
* redaction/editor;
* multimodal image analysis;
* model routing, относящийся к ECG.

Не переносить весь Doctor Opus.

Создать:

```text
docs/DOCTOR_OPUS_ECG_AUDIT.md
```

---

# 4. НОВЫЙ REPOSITORY

Создать новый GitHub repository.

Предварительное название:

```text
doctor-opus-ecg-engine
```

Название можно изменить до начала разработки.

Рекомендуемая структура:

```text
doctor-opus-ecg-engine/

├── README.md
├── LICENSE
├── .gitignore
├── .env.example
│
├── frontend/
│
├── backend/
│
├── ecg_engine/
│   ├── models/
│   ├── inference/
│   ├── preprocessing/
│   ├── features/
│   ├── parsers/
│   ├── quality/
│   └── fusion/
│
├── doctor_opus_ecg/
│   ├── image_analysis/
│   ├── prompts/
│   ├── measurements/
│   └── ui/
│
├── models/
│   └── ecg_ensemble/
│
├── data/
│   ├── examples/
│   └── test/
│
├── tests/
│
├── docs/
│
└── scripts/
```

---

# 5. ДВЕ ОСНОВНЫЕ ТЕХНОЛОГИИ

Новый проект должен иметь два независимых ECG analysis channels.

## CHANNEL A

Моя существующая обученная ECG-сеть:

```text
531 features
       ↓
MLP
CNN
ResNet
       ↓
ensemble
       ↓
24 SCP
```

## CHANNEL B

ECG functionality, перенесённая из Doctor Opus:

```text
ECG image
       ↓
Vision analysis
       ↓
structured ECG findings
```

Они должны работать независимо.

---

# 6. ОСНОВНАЯ ЗАДАЧА — RAW ECG

Создать pipeline:

```text
RAW ECG
   ↓
Parser
   ↓
Signal normalization
   ↓
Signal Quality
   ↓
Feature Extraction
   ↓
531 features
   ↓
Existing ECG Ensemble
   ↓
24 SCP predictions
```

Но нельзя предполагать, что это автоматически возможно.

Сначала необходимо установить происхождение 531 признака.

---

# 7. ИССЛЕДОВАНИЕ 531 FEATURES

Создать:

```text
docs/FEATURE_531_SPEC.md
```

Определить для каждого из 531 признаков:

```text
index
name
source
calculation
lead
unit
required input
```

Особенно определить:

* какие признаки относятся к morphology;
* какие к intervals;
* какие к amplitudes;
* какие к rhythm;
* какие к QRS;
* какие к P-wave;
* какие к T-wave;
* какие зависят от fiducial points;
* какие зависят от нескольких отведений.

---

# 8. ECGDeli

Проверить, являются ли 531 признаков ECGDeli-derived.

Если да:

## ПРИОРИТЕТ

Использовать оригинальный ECGDeli feature extraction pipeline.

Не создавать приблизительную замену, если оригинальный код доступен.

Цель:

```text
raw ECG
    ↓
ECGDeli
    ↓
same 531 feature space
```

---

# 9. ЭТАЛОННЫЙ ТЕСТ

Найти ECG, для которого имеются:

```text
RAW ECG
+
531 FEATURE CSV
```

Использовать его как gold-standard test.

Pipeline:

```text
RAW ECG
   ↓
new extractor
   ↓
531 features A
```

сравнить:

```text
known 531 features B
```

Рассчитать:

```text
absolute error
relative error
correlation
```

Для каждого признака.

Создать отчёт:

```text
docs/FEATURE_EXTRACTION_VALIDATION.md
```

---

# 10. КРИТИЧЕСКИЙ GATE

Не считать raw ECG pipeline готовым только потому, что получено:

```text
531 numbers
```

Необходимо доказать:

```text
new 531 features
≈
original 531 features
```

Если этого доказать нельзя:

```text
RAW ECG → existing model
```

не объявлять валидированным.

В этом случае зафиксировать ограничение и перейти к разработке отдельной raw ECG model.

---

# 11. СУЩЕСТВУЮЩАЯ ECG MODEL

Интегрировать мою существующую модель без изменения весов.

Сохранить:

```text
MLP
1D CNN
ResNet1D
```

Сохранить:

```text
ecg_model.pth
ecg_1dcnn_best.pth
ecg_resnet1d_features_best.pth
```

Сохранить normalization:

```text
ecg_train_mean.npy
ecg_train_std.npy
```

Сохранить:

```text
531 feature order
```

---

# 12. REGRESSION TEST

До любых изменений создать baseline.

Для существующего 531-feature CSV:

```text
old ecg_web_up
```

и:

```text
new Doctor Opus ECG Engine
```

должны дать практически идентичный результат.

Сохранить baseline:

```text
tests/regression/baseline_predictions.json
```

---

# 13. ECG IMAGE

Перенести из Doctor Opus только ECG-specific image functionality.

Поддержать:

```text
PNG
JPG
JPEG
PDF
```

если существующий код это позволяет.

Pipeline:

```text
ECG image
      ↓
image preprocessing
      ↓
vision model
      ↓
ECG findings
```

Не смешивать image model с 531-feature model до этапа Evidence Fusion.

---

# 14. DIGITAL CALIPER

Если в Doctor Opus уже существует ECG digital caliper:

перенести его в новый проект.

Функции:

```text
distance measurement
time measurement
voltage measurement
grid interpretation
```

Должны быть доступны пользователю.

---

# 15. ECG MEASUREMENTS

Создать единый объект:

```json
{
  "heart_rate": null,
  "rhythm": null,
  "pr_ms": null,
  "qrs_ms": null,
  "qt_ms": null,
  "qtc_ms": null,
  "axis": null
}
```

Не заполнять значения, если они не были реально измерены.

---

# 16. SIGNAL QUALITY

Создать отдельный:

```text
ECG Signal Quality Engine
```

Проверять:

* количество отведений;
* sampling rate;
* длительность;
* шум;
* baseline wander;
* clipping;
* flatline;
* missing data;
* lead integrity.

Результат:

```json
{
  "status": "good",
  "score": 0.93,
  "warnings": []
}
```

---

# 17. ЕДИНЫЙ ECG EVIDENCE OBJECT

Все анализаторы должны отдавать результаты в едином формате:

```json
{
  "input": {},
  "signal_quality": {},
  "measurements": {},
  "neural_network": {},
  "vision": {},
  "feature_engine": {},
  "warnings": {}
}
```

Это главный внутренний контракт нового приложения.

---

# 18. NEURAL NETWORK OUTPUT

Существующая сеть выдаёт:

```text
24 SCP codes
```

Например:

```text
AFIB
LVH
PVC
LAFB
...
```

Сохранять raw output всех моделей:

```json
{
  "MLP": {},
  "CNN1D": {},
  "ResNet1D": {},
  "ensemble": {}
}
```

---

# 19. НЕ ДЕЛАТЬ ЛОЖНУЮ CALIBRATION

Если probability не была клинически откалибрована:

не писать:

```text
87% probability of atrial fibrillation
```

Лучше:

```text
model score = 0.87
```

или:

```text
neural network output = 0.87
```

---

# 20. EVIDENCE FUSION

Создать:

```text
ECG Evidence Fusion
```

Он должен объединять:

```text
existing neural network
+
ECG image analysis
+
measurements
+
signal quality
```

Например:

```text
Neural model:
AFIB score 0.91

Vision:
irregular rhythm suspected

Measurement:
RR variability increased

→ concordant evidence
```

Другой случай:

```text
Neural model:
AFIB score 0.91

Vision:
regular rhythm

Measurement:
regular RR

→ discordant evidence
```

Discordance обязательно должна сохраняться в результате.

---

# 21. CLINICAL REASONING

В приложении должен быть отдельный reasoning layer.

Его задача:

не просто перечислять outputs моделей, а сопоставлять:

```text
neural evidence
+
visual evidence
+
measurements
+
signal quality
```

и формировать структурированный ECG interpretation.

При discordance reasoning должен явно сообщать о противоречии.

---

# 22. НОВАЯ RAW ECG MODEL — ВТОРОЙ ЭТАП

После завершения первой версии создать возможность обучения отдельной модели:

```text
Raw ECG
   ↓
Raw ECG neural network
   ↓
24 SCP
```

Эта модель НЕ заменяет мою существующую сеть.

Она становится вторым независимым экспертом.

---

# 23. RAW ECG MODEL

Исследовать:

```text
1D CNN
ResNet1D
InceptionTime
Transformer
```

Начать с сильного baseline:

```text
ResNet1D
```

Затем сравнить с другими архитектурами.

---

# 24. TRAINING DATA

Исследовать открытые datasets.

В первую очередь:

```text
PTB-XL
```

Также рассмотреть другие открытые ECG datasets при наличии совместимой разметки.

Перед обучением определить:

```text
leads
sampling rate
duration
labels
patient split
class distribution
```

---

# 25. DATA SPLIT

Критически важно:

не допускать попадания ECG одного пациента одновременно в train и test.

Использовать:

```text
patient-level split
```

---

# 26. METRICS

Не использовать только accuracy.

Обязательно:

```text
AUROC macro
AUPRC macro
F1 macro
F1 per class
sensitivity
specificity
PPV
NPV
```

Для редких классов особенно важен AUPRC.

---

# 27. RAW MODEL + EXISTING MODEL

После обучения:

```text
                   RAW ECG
                      │
           ┌──────────┴──────────┐
           ▼                     ▼
    531-feature model       Raw ECG model
           │                     │
           ▼                     ▼
       24 outputs            24 outputs
           └──────────┬──────────┘
                      ▼
                Evidence Fusion
```

Это одна из главных целей проекта.

---

# 28. ECG IMAGE + RAW ECG

Если пользователь загружает:

```text
ECG image
```

приложение должно анализировать изображение.

Если пользователь загружает:

```text
raw ECG
```

приложение должно запускать raw pipeline.

Если доступны оба:

```text
image + raw ECG
```

запускать оба независимых канала.

---

# 29. ТРЕТИЙ КАНАЛ — ECG DIGITIZATION

В будущем:

```text
ECG IMAGE
    ↓
digitization
    ↓
raw waveform
    ↓
531 extractor
    ↓
existing network
```

Это должно быть предусмотрено архитектурой, но не обязательно реализовывать в первой версии.

---

# 30. FRONTEND

Приложение должно иметь простой ECG-centric интерфейс.

Главный экран:

```text
Upload ECG

[ RAW ECG ]
[ ECG IMAGE ]
[ 531 FEATURES ]
```

После загрузки:

```text
Signal Quality
        ↓
Measurements
        ↓
Neural Network
        ↓
Vision
        ↓
Evidence Fusion
        ↓
Clinical Interpretation
```

---

# 31. РЕЖИМ RAW ECG

Пользователь загружает raw ECG.

Приложение показывает:

```text
12-lead waveform
```

с возможностью:

* zoom;
* pan;
* lead selection;
* measurement;
* caliper.

---

# 32. РЕЖИМ 531 FEATURES

Можно загрузить существующий CSV.

Приложение должно:

```text
validate 531 columns
↓
normalize
↓
MLP
CNN
ResNet
↓
ensemble
```

Этот режим является главным regression test.

---

# 33. РЕЖИМ ECG IMAGE

Загрузить:

```text
JPG
PNG
PDF
```

и выполнить image analysis.

---

# 34. API

Приложение должно иметь API.

Минимально:

```http
POST /api/ecg/analyze
```

и:

```http
POST /api/ecg/features
```

и:

```http
POST /api/ecg/predict
```

---

# 35. API RAW ECG

```text
POST /api/ecg/analyze
```

Input:

```text
file
input_type
sampling_rate
lead configuration
```

Output:

```json
{
  "input": {},
  "signal_quality": {},
  "features": {
    "count": 531,
    "schema_version": "..."
  },
  "neural_network": {},
  "vision": {},
  "measurements": {},
  "fusion": {},
  "warnings": []
}
```

---

# 36. НЕ ПРИВЯЗЫВАТЬСЯ К DOCTOR OPUS GLOBAL

Новый проект должен иметь собственные:

```text
frontend
backend
prompts
API
configuration
deployment
tests
```

Он не должен импортировать runtime-код напрямую из production `doctor-opus-global`.

Если код переносится:

```text
copy/adapt
```

и зафиксировать источник:

```text
docs/SOURCE_PROVENANCE.md
```

---

# 37. SOURCE PROVENANCE

Для каждого перенесённого компонента записать:

```text
original repository
original path
original commit
date copied
modifications
license
```

---

# 38. TEST DATA

Создать:

```text
data/test/
```

с:

* несколькими 531-feature ECG;
* raw ECG;
* ECG images;
* deliberately corrupted files.

Не коммитить персональные медицинские данные.

---

# 39. AUTOMATED TESTS

Обязательно:

```text
pytest
```

для Python backend.

Проверить:

```text
parser
preprocessing
feature extraction
531 validation
normalization
MLP
CNN
ResNet
ensemble
API
signal quality
fusion
```

---

# 40. REGRESSION TEST

Самый важный тест:

```text
ecg_web_up
        VS
new Doctor Opus ECG Engine
```

На одинаковом 531-feature CSV.

Результаты должны совпадать в пределах установленного tolerance.

---

# 41. ERROR HANDLING

Обработать:

```text
wrong file
wrong leads
wrong sampling rate
wrong number of features
missing columns
NaN
Inf
corrupt ECG
poor signal
unsupported format
model unavailable
LLM unavailable
```

Никогда не выдавать молча результат для некорректного ECG.

---

# 42. MODEL VERSIONING

Каждый результат должен содержать:

```json
{
  "engine_version": "...",
  "feature_schema_version": "...",
  "model_version": "...",
  "preprocessing_version": "..."
}
```

---

# 43. ПРОИЗВОДИТЕЛЬНОСТЬ

PyTorch models должны загружаться один раз при запуске сервера.

Не загружать веса на каждый запрос.

---

# 44. DEPLOYMENT

Архитектура:

```text
Frontend
   │
   ▼
Backend API
   │
   ├── ECG parser
   ├── feature engine
   ├── neural network
   ├── image analysis
   ├── measurements
   └── fusion
```

Можно использовать:

```text
Next.js
+
FastAPI
+
PyTorch
```

или другой обоснованный стек.

Но не усложнять архитектуру без необходимости.

---

# 45. ПОРЯДОК РАБОТЫ GROK

Работать строго по этапам.

## STEP 1 — AUDIT

Проанализировать:

```text
ecg_web_up
doctor-opus-global
```

Создать:

```text
docs/ECG_MODEL_SPEC.md
docs/DOCTOR_OPUS_ECG_AUDIT.md
docs/SOURCE_PROVENANCE.md
```

На этом этапе код нового приложения практически не писать.

---

## STEP 2 — NEW REPOSITORY

Создать структуру нового проекта.

Запустить:

```text
frontend
backend
tests
```

---

## STEP 3 — PORT EXISTING ECG NETWORK

Перенести inference существующей сети.

Проверить:

```text
same input
→ same prediction
```

---

## STEP 4 — PORT ECG FUNCTIONALITY FROM DOCTOR OPUS

Перенести только ECG-related functionality.

Не переносить весь Doctor Opus.

---

## STEP 5 — 531 FEATURE ENGINE

Исследовать происхождение 531 features.

Создать extractor.

Проверить на эталонных данных.

---

## STEP 6 — RAW ECG

Реализовать:

```text
RAW ECG
→ preprocessing
→ 531 features
→ existing ensemble
```

только после успешного Step 5.

---

## STEP 7 — ECG IMAGE

Интегрировать ECG image analysis.

---

## STEP 8 — EVIDENCE FUSION

Объединить:

```text
NN
+
Vision
+
Measurements
+
Signal Quality
```

---

## STEP 9 — RAW ECG MODEL

Создать отдельный training pipeline.

---

## STEP 10 — RAW MODEL INFERENCE

Добавить вторую независимую raw ECG model.

---

## STEP 11 — MULTI-EXPERT ECG ENGINE

Получить:

```text
531-feature expert
+
Raw ECG expert
+
Vision expert
+
Measurement expert
```

---

# 46. ФИНАЛЬНАЯ АРХИТЕКТУРА

Итоговое приложение должно выглядеть концептуально:

```text
                 DOCTOR OPUS ECG ENGINE
                          │
       ┌──────────────────┼──────────────────┐
       │                  │                  │
       ▼                  ▼                  ▼
    RAW ECG           ECG IMAGE        531 CSV
       │                  │                  │
       ▼                  ▼                  │
 Raw preprocessing      Vision              │
       │                  │                  │
       ▼                  │                  │
  531 extractor           │                  │
       │                  │                  │
       ▼                  │                  │
 Existing ECG NN          │                  │
       │                  │                  │
       │                  │                  │
       └────────────┬─────┴──────────────────┘
                    ▼
             ECG Evidence Layer
                    │
          ┌─────────┴─────────┐
          │                   │
          ▼                   ▼
   Measurements         Signal Quality
          │                   │
          └─────────┬─────────┘
                    ▼
              Clinical Reasoning
                    │
                    ▼
               ECG REPORT
```

После добавления новой raw ECG model:

```text
                    RAW ECG
                       │
              ┌────────┴────────┐
              ▼                 ▼
       531 Feature Model    Raw ECG Model
              │                 │
              └────────┬────────┘
                       ▼
                  ECG Fusion
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
       Vision     Measurements   Quality
          └────────────┼────────────┘
                       ▼
                Clinical Reasoning
                       │
                       ▼
                  ECG REPORT
```

---

# 47. ГЛАВНАЯ ЦЕЛЬ

Не пытаться сделать сразу «идеальную медицинскую нейросеть».

Главная исследовательская цель:

### Вопрос №1

Можем ли мы:

```text
RAW ECG
→
те же 531 features
→
моя существующая сеть
```

и получить воспроизводимый результат?

### Вопрос №2

Если это возможно, насколько хорошо работает моя сеть на ECG, которое никогда не проходило через мой первоначальный CSV pipeline?

### Вопрос №3

Даёт ли добавление raw ECG model дополнительную диагностическую информацию?

### Вопрос №4

Даёт ли комбинация:

```text
existing ECG NN
+
raw ECG NN
+
ECG Vision
```

более устойчивый результат, чем любой отдельный канал?

---

# 48. ОГРАНИЧЕНИЕ

До проведения независимой клинической валидации приложение является:

```text
research / decision-support system
```

а не автономной диагностической системой.

Не заявлять clinical-grade performance только на основании внутренних тестов.

---

# 49. DEFINITION OF DONE — ПЕРВАЯ ВЕРСИЯ

Первая полноценная версия считается готовой, если:

```text
[ ] создан новый GitHub repository
[ ] doctor-opus-global не изменён
[ ] ecg_web_up не сломан
[ ] существующая модель работает
[ ] 531 CSV принимается
[ ] результаты совпадают с исходным ecg_web_up
[ ] ECG image анализируется
[ ] ECG measurements работают
[ ] signal quality работает
[ ] создан единый Evidence Object
[ ] создан Evidence Fusion
[ ] raw ECG parser работает
[ ] исследована возможность raw ECG → 531
[ ] создан validation report
[ ] API работает
[ ] frontend работает
[ ] есть automated tests
[ ] есть документация
```

После этого:

```text
RAW ECG
→
531
→
моя сеть
```

становится основным экспериментальным направлением.

---

# 50. ФИНАЛЬНОЕ ПРАВИЛО ДЛЯ GROK

Не спешить писать код.

Сначала:

```text
AUDIT
↓
ARCHITECTURE
↓
PROOF OF 531 FEATURE ORIGIN
↓
BASELINE
↓
IMPLEMENTATION
↓
VALIDATION
```

И главное:

> **Никаких изменений в production `doctor-opus-global`. Весь эксперимент, перенос ECG-кода, raw ECG pipeline, 531-feature extraction, новая raw ECG model и fusion реализуются исключительно в новом репозитории.**

Вот теперь архитектура соответствует твоей задумке: **не модифицируем работающий Doctor Opus, а создаём отдельную лабораторию/приложение, где собираем в одном месте всё лучшее из двух проектов и экспериментируем дальше.**

И я бы **не давал Grok сразу весь этот документ на выполнение одним махом**. Лучше сначала отправить ему только **STEP 1 — аудит двух репозиториев**, получить от него отчёт, и уже после этого дать команду на создание нового репозитория. Так мы не позволим ему самостоятельно додумывать, как устроены твои 531 признаков и ECG-часть Opus.
