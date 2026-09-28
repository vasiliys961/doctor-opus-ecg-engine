# Происхождение исходников

На шаге аудита код в новый репозиторий не копировался. Таблица ниже фиксирует снимки, по которым написаны `docs/ECG_MODEL_SPEC.md` и `docs/DOCTOR_OPUS_ECG_AUDIT.md`, и список компонентов, которые можно будет copy/adapt позже.

Дата аудита: 2026-09-28.

Локальные клоны для чтения лежат в `_audit/` и в состав приложения не входят. Оба исходных репозитория этим аудитом не изменялись.

## Снимки

| Репозиторий | Коммит | Дата коммита | Лицензия в снимке |
| --- | --- | --- | --- |
| `https://github.com/vasiliys961/ecg_web_up` | `bfe7c1738baca97f9f501ede25f1716f2aa94cf6` | 2026-09-25 20:23:28 +0300 | файла LICENSE нет |
| `https://github.com/vasiliys961/doctor-opus-global` | `9221b37abbd5423d51f9f294c9a1873365d8e9e1` | 2026-09-25 18:48:07 +0300 | `LICENSE` в корне, copyright 2025, Vasily |

Клоны shallow (`--depth 1`): проверен HEAD, история до него не просматривалась.

## Лицензия Doctor Opus

`doctor-opus-global/LICENSE` разрешает личное, учебное, исследовательское и внутреннее некоммерческое использование. Коммерческое использование запрещено без письменного разрешения автора. Распространение копий требует сохранять файл LICENSE и указание автора. Имя Doctor-Opus нельзя использовать для продвижения производного продукта без разрешения. Софт не является медицинским изделием.

Любой файл, скопированный из `doctor-opus-global`, нужно сопровождать этой лицензией и атрибуцией. Перед публикацией нового репозитория нужно отдельно решить, совместима ли его лицензия с этим ограничением. Молчаливый relicense невозможен.

У `ecg_web_up` лицензии в репозитории нет. Веса `models/*.pth` и `*.npy` лежат в том же снимке. Перенос весов и инференса в публичный репозиторий лучше подтвердить у автора до публикации, даже если репозиторий сейчас открыт на GitHub.

## Реестр переноса

Код Doctor Opus по-прежнему не копировался. Из `ecg_web_up` на 2026-09-28 перенесены веса, статистика нормализации, эталонный CSV и логика инференса.

SHA-256 весов и статистики совпадает с файлами в снимке `bfe7c173`:

| Файл в новом репозитории | SHA-256 |
| --- | --- |
| `models/ecg_ensemble/ecg_model.pth` | `363c52370d483d3ac9fb89a3750a1048118bac80ef8d4c53dbd5df818df4c4c5` |
| `models/ecg_ensemble/ecg_1dcnn_best.pth` | `48ca8fc667a99cbed05216e33f4cb39516d8cdafea0c2c3bf1d6aeb940dec417` |
| `models/ecg_ensemble/ecg_resnet1d_features_best.pth` | `de2411bfc188e988d4730bc41d15359851407f60174063676067da00e9ef77bb` |
| `models/ecg_ensemble/ecg_train_mean.npy` | `e0488d85e7839ee25abbe13c0bce1ca91712938bd271448b871e201ca35575bb` |
| `models/ecg_ensemble/ecg_train_std.npy` | `c7938bd9169be07ed12f7216cd8be948b134bedf7af00c8b77cd557a3418ab28` |

| Компонент | Репозиторий | Путь | Коммит | Дата копирования | Изменения | Лицензия |
| --- | --- | --- | --- | --- | --- | --- |
| Инференс ансамбля | ecg_web_up | `analysis_scripts/predict_csv.py` | `bfe7c173` | 2026-09-28 | Переписан в `ecg_engine/networks.py` и `ecg_engine/ensemble.py`. Архитектуры и среднее сигмоид те же. NaN больше не заполняется молча | не указана |
| Веса MLP | ecg_web_up | `models/ecg_model.pth` | `bfe7c173` | 2026-09-28 | байтовая копия в `models/ecg_ensemble/` | не указана |
| Веса 1D CNN | ecg_web_up | `models/ecg_1dcnn_best.pth` | `bfe7c173` | 2026-09-28 | байтовая копия | не указана |
| Веса ResNet1D | ecg_web_up | `models/ecg_resnet1d_features_best.pth` | `bfe7c173` | 2026-09-28 | байтовая копия | не указана |
| Нормализация | ecg_web_up | `models/ecg_train_mean.npy`, `models/ecg_train_std.npy` | `bfe7c173` | 2026-09-28 | байтовая копия | не указана |
| Эталон 531 CSV | ecg_web_up | `templates/another_ecg_features.csv` | `bfe7c173` | 2026-09-28 | копия в `tests/fixtures/another_ecg_features.csv` | не указана |
| Страница ЭКГ | doctor-opus-global | `app/ecg/page.tsx` | `9221b37a` | — | не копировался | LICENSE Doctor Opus |
| Линейка | doctor-opus-global | `components/EcgCaliper.tsx` | `9221b37a` | — | не копировался | LICENSE Doctor Opus |
| Редактор маски | doctor-opus-global | `components/ImageEditor.tsx` | `9221b37a` | — | не копировался | LICENSE Doctor Opus |
| ECG-критерии и промпт наблюдателя | doctor-opus-global | `lib/prompts.ts` (блоки `ecg`) | `9221b37a` | — | не копировался | LICENSE Doctor Opus |
| Промпт функционального заключения | doctor-opus-global | `lib/diagnostic-report.ts` | `9221b37a` | — | не копировался | LICENSE Doctor Opus |
| Шаблон заключения | doctor-opus-global | `lib/protocol-templates.ts` (`ecg-functional-conclusion`) | `9221b37a` | — | не копировался | LICENSE Doctor Opus |
| Ссылки по тексту ЭКГ | doctor-opus-global | `lib/ecg-reference-links.ts` | `9221b37a` | — | не копировался | LICENSE Doctor Opus |
| Приведение формата и компрессия | doctor-opus-global | `lib/server-image-processing.ts` | `9221b37a` | — | не копировался | LICENSE Doctor Opus |

`lib/prompts.ts` и `lib/server-image-processing.ts` содержат много не-ECG кода. При переносе брать только ECG-нужные функции, а не файлы целиком.

Не переносить как runtime-зависимость и не импортировать из production:

- `lib/openrouter.ts`
- биллинг, auth, база пациентов
- `app/api/analyze/ecg/route.ts` (страница его не вызывает)
- Docker, nginx и деплой `doctor-opus-global`

Секрет `app.secret_key = 'supersecretkey'` из `ecg_web_up/app.py` в новый проект не переносится.
