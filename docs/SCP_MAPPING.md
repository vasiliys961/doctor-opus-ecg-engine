# SCP mapping

Индекс выхода сети совпадает с `TOP_24_CODES` в `ecg_web_up`. Код модели не переименовывался.

`canonical_description` и `statement_category` взяты из `scp_statements.csv` PTB-XL 1.0.3 (PhysioNet). `legacy_display_name` — русская строка из `analysis_scripts/predict_csv.py`. Она остаётся в таблице как историческая подпись. В JSON инференса наружу идут коды и числа, без замены имени выхода.

| index | model_code | canonical_description | statement_category | legacy_display_name | note |
| ---: | --- | --- | --- | --- | --- |
| 0 | SR | sinus rhythm | impulse formation | Синусовый ритм | |
| 1 | NORM | normal ECG | Normal/abnormal | Нормальная ЭКГ | |
| 2 | ABQRS | abnormal QRS | Normal/abnormal | Аберрантный QRS комплекс | |
| 3 | IMI | inferior myocardial infarction | Myocardial Infarction | Инфаркт миокарда (нижняя стенка) | |
| 4 | ASMI | anteroseptal myocardial infarction | Myocardial Infarction | Инфаркт миокарда (переднеперегородочная стенка) | |
| 5 | LVH | left ventricular hypertrophy | Ventricular Hypertrophy | Гипертрофия левого желудочка | |
| 6 | NDT | non-diagnostic T abnormalities | other ST-T descriptive statements | Неспецифические изменения ST-T | канонически только T |
| 7 | LAFB | left anterior fascicular block | Conduction disturbances | Блокада передней ветви левой ножки пучка Гиса | |
| 8 | AFIB | atrial fibrillation | impulse formation | Фибрилляция предсердий | |
| 9 | PVC | ventricular premature complex | ectopic rhythm | Преждевременное желудочковое сокращение | |
| 10 | IRBBB | incomplete right bundle branch block | Conduction disturbances | Неполная блокада правой ножки пучка Гиса | |
| 11 | VCLVH | voltage criteria (QRS) for left ventricular hypertrophy | Ventricular Hypertrophy | Гипертрофия желудочков или левого желудочка | это вольтажный критерий, не общий диагноз ГЛЖ |
| 12 | STACH | sinus tachycardia | impulse formation | Синусовая тахикардия | |
| 13 | IVCD | non-specific intraventricular conduction disturbance (block) | Conduction disturbances | Внутрижелудочковая блокада | |
| 14 | SARRH | sinus arrhythmia | impulse formation | Синусовый ритм с аберрантным проведением | подпись в коде описывает другой феномен |
| 15 | ISCAL | ischemic in anterolateral leads | ischemic ST-T changes | Ишемия миокарда (нижняя стенка) | нижняя стенка в SCP — ISCI |
| 16 | SBRAD | sinus bradycardia | impulse formation | Синусовая брадикардия | |
| 17 | QWAVE | Q waves present | QRS morphology | Патологический Q-волновой комплекс | |
| 18 | CRBBB | complete right bundle branch block | Conduction disturbances | Полная блокада правой ножки пучка Гиса | |
| 19 | CLBBB | complete left bundle branch block | Conduction disturbances | Полная блокада левой ножки пучка Гиса | |
| 20 | ILMI | inferolateral myocardial infarction | Myocardial Infarction | Инфаркт миокарда (нижнебоковая стенка) | |
| 21 | LOWT | low amplitude T-waves | other ST-T descriptive statements | Низкий T-волновой комплекс | |
| 22 | PAC | atrial premature complex | ectopic rhythm | Преждевременное предсердное сокращение | |
| 23 | AMI | anterior myocardial infarction | Myocardial Infarction | Острый инфаркт миокарда (передняя стенка) | в каноническом описании нет слова «острый» |

Машинная копия тех же полей: `ecg_engine/scp.py`.

Порядок ключей в `predictions` CLI совпадает с этой таблицей сверху вниз. Сортировки по убыванию score, как на HTML-странице `ecg_web_up`, в новом JSON нет: для сверки нужен стабильный индекс.
