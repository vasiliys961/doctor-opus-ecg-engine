"""Коды выхода модели и сверка с формулировками SCP из PTB-XL.

Индекс в этом списке — индекс выхода сети. Его нельзя менять.
canonical_description взят из scp_statements.csv PTB-XL 1.0.3.
legacy_display_name — русская подпись из ecg_web_up. Она сохранена как историческая
и не заменяет каноническое описание там, где формулировки расходятся.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SCPOutput:
    index: int
    model_code: str
    canonical_description: str
    statement_category: str
    legacy_display_name: str
    naming_note: str = ""


SCP_OUTPUTS: tuple[SCPOutput, ...] = (
    SCPOutput(0, "SR", "sinus rhythm", "Statements related to impulse formation (abnormalities)", "Синусовый ритм"),
    SCPOutput(1, "NORM", "normal ECG", "Normal/abnormal", "Нормальная ЭКГ"),
    SCPOutput(2, "ABQRS", "abnormal QRS", "Normal/abnormal", "Аберрантный QRS комплекс"),
    SCPOutput(3, "IMI", "inferior myocardial infarction", "Myocardial Infarction", "Инфаркт миокарда (нижняя стенка)"),
    SCPOutput(4, "ASMI", "anteroseptal myocardial infarction", "Myocardial Infarction", "Инфаркт миокарда (переднеперегородочная стенка)"),
    SCPOutput(5, "LVH", "left ventricular hypertrophy", "Ventricular Hypertrophy", "Гипертрофия левого желудочка"),
    SCPOutput(
        6,
        "NDT",
        "non-diagnostic T abnormalities",
        "other ST-T descriptive statements",
        "Неспецифические изменения ST-T",
        "Каноническое описание относится к зубцу T. Русская подпись добавляет ST.",
    ),
    SCPOutput(7, "LAFB", "left anterior fascicular block", "Intraventricular and intra-atrial Conduction disturbances", "Блокада передней ветви левой ножки пучка Гиса"),
    SCPOutput(8, "AFIB", "atrial fibrillation", "Statements related to impulse formation (abnormalities)", "Фибрилляция предсердий"),
    SCPOutput(9, "PVC", "ventricular premature complex", "Statements related to ectopic rhythm abnormalities", "Преждевременное желудочковое сокращение"),
    SCPOutput(10, "IRBBB", "incomplete right bundle branch block", "Intraventricular and intra-atrial Conduction disturbances", "Неполная блокада правой ножки пучка Гиса"),
    SCPOutput(
        11,
        "VCLVH",
        "voltage criteria (QRS) for left ventricular hypertrophy",
        "Ventricular Hypertrophy",
        "Гипертрофия желудочков или левого желудочка",
        "Канонически это вольтажные критерии QRS для ГЛЖ, а не общий диагноз гипертрофии.",
    ),
    SCPOutput(12, "STACH", "sinus tachycardia", "Statements related to impulse formation (abnormalities)", "Синусовая тахикардия"),
    SCPOutput(13, "IVCD", "non-specific intraventricular conduction disturbance (block)", "Intraventricular and intra-atrial Conduction disturbances", "Внутрижелудочковая блокада"),
    SCPOutput(
        14,
        "SARRH",
        "sinus arrhythmia",
        "Statements related to impulse formation (abnormalities)",
        "Синусовый ритм с аберрантным проведением",
        "Канонически sinus arrhythmia. Русская подпись в ecg_web_up описывает другой феномен.",
    ),
    SCPOutput(
        15,
        "ISCAL",
        "ischemic in anterolateral leads",
        "ischemic ST-T changes",
        "Ишемия миокарда (нижняя стенка)",
        "Канонически переднебоковые отведения. Нижняя стенка в SCP — это ISCI, не ISCAL.",
    ),
    SCPOutput(16, "SBRAD", "sinus bradycardia", "Statements related to impulse formation (abnormalities)", "Синусовая брадикардия"),
    SCPOutput(17, "QWAVE", "Q waves present", "Other QRS morphology descriptive statements", "Патологический Q-волновой комплекс"),
    SCPOutput(18, "CRBBB", "complete right bundle branch block", "Intraventricular and intra-atrial Conduction disturbances", "Полная блокада правой ножки пучка Гиса"),
    SCPOutput(19, "CLBBB", "complete left bundle branch block", "Intraventricular and intra-atrial Conduction disturbances", "Полная блокада левой ножки пучка Гиса"),
    SCPOutput(20, "ILMI", "inferolateral myocardial infarction", "Myocardial Infarction", "Инфаркт миокарда (нижнебоковая стенка)"),
    SCPOutput(21, "LOWT", "low amplitude T-waves", "other ST-T descriptive statements", "Низкий T-волновой комплекс"),
    SCPOutput(22, "PAC", "atrial premature complex", "Statements related to ectopic rhythm abnormalities", "Преждевременное предсердное сокращение"),
    SCPOutput(
        23,
        "AMI",
        "anterior myocardial infarction",
        "Myocardial Infarction",
        "Острый инфаркт миокарда (передняя стенка)",
        "Каноническое описание не содержит слова «острый».",
    ),
)

TOP_24_CODES: tuple[str, ...] = tuple(item.model_code for item in SCP_OUTPUTS)

if len(TOP_24_CODES) != 24 or len(set(TOP_24_CODES)) != 24:
    raise RuntimeError("Список SCP-выходов должен содержать 24 уникальных кода.")
