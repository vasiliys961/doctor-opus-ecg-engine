"""Разметка зубцов цифровой кривой рядом с оценками ECGFounder.

NeuroKit2 размечает комплексы по выступам. Ритм и интервалы считаются
по отведению II, длительность QRS — по верхней четверти 12 отведений.
Ось берётся по площади комплекса в I и aVF. Это не измеритель аппарата.
"""

from __future__ import annotations

import warnings

import numpy as np

from ecg_engine.ecgfounder import LEADS

II = LEADS.index("II")
I = LEADS.index("I")
III = LEADS.index("III")
AVL = LEADS.index("aVL")
AVF = LEADS.index("aVF")
V1 = LEADS.index("V1")
V3 = LEADS.index("V3")
V5 = LEADS.index("V5")
V6 = LEADS.index("V6")
PREMATURE_FRACTION = 0.8
WIDE_QRS_MS = 120.0
_PROMINENCE = {
    "max_qrs_interval": 220,
    "max_r_basepoint_interval": 200,
    "max_r_rise_time": 160,
    "max_p_basepoint_interval": 140,
    "max_pr_interval": 360,
    "max_t_basepoint_interval": 260,
    "typical_st_segment": 120,
}


def measure_tracing(leads: np.ndarray, sampling_rate: float) -> dict[str, object]:
    try:
        return _measure(np.asarray(leads, dtype=np.float64), float(sampling_rate))
    except Exception:
        return _unavailable("Разметка зубцов не посчиталась.")


def premature_counts(r_peaks: np.ndarray, qrs_ms: list[float | None], sampling_rate: float) -> tuple[int, int]:
    peaks = np.asarray(r_peaks, dtype=np.float64)
    peaks = peaks[np.isfinite(peaks)]
    if peaks.size < 3 or sampling_rate <= 0:
        return 0, 0
    gaps = np.diff(peaks)
    median = float(np.median(gaps))
    if median <= 0:
        return 0, 0
    limit = PREMATURE_FRACTION * median
    premature = 0
    wide = 0
    for index, gap in enumerate(gaps):
        if float(gap) >= limit:
            continue
        previous_ok = index == 0 or float(gaps[index - 1]) >= limit
        following_ok = index + 1 == len(gaps) or float(gaps[index + 1]) >= limit
        if not (previous_ok and following_ok):
            continue
        premature += 1
        beat = index + 1
        duration = qrs_ms[beat] if beat < len(qrs_ms) else None
        if duration is not None and duration > WIDE_QRS_MS:
            wide += 1
    return premature, wide


def qrs_axis_degrees(lead_i: np.ndarray, lead_avf: np.ndarray, onsets: np.ndarray, offsets: np.ndarray) -> float | None:
    areas_i: list[float] = []
    areas_avf: list[float] = []
    limit = min(len(onsets), len(offsets), lead_i.size, lead_avf.size)
    for index in range(limit):
        start, end = onsets[index], offsets[index]
        if not (np.isfinite(start) and np.isfinite(end)):
            continue
        left, right = int(start), int(end)
        if right - left < 2 or left < 0 or right > lead_i.size:
            continue
        areas_i.append(float(np.trapz(lead_i[left:right])))
        areas_avf.append(float(np.trapz(lead_avf[left:right])))
    if len(areas_i) < 2:
        return None
    net_i = float(np.median(areas_i))
    net_avf = float(np.median(areas_avf))
    if abs(net_i) + abs(net_avf) < 1e-8:
        return None
    return float(np.degrees(np.arctan2(net_avf, net_i)))


def _measure(leads: np.ndarray, sampling_rate: float) -> dict[str, object]:
    if leads.ndim != 2 or leads.shape[0] != 12:
        return _unavailable("Для разметки нужны 12 отведений.")
    if not np.isfinite(sampling_rate) or sampling_rate <= 0:
        return _unavailable("Частота дискретизации должна быть положительным числом.")
    rate = int(round(sampling_rate))
    lead_ii = leads[II]
    if lead_ii.size < rate * 2 or float(np.std(lead_ii)) < 1e-8:
        return _unavailable("На отведении II не удалось выделить комплексы QRS.")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        traced = [_delineate_lead(leads[index], rate) for index in range(12)]
    if traced[II] is None:
        return _unavailable("На отведении II не удалось выделить комплексы QRS.")
    r_peaks, waves = traced[II]
    qrs_ms = _quartile_qrs(traced, rate)
    amplitudes = [_lead_heights(LEADS[index], leads[index], traced[index], rate) for index in range(12)]

    def wave(name: str) -> np.ndarray:
        return np.asarray(waves.get(name, []), dtype=np.float64)

    qrs_onsets = wave("ECG_R_Onsets")
    qrs_offsets = wave("ECG_R_Offsets")
    p_onsets = wave("ECG_P_Onsets")
    p_offsets = wave("ECG_P_Offsets")
    t_onsets = wave("ECG_T_Onsets")
    t_offsets = wave("ECG_T_Offsets")
    t_peaks = wave("ECG_T_Peaks")
    qrs_per_beat = _per_beat_ms(qrs_onsets, qrs_offsets, rate)
    premature, wide = premature_counts(r_peaks, qrs_per_beat, rate)
    rr_ms = _median_delta_ms(r_peaks[:-1], r_peaks[1:], rate, 200, 3000)
    heart_rate = None if rr_ms is None or rr_ms <= 0 else 60000.0 / rr_ms
    pq_values = _plausible_values(p_onsets, qrs_onsets, rate, 60, 400)
    pq_ms = _median_present(pq_values)
    p_ms = _median_delta_ms(p_onsets, p_offsets, rate, 40, 200)
    qt_ms = _median_delta_ms(qrs_onsets, t_offsets, rate, 200, 700)
    t_ms = _median_delta_ms(t_onsets, t_offsets, rate, 40, 400)
    qtc_ms = _bazett(qt_ms, rr_ms)
    st_point_ms = 80 if heart_rate is not None and heart_rate < 100 else 60
    st_value = _st_displacement(lead_ii, p_offsets, qrs_onsets, qrs_offsets, rate, st_point_ms)
    t_polarity = _t_polarity(lead_ii, t_peaks, p_offsets, qrs_onsets, rate)
    axis = qrs_axis_degrees(leads[I], leads[AVF], qrs_onsets, qrs_offsets)
    rr_regular = _rr_regular(r_peaks)
    p_state = _p_state(pq_values, int(r_peaks.size))
    if p_state == "stable" and not rr_regular:
        p_state = "unstable"

    lines = [
        {"name": "Ритм", "value": _rhythm_text(p_state, premature, wide, rr_regular)},
        {"name": "ЧСС", "value": _fmt_bpm(heart_rate)},
        {"name": "Электрическая ось сердца", "value": _fmt_axis(axis)},
        {"name": "Зубец P", "value": _fmt_ms(p_ms)},
        {"name": "Интервал PQ", "value": _fmt_ms(pq_ms)},
        {"name": "Комплекс QRS", "value": _fmt_ms(qrs_ms)},
        {"name": "Сегмент ST", "value": _fmt_st(st_value, st_point_ms)},
        {"name": "Зубец T", "value": _fmt_t(t_polarity, t_ms)},
        {"name": "Интервал QT", "value": _fmt_qt(qt_ms, qtc_ms)},
        {"name": "Внеочередные комплексы", "value": _fmt_premature(premature, wide)},
    ]
    rr_min_ms, rr_max_ms = _rr_edges(r_peaks, rate)
    t_axis = qrs_axis_degrees(leads[I], leads[AVF], t_onsets, t_offsets)
    qt_by_lead = [_lead_qt_ms(item, rate) for item in traced]
    lead_rows = [
        _lead_row(LEADS[index], leads[index], traced[index], amplitudes[index], rate, st_point_ms)
        for index in range(12)
    ]
    extra = [
        {"name": "RR, короткий и длинный", "value": _fmt_rr_edges(rr_min_ms, rr_max_ms)},
        {"name": "Зубец P во II", "value": _fmt_height(amplitudes[II]["p"])},
        {"name": "Отрицательная часть P в V1", "value": _fmt_p_negative(_p_negative(leads[V1], traced[V1], rate))},
        {"name": "R/S в V1", "value": _fmt_ratio(_rs_ratio(amplitudes[V1]["r"], amplitudes[V1]["s"]))},
        {"name": "R/S в V5", "value": _fmt_ratio(_rs_ratio(amplitudes[V5]["r"], amplitudes[V5]["s"]))},
        {"name": "Соколов–Лайон", "value": _fmt_height(sokolow_index(amplitudes[V1]["s"], amplitudes[V5]["r"], amplitudes[V6]["r"]))},
        {"name": "Корнелл", "value": _fmt_height(cornell_voltage(amplitudes[AVL]["r"], amplitudes[V3]["s"]))},
        {"name": "Льюис", "value": _fmt_height(lewis_index(amplitudes[I]["r"], amplitudes[I]["s"], amplitudes[III]["r"], amplitudes[III]["s"]))},
        {"name": "Губнер", "value": _fmt_height(gubner_index(amplitudes[I]["r"], amplitudes[III]["s"]))},
        {"name": "Время до вершины R", "value": _fmt_intrinsicoid(lead_rows)},
        {"name": "Ось T", "value": _fmt_axis(t_axis)},
        {"name": "Угол QRS–T", "value": _fmt_angle(qrs_t_angle(axis, t_axis))},
        {"name": "Разброс QT", "value": _fmt_ms(qt_dispersion(qt_by_lead))},
    ]
    description = _description(int(r_peaks.size), lines)
    lines = lines + extra
    return {
        "available": True,
        "method": "NeuroKit2, разметка по выступам",
        "lead": "II",
        "beats": int(r_peaks.size),
        "heart_rate_bpm": None if heart_rate is None else round(heart_rate, 1),
        "rr_ms": None if rr_ms is None else round(rr_ms, 1),
        "premature_beats": premature,
        "wide_premature_beats": wide,
        "axis_degrees": None if axis is None else round(axis, 1),
        "p_ms": None if p_ms is None else round(p_ms, 1),
        "pq_ms": None if pq_ms is None else round(pq_ms, 1),
        "qrs_ms": None if qrs_ms is None else round(qrs_ms, 1),
        "qt_ms": None if qt_ms is None else round(qt_ms, 1),
        "qtc_bazett_ms": None if qtc_ms is None else round(qtc_ms, 1),
        "st_displacement": None if st_value is None else round(st_value, 4),
        "st_point_ms": st_point_ms,
        "lines": lines,
        "description": description,
        "amplitudes": amplitudes,
        "leads": lead_rows,
        "rr_min_ms": None if rr_min_ms is None else round(rr_min_ms, 1),
        "rr_max_ms": None if rr_max_ms is None else round(rr_max_ms, 1),
        "t_axis_degrees": None if t_axis is None else round(t_axis, 1),
        "qrs_t_angle_degrees": None if qrs_t_angle(axis, t_axis) is None else round(qrs_t_angle(axis, t_axis), 1),
        "qt_dispersion_ms": None if qt_dispersion(qt_by_lead) is None else round(qt_dispersion(qt_by_lead), 1),
        "note": (
            "Ритм и интервалы сняты с отведения II. Длительность QRS — верхняя четверть по 12 отведениям. "
            "Оси QRS и T — по площади зубца в I и aVF. Смещение ST, высота, Соколов–Лайон, Корнелл, Льюис и Губнер — в единицах CSV, не в миллиметрах. "
            "Ширина Q — от начала QRS до возврата к изолинии. Качество — сходство комплексов на отведении, от 0 до 1. "
            "Внеочередной комплекс — изолированный RR короче 80% медианы. "
            "Это не измерение сертифицированного аппарата."
        ),
    }


def _unavailable(note: str) -> dict[str, object]:
    return {"available": False, "lines": [], "description": "", "amplitudes": [], "leads": [], "note": note}


def _delineate_lead(lead: np.ndarray, sampling_rate: int):
    import neurokit2 as nk

    if lead.size < sampling_rate * 2 or float(np.std(lead)) < 1e-8:
        return None
    cleaned = nk.ecg_clean(lead, sampling_rate=sampling_rate)
    _, peaks = nk.ecg_peaks(cleaned, sampling_rate=sampling_rate, correct_artifacts=False)
    r_peaks = np.asarray(peaks.get("ECG_R_Peaks", []), dtype=np.float64)
    if r_peaks.size < 3:
        return None
    _, waves = nk.ecg_delineate(
        cleaned,
        peaks,
        sampling_rate=sampling_rate,
        method="prominence",
        **_PROMINENCE,
    )
    return r_peaks, waves


def _quartile_qrs(traced: list, sampling_rate: int) -> float | None:
    widths: list[float] = []
    for item in traced:
        if item is None:
            continue
        _, waves = item
        width = _median_delta_ms(
            np.asarray(waves.get("ECG_R_Onsets", []), dtype=np.float64),
            np.asarray(waves.get("ECG_R_Offsets", []), dtype=np.float64),
            sampling_rate,
            40,
            260,
        )
        if width is not None:
            widths.append(width)
    if not widths:
        return None
    return float(np.percentile(widths, 75))


def _lead_heights(name: str, lead: np.ndarray, delineated, sampling_rate: int) -> dict[str, object]:
    empty = {"lead": name, "p": None, "q": None, "r": None, "s": None, "t": None}
    if delineated is None:
        return empty
    r_peaks, waves = delineated
    p_offsets = np.asarray(waves.get("ECG_P_Offsets", []), dtype=np.float64)
    qrs_onsets = np.asarray(waves.get("ECG_R_Onsets", []), dtype=np.float64)
    series = {
        "p": np.asarray(waves.get("ECG_P_Peaks", []), dtype=np.float64),
        "q": np.asarray(waves.get("ECG_Q_Peaks", []), dtype=np.float64),
        "r": np.asarray(r_peaks, dtype=np.float64),
        "s": np.asarray(waves.get("ECG_S_Peaks", []), dtype=np.float64),
        "t": np.asarray(waves.get("ECG_T_Peaks", []), dtype=np.float64),
    }
    collected = {key: [] for key in series}
    count = max((len(values) for values in series.values()), default=0)
    for index in range(count):
        onset = qrs_onsets[index] if index < len(qrs_onsets) else np.nan
        if not np.isfinite(onset) and index < len(series["r"]) and np.isfinite(series["r"][index]):
            onset = float(series["r"][index]) - 0.04 * sampling_rate
        if not np.isfinite(onset):
            continue
        baseline = _baseline(lead, p_offsets, index, int(onset), sampling_rate)
        if baseline is None:
            continue
        for key, values in series.items():
            if index >= len(values) or not np.isfinite(values[index]):
                continue
            sample = int(values[index])
            if 0 <= sample < lead.size:
                collected[key].append(float(lead[sample] - baseline))
    row = dict(empty)
    for key, values in collected.items():
        if values:
            row[key] = round(float(np.median(values)), 3)
    return row


def sokolow_index(s_v1: float | None, r_v5: float | None, r_v6: float | None) -> float | None:
    if s_v1 is None or s_v1 >= 0:
        return None
    lateral = [value for value in (r_v5, r_v6) if value is not None and value > 0]
    if not lateral:
        return None
    return round(abs(s_v1) + max(lateral), 3)


def cornell_voltage(r_avl: float | None, s_v3: float | None) -> float | None:
    if r_avl is None or s_v3 is None or r_avl <= 0 or s_v3 >= 0:
        return None
    return round(r_avl + abs(s_v3), 3)


def lewis_index(
    r_i: float | None,
    s_i: float | None,
    r_iii: float | None,
    s_iii: float | None,
) -> float | None:
    if all(value is None for value in (r_i, s_i, r_iii, s_iii)):
        return None
    r_lead_i = r_i if r_i is not None and r_i > 0 else 0.0
    s_lead_i = abs(s_i) if s_i is not None and s_i < 0 else 0.0
    r_lead_iii = r_iii if r_iii is not None and r_iii > 0 else 0.0
    s_lead_iii = abs(s_iii) if s_iii is not None and s_iii < 0 else 0.0
    return round((r_lead_i + s_lead_iii) - (s_lead_i + r_lead_iii), 3)


def gubner_index(r_i: float | None, s_iii: float | None) -> float | None:
    if r_i is None or s_iii is None or r_i <= 0 or s_iii >= 0:
        return None
    return round(r_i + abs(s_iii), 3)


def qrs_t_angle(qrs_degrees: float | None, t_degrees: float | None) -> float | None:
    if qrs_degrees is None or t_degrees is None:
        return None
    difference = abs(qrs_degrees - t_degrees) % 360
    if difference > 180:
        difference = 360 - difference
    return float(difference)


def qt_dispersion(qt_by_lead: list[float | None]) -> float | None:
    present = [value for value in qt_by_lead if value is not None]
    if len(present) < 2:
        return None
    return float(max(present) - min(present))


def _rs_ratio(r_value: float | None, s_value: float | None) -> float | None:
    if r_value is None or s_value is None or abs(s_value) < 1e-6:
        return None
    return round(abs(r_value) / abs(s_value), 2)


def _rr_edges(r_peaks: np.ndarray, sampling_rate: int) -> tuple[float | None, float | None]:
    gaps = np.diff(np.asarray(r_peaks, dtype=np.float64))
    intervals = [float(gap) / sampling_rate * 1000.0 for gap in gaps if np.isfinite(gap)]
    plausible = [value for value in intervals if 200 <= value <= 3000]
    if not plausible:
        return None, None
    return min(plausible), max(plausible)


def _lead_qt_ms(delineated, sampling_rate: int) -> float | None:
    if delineated is None:
        return None
    _, waves = delineated
    return _median_delta_ms(
        np.asarray(waves.get("ECG_R_Onsets", []), dtype=np.float64),
        np.asarray(waves.get("ECG_T_Offsets", []), dtype=np.float64),
        sampling_rate,
        200,
        700,
    )


def _intrinsicoid_ms(delineated, sampling_rate: int) -> float | None:
    if delineated is None:
        return None
    r_peaks, waves = delineated
    return _median_delta_ms(
        np.asarray(waves.get("ECG_R_Onsets", []), dtype=np.float64),
        np.asarray(r_peaks, dtype=np.float64),
        sampling_rate,
        10,
        160,
    )


def _q_duration_ms(lead: np.ndarray, delineated, sampling_rate: int, q_amplitude: float | None) -> float | None:
    if delineated is None or q_amplitude is None or q_amplitude >= 0:
        return None
    r_peaks, waves = delineated
    onsets = np.asarray(waves.get("ECG_R_Onsets", []), dtype=np.float64)
    q_peaks = np.asarray(waves.get("ECG_Q_Peaks", []), dtype=np.float64)
    p_offsets = np.asarray(waves.get("ECG_P_Offsets", []), dtype=np.float64)
    durations: list[float] = []
    count = min(len(onsets), len(q_peaks), len(r_peaks))
    for index in range(count):
        onset, q_peak, r_peak = onsets[index], q_peaks[index], r_peaks[index]
        if not (np.isfinite(onset) and np.isfinite(q_peak) and np.isfinite(r_peak)):
            continue
        nadir, stop = int(q_peak), int(r_peak)
        anchor = int(onset) if np.isfinite(onset) else nadir
        if not (0 <= nadir < stop <= lead.size):
            continue
        baseline = _baseline(lead, p_offsets, index, min(anchor, nadir), sampling_rate)
        if baseline is None or lead[nadir] >= baseline:
            continue
        floor = max(0, nadir - int(0.04 * sampling_rate))
        start = floor
        for sample in range(nadir, floor, -1):
            if lead[sample] >= baseline:
                start = sample
                break
        end = stop
        for sample in range(nadir, stop):
            if lead[sample] >= baseline:
                end = sample
                break
        duration = (end - start) / sampling_rate * 1000.0
        if 10 <= duration <= 120:
            durations.append(duration)
    return _median_present(durations)


def _p_negative(lead: np.ndarray, delineated, sampling_rate: int) -> float | None:
    if delineated is None:
        return None
    _, waves = delineated
    onsets = np.asarray(waves.get("ECG_P_Onsets", []), dtype=np.float64)
    offsets = np.asarray(waves.get("ECG_P_Offsets", []), dtype=np.float64)
    qrs_onsets = np.asarray(waves.get("ECG_R_Onsets", []), dtype=np.float64)
    troughs: list[float] = []
    count = min(len(onsets), len(offsets), len(qrs_onsets))
    for index in range(count):
        start, end, qrs_onset = onsets[index], offsets[index], qrs_onsets[index]
        if not (np.isfinite(start) and np.isfinite(end) and np.isfinite(qrs_onset)):
            continue
        left, right = int(start), int(end)
        if right - left < 2 or left < 0 or right > lead.size:
            continue
        baseline = _baseline(lead, offsets, index, int(qrs_onset), sampling_rate)
        if baseline is None:
            continue
        troughs.append(float(np.min(lead[left:right]) - baseline))
    if not troughs:
        return None
    return round(float(np.median(troughs)), 3)


def _lead_quality(lead: np.ndarray, delineated, sampling_rate: int) -> float | None:
    if delineated is None or float(np.std(lead)) < 1e-8:
        return None
    import neurokit2 as nk

    r_peaks, _waves = delineated
    try:
        cleaned = nk.ecg_clean(lead, sampling_rate=sampling_rate)
        quality = nk.ecg_quality(
            cleaned,
            rpeaks=np.asarray(r_peaks, dtype=int),
            sampling_rate=sampling_rate,
            method="templatematch",
        )
        value = float(np.nanmedian(np.asarray(quality, dtype=np.float64)))
    except Exception:
        return None
    if not np.isfinite(value):
        return None
    return round(value, 2)


def _lead_row(name: str, lead: np.ndarray, delineated, amplitude: dict[str, object], sampling_rate: int, st_point_ms: int) -> dict[str, object]:
    st_value = None
    if delineated is not None:
        _r_peaks, waves = delineated
        st_value = _st_displacement(
            lead,
            np.asarray(waves.get("ECG_P_Offsets", []), dtype=np.float64),
            np.asarray(waves.get("ECG_R_Onsets", []), dtype=np.float64),
            np.asarray(waves.get("ECG_R_Offsets", []), dtype=np.float64),
            sampling_rate,
            st_point_ms,
        )
    q_amplitude = amplitude.get("q")
    q_ms = _q_duration_ms(lead, delineated, sampling_rate, q_amplitude if isinstance(q_amplitude, (int, float)) else None)
    qt_ms = _lead_qt_ms(delineated, sampling_rate)
    intrinsicoid = _intrinsicoid_ms(delineated, sampling_rate)
    return {
        "lead": name,
        "st": None if st_value is None else round(float(st_value), 3),
        "q_ms": None if q_ms is None else round(q_ms, 1),
        "qt_ms": None if qt_ms is None else round(qt_ms, 1),
        "intrinsicoid_ms": None if intrinsicoid is None else round(intrinsicoid, 1),
        "quality": _lead_quality(lead, delineated, sampling_rate),
    }


def _description(beats: int, lines: list[dict[str, str]]) -> str:
    values = {line["name"]: line["value"] for line in lines}
    return " ".join(
        [
            f"На отведении II размечено {beats} комплексов.",
            f"Ритм: {values['Ритм']}.",
            f"ЧСС {values['ЧСС']}, ось QRS {values['Электрическая ось сердца']}.",
            f"Зубец P {values['Зубец P']}, интервал PQ {values['Интервал PQ']}, комплекс QRS {values['Комплекс QRS']}.",
            f"Сегмент ST {values['Сегмент ST']}.",
            f"Зубец T {values['Зубец T']}. Интервал QT {values['Интервал QT']}.",
            f"Внеочередные комплексы: {values['Внеочередные комплексы']}.",
            "Высота зубца — медиана вершины минус участок PQ, в единицах CSV, отдельно по каждому отведению.",
            "Дальше в таблице RR по краям, вольтаж, ось T, разброс QT, а по отведениям ST, ширина Q и качество сигнала.",
        ]
    )


def _median_delta_ms(
    start: np.ndarray,
    end: np.ndarray,
    sampling_rate: int,
    low: float | None = None,
    high: float | None = None,
) -> float | None:
    return _median_present(_plausible_values(start, end, sampling_rate, low, high))


def _plausible_values(
    start: np.ndarray,
    end: np.ndarray,
    sampling_rate: int,
    low: float | None,
    high: float | None,
) -> list[float]:
    values: list[float] = []
    for value in _per_beat_ms(start, end, sampling_rate):
        if value is None:
            continue
        if low is not None and value < low:
            continue
        if high is not None and value > high:
            continue
        values.append(value)
    return values


def _per_beat_ms(start: np.ndarray, end: np.ndarray, sampling_rate: int) -> list[float | None]:
    count = min(len(start), len(end))
    values: list[float | None] = []
    for index in range(count):
        left, right = start[index], end[index]
        if not (np.isfinite(left) and np.isfinite(right)) or right <= left:
            values.append(None)
            continue
        values.append(float(right - left) / sampling_rate * 1000.0)
    return values


def _median_present(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    if not present:
        return None
    return float(np.median(present))


def _bazett(qt_ms: float | None, rr_ms: float | None) -> float | None:
    if qt_ms is None or rr_ms is None or rr_ms <= 0:
        return None
    return qt_ms / np.sqrt(rr_ms / 1000.0)


def _st_displacement(
    lead: np.ndarray,
    p_offsets: np.ndarray,
    qrs_onsets: np.ndarray,
    qrs_offsets: np.ndarray,
    sampling_rate: int,
    point_ms: int,
) -> float | None:
    shift = int(round(point_ms / 1000.0 * sampling_rate))
    values: list[float] = []
    count = min(len(qrs_onsets), len(qrs_offsets))
    for index in range(count):
        onset, offset = qrs_onsets[index], qrs_offsets[index]
        if not (np.isfinite(onset) and np.isfinite(offset)):
            continue
        point = int(offset) + shift
        if point < 0 or point >= lead.size:
            continue
        baseline = _baseline(lead, p_offsets, index, int(onset), sampling_rate)
        if baseline is None:
            continue
        values.append(float(lead[point] - baseline))
    if not values:
        return None
    return float(np.median(values))


def _baseline(lead: np.ndarray, p_offsets: np.ndarray, index: int, onset: int, sampling_rate: int) -> float | None:
    left = onset - int(round(0.04 * sampling_rate))
    if index < len(p_offsets) and np.isfinite(p_offsets[index]):
        candidate = int(p_offsets[index])
        if 0 <= candidate < onset:
            left = candidate
    left = max(0, left)
    if onset - left < 2 or onset > lead.size:
        return None
    return float(np.median(lead[left:onset]))


def _t_polarity(
    lead: np.ndarray,
    t_peaks: np.ndarray,
    p_offsets: np.ndarray,
    qrs_onsets: np.ndarray,
    sampling_rate: int,
) -> str | None:
    values: list[float] = []
    count = min(len(t_peaks), len(qrs_onsets))
    for index in range(count):
        peak = t_peaks[index]
        onset = qrs_onsets[index]
        if not (np.isfinite(peak) and np.isfinite(onset)):
            continue
        sample = int(peak)
        if sample < 0 or sample >= lead.size:
            continue
        baseline = _baseline(lead, p_offsets, index, int(onset), sampling_rate)
        if baseline is None:
            continue
        values.append(float(lead[sample] - baseline))
    if not values:
        return None
    amplitude = float(np.median(values))
    scale = float(np.std(lead))
    if abs(amplitude) < max(1e-6, 0.05 * scale):
        return "около изолинии"
    if amplitude > 0:
        return "положительный"
    return "отрицательный"


def _p_state(pq_values: list[float], beats: int) -> str:
    if beats <= 0 or len(pq_values) < max(2, int(0.5 * beats)):
        return "absent"
    if float(np.std(pq_values)) <= 40:
        return "stable"
    return "unstable"


def _rr_regular(r_peaks: np.ndarray) -> bool:
    gaps = np.diff(r_peaks[np.isfinite(r_peaks)])
    if gaps.size < 2 or float(np.mean(gaps)) <= 0:
        return False
    return float(np.std(gaps) / np.mean(gaps)) < 0.12


def _rhythm_text(p_state: str, premature: int, wide: int, regular: bool) -> str:
    if p_state == "stable":
        parts = ["зубец P перед QRS"]
    elif p_state == "unstable":
        parts = ["зубец P перед QRS не устойчив"]
    else:
        parts = ["зубец P перед QRS не выделен"]
    if premature:
        parts.append(f"внеочередных комплексов {premature}")
        if wide:
            parts.append(f"из них с QRS шире 120 мс: {wide}")
    elif regular:
        parts.append("интервалы RR близки")
    else:
        parts.append("интервалы RR различаются")
    return ", ".join(parts)


def _fmt_height(value: float | None) -> str:
    if value is None:
        return "не вычислено"
    return f"{value:+.3f} ед. сигнала"


def _fmt_rr_edges(short_ms: float | None, long_ms: float | None) -> str:
    if short_ms is None or long_ms is None:
        return "не вычислено"
    return f"{int(round(short_ms))}–{int(round(long_ms))} мс"


def _fmt_p_negative(value: float | None) -> str:
    if value is None:
        return "не вычислено"
    if value >= -0.005:
        return "отрицательной фазы нет"
    return f"{value:+.3f} ед. сигнала"


def _fmt_ratio(value: float | None) -> str:
    if value is None:
        return "не вычислено"
    return f"{value:.2f}"


def _fmt_intrinsicoid(rows: list[dict[str, object]]) -> str:
    by_lead = {row["lead"]: row.get("intrinsicoid_ms") for row in rows}
    parts = []
    for name in ("V1", "V5", "V6"):
        value = by_lead.get(name)
        parts.append(f"{name} {_fmt_ms(value if isinstance(value, (int, float)) else None)}")
    return ", ".join(parts)


def _fmt_angle(value: float | None) -> str:
    if value is None:
        return "не вычислен"
    return f"{int(round(value))}°"


def _fmt_ms(value: float | None) -> str:
    if value is None:
        return "не выделено"
    return f"{int(round(value))} мс"


def _fmt_bpm(value: float | None) -> str:
    if value is None:
        return "не выделено"
    return f"{int(round(value))} уд/мин"


def _fmt_axis(value: float | None) -> str:
    if value is None:
        return "не вычислена"
    return f"{int(round(value))}°"


def _fmt_st(value: float | None, point_ms: int) -> str:
    if value is None:
        return "не выделено"
    return f"{value:+.3f} ед. сигнала, точка J+{point_ms} мс, отведение II"


def _fmt_t(polarity: str | None, duration: float | None) -> str:
    if polarity is None and duration is None:
        return "не выделен"
    parts = []
    if polarity:
        parts.append(polarity)
    if duration is not None:
        parts.append(f"{int(round(duration))} мс")
    return ", ".join(parts)


def _fmt_qt(qt_ms: float | None, qtc_ms: float | None) -> str:
    text = _fmt_ms(qt_ms)
    if qtc_ms is None:
        return text
    return f"{text}, QTc Bazett {int(round(qtc_ms))} мс"


def _fmt_premature(premature: int, wide: int) -> str:
    if premature == 0:
        return "нет"
    if wide:
        return f"{premature}, из них с QRS шире 120 мс: {wide}"
    return str(premature)
