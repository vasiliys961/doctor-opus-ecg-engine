from __future__ import annotations

import numpy as np

from ecg_engine.delineation import measure_tracing, premature_counts, qrs_axis_degrees


def test_early_rr_is_a_premature_beat_and_a_wide_one_is_counted():
    peaks = np.array([0, 500, 1000, 1300, 1800], dtype=float)
    qrs = [80.0, 80.0, 80.0, 140.0, 80.0]
    premature, wide = premature_counts(peaks, qrs, 500)
    assert premature == 1
    assert wide == 1


def test_sokolow_adds_s_depth_in_v1_to_the_taller_lateral_r():
    from ecg_engine.delineation import cornell_voltage, gubner_index, lewis_index, sokolow_index

    assert sokolow_index(-1.2, 0.4, 0.9) == 2.1
    assert sokolow_index(0.2, 1.0, 1.0) is None
    assert cornell_voltage(0.8, -1.1) == 1.9
    assert cornell_voltage(0.8, 0.2) is None
    assert lewis_index(1.0, -0.2, 0.4, -0.9) == 1.3
    assert lewis_index(None, None, None, None) is None
    assert gubner_index(1.0, -0.5) == 1.5
    assert gubner_index(1.0, 0.2) is None


def test_qrs_t_angle_is_the_short_turn_between_axes():
    from ecg_engine.delineation import qrs_t_angle

    assert qrs_t_angle(10, 40) == 30
    assert qrs_t_angle(170, -170) == 20


def test_positive_lead_i_and_flat_avf_give_axis_near_zero():
    lead_i = np.zeros(100)
    lead_avf = np.zeros(100)
    lead_i[10:40] = 1.0
    lead_i[60:90] = 1.0
    axis = qrs_axis_degrees(lead_i, lead_avf, np.array([10.0, 60.0]), np.array([40.0, 90.0]))
    assert axis == 0.0


def test_flat_tracing_has_no_measurements():
    result = measure_tracing(np.zeros((12, 5000)), 500)
    assert result["available"] is False
    assert result["lines"] == []


def test_simulated_lead_ii_yields_rate_and_intervals():
    import neurokit2 as nk

    lead_ii = nk.ecg_simulate(duration=10, sampling_rate=500, heart_rate=60, random_state=1)
    leads = np.zeros((12, lead_ii.size))
    leads[0] = lead_ii
    leads[1] = lead_ii
    leads[5] = lead_ii * 0.5
    result = measure_tracing(leads, 500)
    assert result["available"] is True
    assert 50 <= result["heart_rate_bpm"] <= 70
    assert result["pq_ms"] > 0
    assert result["qrs_ms"] > 0
    assert result["qt_ms"] > result["qrs_ms"]
    names = [line["name"] for line in result["lines"]]
    assert names[:3] == ["Ритм", "ЧСС", "Электрическая ось сердца"]
    assert "Внеочередные комплексы" in names
    assert "10 комплексов" in result["description"] or "комплексов" in result["description"]
    assert [row["lead"] for row in result["amplitudes"]] == ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    assert result["amplitudes"][1]["r"] > 0
    assert result["amplitudes"][6]["r"] is None
    assert any(line["name"] == "Соколов–Лайон" for line in result["lines"])
    assert any(line["name"] == "Корнелл" for line in result["lines"])
    assert any(line["name"] == "Льюис" for line in result["lines"])
    assert any(line["name"] == "Губнер" for line in result["lines"])
    assert any(line["name"] == "Разброс QT" for line in result["lines"])
    assert [row["lead"] for row in result["leads"]] == ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    assert result["leads"][1]["qt_ms"] is not None
    assert "quality" in result["leads"][1]
