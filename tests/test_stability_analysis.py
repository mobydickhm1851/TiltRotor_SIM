from bokeh.document import Document
from bokeh.models import Select
import numpy as np

from rotorpy_tiltrotor.stability_analysis import analyze_all_loops
from rotorpy_tiltrotor.v046_dashboard import build_dashboard


def test_all_local_bode_loops_are_closed_loop_stable():
    results = analyze_all_loops()
    expected = {
        "hover_altitude",
        "cruise_airspeed",
        "pitch_hover",
        "pitch_cruise",
        "cruise_cross_track",
    }
    assert set(results) == expected
    for result in results.values():
        assert result.stable, result.definition.label
        assert np.all(np.real(result.closed_loop_poles) < 0.0)
        assert np.isfinite(result.phase_margin_deg)
        assert result.phase_margin_deg > 45.0, result.definition.label
        assert result.closed_loop_bandwidth_rad_s > 0.0
        assert result.closed_loop_peak_db < 3.1


def test_reported_phase_margins_stay_close_to_current_controller_values():
    results = analyze_all_loops()
    expected_pm = {
        "hover_altitude": 66.4,
        "cruise_airspeed": 84.2,
        "pitch_hover": 50.7,
        "pitch_cruise": 53.2,
        "cruise_cross_track": 56.2,
    }
    for key, expected in expected_pm.items():
        assert abs(results[key].phase_margin_deg - expected) < 1.5


def test_dashboard_contains_selectable_bode_analysis():
    doc = Document()
    build_dashboard(doc)
    selectors = [
        model for model in doc.select({"type": Select})
        if str(getattr(model, "title", "")) == "Local loop"
    ]
    assert len(selectors) == 1
    values = {
        option[0] if isinstance(option, (tuple, list)) else option
        for option in selectors[0].options
    }
    assert "hover_altitude" in values
    assert "cruise_cross_track" in values
