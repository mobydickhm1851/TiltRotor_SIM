import numpy as np
from bokeh.document import Document
from bokeh.models import NumericInput, Select

from rotorpy_tiltrotor.commands import AutomaticMission
from rotorpy_tiltrotor.enhanced_dashboard import _new_simulation
from rotorpy_tiltrotor.scenarios import WindScenarioConfig
from rotorpy_tiltrotor.v042_dashboard import (
    CS_AWO_LABEL,
    CS_AWO_SIGMA_RATIO,
    CS_AWO_TURBULENCE_SCALE_M,
    LowAltitudeCSAWOWindModel,
    build_dashboard,
)


def test_cs_awo_continuous_turbulence_is_vertical_smooth_and_scaled():
    reference_mean_wind = 5.0
    expected_sigma = CS_AWO_SIGMA_RATIO * reference_mean_wind
    model = LowAltitudeCSAWOWindModel(WindScenarioConfig(
        enabled=True,
        mode="continuous_turbulence",
        disturbance_amplitude_mps=reference_mean_wind,
        start_time_s=0.0,
        random_seed=13,
    ))
    state = {
        "x": np.array([0.0, 0.0, 30.0]),
        "v": np.array([15.0, 0.0, 0.0]),
    }
    values = []
    dt = 0.01
    for t in np.arange(0.0, 180.0, dt):
        values.append(model.sample(float(t), state).disturbance_mps.copy())
    values = np.asarray(values)

    assert np.max(np.abs(values[:, 0])) < 1e-10
    assert np.max(np.abs(values[:, 1])) < 1e-10
    steady = values[int(10.0 / dt):, 2]
    measured_sigma = np.std(steady)
    assert 0.45 * expected_sigma < measured_sigma < 1.55 * expected_sigma
    max_wind_accel = np.max(np.abs(np.diff(steady) / dt))
    assert max_wind_accel < 3.0
    assert np.isclose(CS_AWO_TURBULENCE_SCALE_M, 183.0)
    assert np.isclose(CS_AWO_SIGMA_RATIO, 0.15)


def test_comfort_guard_keeps_nominal_takeoff_measured_jerk_near_target():
    sim = _new_simulation()
    sim.controller.max_accel = 0.50
    sim.controller.max_command_jerk_mps3 = 1.50
    sim.commander.vertical_takeoff(
        sim.t,
        altitude_m=30.0,
        hold_x_m=0.0,
        hold_y_m=0.0,
    )

    peak_jerk = 0.0
    for _ in range(int(12.0 / sim.dt)):
        sample = sim.step(np.zeros(3))
        peak_jerk = max(
            peak_jerk,
            float(np.linalg.norm(sample.diagnostics["jerk_world"])),
        )

    assert peak_jerk <= 1.50 * 1.05
    assert sim.state["x"][2] > 1.0


def test_default_cs_awo_turbulence_does_not_crash_transition():
    sim = _new_simulation()
    mission = AutomaticMission(
        sim.commander,
        altitude_m=30.0,
        cruise_speed_mps=15.0,
        cruise_distance_m=180.0,
    )
    wind = LowAltitudeCSAWOWindModel(WindScenarioConfig(
        enabled=True,
        mode="continuous_turbulence",
        disturbance_amplitude_mps=5.0,
        start_time_s=35.0,
        random_seed=21,
    ))

    min_altitude_after_40 = 1e9
    max_body_rate_deg_s = 0.0
    for _ in range(int(115.0 / sim.dt)):
        mission.update(sim.t, sim.state)
        sample_wind = wind.sample(sim.t, sim.state)
        sim.step(sample_wind.vector_mps)
        assert np.all(np.isfinite(sim.state["x"]))
        assert np.all(np.isfinite(sim.state["v"]))
        assert np.all(np.isfinite(sim.state["q"]))
        if sim.t > 40.0:
            min_altitude_after_40 = min(
                min_altitude_after_40,
                float(sim.state["x"][2]),
            )
            max_body_rate_deg_s = max(
                max_body_rate_deg_s,
                float(np.linalg.norm(sim.state["w"])) * 180.0 / np.pi,
            )

    assert min_altitude_after_40 > 18.0
    assert max_body_rate_deg_s < 90.0


def test_dashboard_labels_low_alt_turbulence_as_reference_mean_wind():
    doc = Document()
    build_dashboard(doc)
    selector = next(
        item for item in doc.select({"type": Select})
        if item.title == "Wind scenario"
    )
    assert CS_AWO_LABEL in selector.options
    selector.value = CS_AWO_LABEL

    inputs = list(doc.select({"type": NumericInput}))
    amplitude = next(
        item for item in inputs
        if item.title == "Reference mean wind U [m/s] (sigma = 0.15 U)"
    )
    assert np.isclose(amplitude.value, 5.0)
