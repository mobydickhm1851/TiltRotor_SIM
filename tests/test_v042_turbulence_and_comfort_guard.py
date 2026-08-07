import numpy as np
from bokeh.document import Document
from bokeh.models import NumericInput, Select

from rotorpy_tiltrotor.commands import AutomaticMission, FlightPhase
from rotorpy_tiltrotor.enhanced_dashboard import _new_simulation
from rotorpy_tiltrotor.scenarios import WindScenarioConfig
from rotorpy_tiltrotor.v042_dashboard import (
    CS_AWO_LABEL,
    CS_AWO_HORIZONTAL_SCALE_M,
    CS_AWO_HORIZONTAL_SIGMA_RATIO,
    CS_AWO_MAX_FREQUENCY_HZ,
    CS_AWO_VERTICAL_SIGMA_RATIO,
    LowAltitudeCSAWOWindModel,
    build_dashboard,
    cs_awo_vertical_scale_m,
)


def test_cs_awo_vertical_turbulence_is_smooth_and_scaled_at_30m():
    reference_mean_wind = 5.0
    expected_sigma_w = CS_AWO_VERTICAL_SIGMA_RATIO * reference_mean_wind
    model = LowAltitudeCSAWOWindModel(WindScenarioConfig(
        enabled=True,
        mode="continuous_turbulence",
        disturbance_amplitude_mps=reference_mean_wind,
        reference_altitude_m=30.0,
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
    assert 0.45 * expected_sigma_w < measured_sigma < 1.55 * expected_sigma_w
    # Adjacent 100-Hz samples must be strongly correlated rather than resemble
    # the v0.4.1 short-correlation multi-axis noise.
    assert np.max(np.abs(np.diff(steady))) < 0.15
    assert np.corrcoef(steady[:-1], steady[1:])[0, 1] > 0.98

    assert np.isclose(CS_AWO_VERTICAL_SIGMA_RATIO, 0.09)
    assert np.isclose(cs_awo_vertical_scale_m(30.0), 15.0)
    # Keep the horizontal values explicit so future code does not accidentally
    # reuse them for the vertical transition benchmark.
    assert np.isclose(CS_AWO_HORIZONTAL_SIGMA_RATIO, 0.15)
    assert np.isclose(CS_AWO_HORIZONTAL_SCALE_M, 183.0)
    assert CS_AWO_MAX_FREQUENCY_HZ <= 1.0


def test_comfort_guard_keeps_nominal_takeoff_measured_jerk_near_target():
    sim = _new_simulation()
    sim.controller.max_accel = 0.50
    sim.controller.max_command_jerk_mps3 = 1.50
    sim.commander.vertical_takeoff(
        sim.t,
        30.0,
        float(sim.state["x"][0]),
        float(sim.state["x"][1]),
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


def test_default_cs_awo_turbulence_does_not_crash_forward_transition():
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
        reference_altitude_m=30.0,
        start_time_s=35.0,
        random_seed=21,
    ))

    min_forward_altitude = np.inf
    max_forward_body_rate_deg_s = 0.0
    saw_forward_flight = False
    for _ in range(int(120.0 / sim.dt)):
        mission.update(sim.t, sim.state)
        phase = sim.commander.setpoint.phase
        sample_wind = wind.sample(sim.t, sim.state)
        sim.step(sample_wind.vector_mps)

        assert np.all(np.isfinite(sim.state["x"]))
        assert np.all(np.isfinite(sim.state["v"]))
        assert np.all(np.isfinite(sim.state["q"]))

        if phase in (
            FlightPhase.TRANSITION_TO_CRUISE,
            FlightPhase.CRUISE,
        ):
            saw_forward_flight = True
            min_forward_altitude = min(
                min_forward_altitude,
                float(sim.state["x"][2]),
            )
            max_forward_body_rate_deg_s = max(
                max_forward_body_rate_deg_s,
                float(np.linalg.norm(sim.state["w"])) * 180.0 / np.pi,
            )
        elif saw_forward_flight and phase == FlightPhase.TRANSITION_TO_HOVER:
            break

    assert saw_forward_flight
    assert min_forward_altitude > 20.0
    assert max_forward_body_rate_deg_s < 60.0


def test_dashboard_labels_low_alt_vertical_turbulence_semantics():
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
        if item.title == (
            "Reference mean wind U [m/s] (vertical sigma_w = 0.09 U)"
        )
    )
    assert np.isclose(amplitude.value, 5.0)
