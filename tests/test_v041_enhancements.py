import numpy as np
from bokeh.document import Document
from bokeh.models import ColumnDataSource, NumericInput, Select, Toggle

from rotorpy_tiltrotor.enhanced_dashboard import (
    ComfortAwareController,
    ComfortAwareSimulation,
    build_dashboard,
)
from rotorpy_tiltrotor.commands import CommandManager
from rotorpy_tiltrotor.scenarios import (
    ComfortLimits,
    PassengerComfortMonitor,
    UrbanWindModel,
    WindScenarioConfig,
)
from rotorpy_tiltrotor.vehicle import Tiltrotor


def test_discrete_gust_delay_is_relative_to_enable_time():
    model = UrbanWindModel(WindScenarioConfig(
        enabled=True,
        mode="discrete_gust",
        disturbance_amplitude_mps=5.0,
        disturbance_direction_deg=90.0,
        start_time_s=2.0,
        duration_s=4.0,
    ))
    state = {"x": np.array([0.0, 0.0, 30.0])}

    just_enabled = model.sample(100.0, state)
    before = model.sample(101.0, state)
    middle = model.sample(104.0, state)
    after = model.sample(107.0, state)

    assert not just_enabled.active
    assert not before.active
    assert middle.active
    assert middle.disturbance_mps[1] > 4.9
    assert not after.active


def test_repeated_gusts_produce_multiple_distinct_events():
    model = UrbanWindModel(WindScenarioConfig(
        enabled=True,
        mode="repeated_gusts",
        disturbance_amplitude_mps=5.0,
        disturbance_direction_deg=90.0,
        start_time_s=0.5,
        duration_s=1.5,
        wake_frequency_hz=0.25,
    ))
    state = {"x": np.array([0.0, 0.0, 30.0])}
    active_indices = set()
    values = []
    for t in np.arange(0.0, 18.0, 0.05):
        sample = model.sample(float(t), state)
        values.append(sample.disturbance_mps.copy())
        if sample.active:
            active_indices.add(int(sample.gust_event_index))

    assert len(active_indices) >= 3
    magnitudes = np.linalg.norm(np.asarray(values), axis=1)
    assert magnitudes.max() > 2.0
    assert np.count_nonzero(magnitudes < 1e-6) > 20


def test_continuous_turbulence_is_continuously_variable_and_repeatable():
    cfg = WindScenarioConfig(
        enabled=True,
        mode="continuous_turbulence",
        disturbance_amplitude_mps=3.0,
        start_time_s=0.0,
        wake_frequency_hz=0.4,
        random_seed=11,
    )
    state = {"x": np.array([0.0, 0.0, 30.0])}
    a = UrbanWindModel(cfg)
    b = UrbanWindModel(cfg)
    seq_a = []
    seq_b = []
    for t in np.arange(0.0, 2.0, 0.01):
        seq_a.append(a.sample(float(t), state).vector_mps.copy())
        seq_b.append(b.sample(float(t), state).vector_mps.copy())
    seq_a = np.asarray(seq_a)
    seq_b = np.asarray(seq_b)

    assert np.allclose(seq_a, seq_b)
    assert np.std(seq_a[:, 0]) > 0.05
    assert np.std(seq_a[:, 1]) > 0.05
    assert np.std(seq_a[:, 2]) > 0.01


def test_iso_style_weighting_reports_weighted_rms():
    dt = 0.01
    monitor = PassengerComfortMonitor(
        limits=ComfortLimits(
            acceleration_mps2=0.50,
            jerk_mps3=1.50,
            angular_rate_deg_s=10.0,
            rms_window_s=5.0,
            iso_weighted_rms_mps2=0.315,
        ),
        sample_period_s=dt,
    )
    result = None
    for k in range(700):
        t = k * dt
        # 5-Hz vertical vibration lies close to the high-sensitivity Wk region.
        accel = np.array([0.0, 0.0, 0.4 * np.sin(2.0 * np.pi * 5.0 * t)])
        result = monitor.update(t, accel, np.zeros(3), np.zeros(3))

    assert result is not None
    assert result["comfort_iso_weighted_rms_mps2"] > 0.15
    assert result["comfort_iso_weighted_rms_mps2"] < 0.50
    assert np.isfinite(result["comfort_iso_index"])


def test_ground_kinematic_jerk_starts_at_zero():
    sim = ComfortAwareSimulation(dt=0.01)
    sample = sim.step(np.zeros(3))
    assert np.allclose(sample.diagnostics["accel_world"], 0.0, atol=1e-8)
    assert np.allclose(sample.diagnostics["jerk_world"], 0.0, atol=1e-8)


def test_jerk_guard_limits_vertical_takeoff_command_ramp():
    vehicle = Tiltrotor()
    commander = CommandManager()
    controller = ComfortAwareController(vehicle, commander)
    controller.max_accel = 0.50
    controller.max_command_jerk_mps3 = 1.50
    commander.vertical_takeoff(0.0, altitude_m=30.0)
    state = vehicle.initial_state

    first = controller.update(0.0, state)
    second = controller.update(0.01, state)
    g = vehicle.params["gravity"]
    mass = vehicle.mass
    a0 = float(first["cmd_force_body"][2] / mass - g)
    a1 = float(second["cmd_force_body"][2] / mass - g)

    assert abs(a0) < 1e-8
    assert abs(a1 - a0) <= 1.50 * 0.01 + 1e-6


def test_enhanced_dashboard_exposes_new_scenarios_and_iso_stream():
    doc = Document()
    sim = build_dashboard(doc)
    assert sim is not None

    selectors = {
        item.title: item
        for item in doc.select({"type": Select})
    }
    assert "Repeated gusts" in selectors["Wind scenario"].options
    assert "Continuous turbulence" in selectors["Wind scenario"].options
    assert selectors["Wind scenario"].value == "Repeated gusts"

    inputs = {
        item.title: item
        for item in doc.select({"type": NumericInput})
    }
    assert inputs["Delay after scenario enable [s]"].value == 2.0
    assert np.isclose(
        inputs["Transient accel target [m/s²] (NASA-informed)"].value,
        0.50,
    )

    labels = {
        item.label for item in doc.select({"type": Toggle})
    }
    assert "Comfort accel + jerk guard: OFF" in labels

    assert any(
        "comfort_iso_weighted_rms_mps2" in source.data
        for source in doc.select({"type": ColumnDataSource})
    )
