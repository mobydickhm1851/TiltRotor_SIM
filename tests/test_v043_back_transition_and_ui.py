import numpy as np
from bokeh.document import Document
from bokeh.models import Div, NumericInput, Select, Toggle
from scipy.spatial.transform import Rotation

from rotorpy_tiltrotor import dashboard as base_dashboard
from rotorpy_tiltrotor.aerodynamics import aerodynamic_wrench
from rotorpy_tiltrotor.commands import AutomaticMission, FlightPhase
from rotorpy_tiltrotor.dashboard import planned_hover_target, supervise_manual_phase
from rotorpy_tiltrotor.math_utils import rotation_matrix
from rotorpy_tiltrotor.parameters import default_tiltrotor_params
from rotorpy_tiltrotor.scenarios import UrbanWindModel, WindScenarioConfig
from rotorpy_tiltrotor.v042_dashboard import CS_AWO_LABEL
from rotorpy_tiltrotor.v043_dashboard import (
    BACK_TRANSITION_DECEL_MPS2,
    BACK_TRANSITION_DURATION_S,
    BackTransitionSafeController,
    _new_simulation,
    build_dashboard,
)
from rotorpy_tiltrotor.vehicle import Tiltrotor


def test_lateral_coefficients_saturate_outside_linear_beta_envelope():
    p = default_tiltrotor_params()
    result = aerodynamic_wrench(
        body_airspeed=np.array([3.0, 8.0, 0.0]),
        body_rates=np.zeros(3),
        surfaces=np.zeros(3),
        params=p,
        tilt_angle=0.0,
    )

    beta_limit = float(p["beta_linear_limit"])
    assert abs(result.beta) > beta_limit
    # With zero rates/surfaces the directional coefficient should be bounded by
    # the soft-saturated beta input, not the raw ~70-deg sideslip angle.
    expected_bound = (
        result.wing_blend * abs(p["Cn_beta"]) * beta_limit * 1.01
    )
    assert abs(result.coefficients["Cn"]) <= expected_bound


def test_world_vertical_collective_floor_supports_banked_hover():
    vehicle = Tiltrotor()
    controller = BackTransitionSafeController(vehicle)
    controller.commander.hover(0.0, 30.0, 0.0, 0.0)
    state = vehicle.initial_state
    state["x"] = np.array([0.0, 0.0, 30.0])
    state["q"] = Rotation.from_euler(
        "xyz", [np.deg2rad(24.0), np.deg2rad(18.0), 0.0]
    ).as_quat()
    state["tilt_angle"] = np.array(0.0)

    control = controller.update(0.0, state)
    p = vehicle.params
    thrusts = p["k_eta"] * np.asarray(control["cmd_motor_speeds"]) ** 2
    R = rotation_matrix(state["q"])
    axis_world_z = float((R @ np.array([0.0, 0.0, 1.0]))[2])
    commanded_world_z = float(np.sum(thrusts)) * axis_world_z
    required_world_z = float((R @ control["cmd_force_body"])[2])

    assert commanded_world_z >= 0.995 * required_world_z


def _reach_stable_cruise(sim) -> None:
    mission = AutomaticMission(
        sim.commander,
        altitude_m=30.0,
        cruise_speed_mps=15.0,
        cruise_distance_m=10000.0,
    )
    cruise_since = None
    for _ in range(int(75.0 / sim.dt)):
        mission.update(sim.t, sim.state)
        sim.step(np.zeros(3))
        if sim.commander.setpoint.phase == FlightPhase.CRUISE:
            if cruise_since is None:
                cruise_since = sim.t
            if sim.t - cruise_since >= 3.0:
                return
    raise AssertionError("simulation did not reach stable cruise")


def test_back_transition_with_repeated_gusts_avoids_large_altitude_drop():
    sim = _new_simulation()
    _reach_stable_cruise(sim)

    # Match the comfort-oriented dashboard settings before commanding the back
    # transition. Safety-critical world-z collective protection is allowed to
    # override command shaping when required to hold altitude.
    sim.controller.max_accel = 0.50
    sim.controller.max_command_jerk_mps3 = 1.50

    sp = sim.commander.setpoint
    hover_target, _ = planned_hover_target(
        sim.state,
        sp,
        duration_s=BACK_TRANSITION_DURATION_S,
        comfortable_decel_mps2=min(0.50, BACK_TRANSITION_DECEL_MPS2),
    )
    sim.commander.transition_to_hover(
        sim.t,
        altitude_m=30.0,
        x_m=float(hover_target[0]),
        y_m=float(hover_target[1]),
        duration_s=BACK_TRANSITION_DURATION_S,
    )

    wind = UrbanWindModel(WindScenarioConfig(
        enabled=True,
        mode="repeated_gusts",
        disturbance_amplitude_mps=5.0,
        disturbance_direction_deg=90.0,
        start_time_s=0.0,
        duration_s=3.0,
        wake_frequency_hz=0.25,
        reference_altitude_m=30.0,
        random_seed=7,
    ))

    settle_time = 0.0
    min_altitude = float(sim.state["x"][2])
    max_body_rate_deg_s = 0.0
    hover_entered_at = None

    for _ in range(int(55.0 / sim.dt)):
        settle_time = supervise_manual_phase(
            sim, settle_time, sim.dt
        )
        wind_sample = wind.sample(sim.t, sim.state)
        sim.step(wind_sample.vector_mps)

        assert np.all(np.isfinite(sim.state["x"]))
        assert np.all(np.isfinite(sim.state["v"]))
        assert np.all(np.isfinite(sim.state["q"]))
        min_altitude = min(min_altitude, float(sim.state["x"][2]))
        max_body_rate_deg_s = max(
            max_body_rate_deg_s,
            float(np.linalg.norm(sim.state["w"])) * 180.0 / np.pi,
        )

        if sim.commander.setpoint.phase == FlightPhase.HOVER:
            if hover_entered_at is None:
                hover_entered_at = sim.t
            if sim.t - hover_entered_at >= 10.0:
                break

    assert hover_entered_at is not None
    # The previous dashboard trace lost roughly 12-13 m. Keep the regression
    # threshold tight enough that this failure cannot silently return.
    assert min_altitude > 26.0
    assert max_body_rate_deg_s < 45.0
    assert abs(float(sim.state["x"][2]) - 30.0) < 2.0


def test_v043_dashboard_has_single_label_writer_and_compact_titles():
    doc = Document()
    build_dashboard(doc)

    periods = [
        getattr(callback, "period", None)
        for callback in doc.session_callbacks
    ]
    assert 50 in periods
    assert 100 not in periods

    guard = next(
        model for model in doc.select({"type": Toggle})
        if str(model.label).startswith("Comfort accel guard")
    )
    assert guard.label == "Comfort accel guard: OFF"

    selector = next(
        model for model in doc.select({"type": Select})
        if model.title == "Wind scenario"
    )
    selector.value = CS_AWO_LABEL

    inputs = list(doc.select({"type": NumericInput}))
    assert any(model.title == "Mean wind U [m/s]" for model in inputs)
    assert any(model.title == "Duration [s]" for model in inputs)
    assert any(model.title == "Frequency [Hz]" for model in inputs)
    assert any(model.title == "Accel target [m/s²]" for model in inputs)
    assert any(model.title == "Jerk target [m/s³]" for model in inputs)

    help_blocks = [
        str(model.text) for model in doc.select({"type": Div})
    ]
    assert any("CS-AWO vertical low-altitude" in text for text in help_blocks)
    assert np.isclose(base_dashboard.BACK_TRANSITION_DURATION_S, 24.0)
    assert np.isclose(base_dashboard.BACK_TRANSITION_COMFORT_DECEL_MPS2, 0.75)
