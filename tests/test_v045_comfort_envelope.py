from copy import deepcopy

import numpy as np
from bokeh.document import Document
from bokeh.models import Div

from rotorpy_tiltrotor.commands import FlightPhase
from rotorpy_tiltrotor.simulator import SimulationSample
from rotorpy_tiltrotor.v045_dashboard import (
    BACK_TRANSITION_BRAKE_FRACTION,
    ComfortEnvelopeController,
    GuardAwareAutomaticMission,
    _new_simulation,
    build_dashboard,
    sample_to_record_v045,
)
from rotorpy_tiltrotor.vehicle import Tiltrotor


def _safe_cruise_state(vehicle, speed=15.0):
    state = deepcopy(vehicle.initial_state)
    state["x"] = np.array([0.0, 0.0, 30.0])
    state["v"] = np.array([speed, 0.0, 0.0])
    state["wind"] = np.zeros(3)
    state["tilt_angle"] = np.array(np.deg2rad(90.0))
    return state


def test_nominal_guard_is_true_3d_command_norm_envelope():
    vehicle = Tiltrotor()
    controller = ComfortEnvelopeController(vehicle)
    controller.max_accel = 0.50
    controller.max_command_jerk_mps3 = None
    controller.commander.transition_to_cruise(
        0.0,
        altitude_m=30.0,
        airspeed_mps=15.0,
        route_x_m=0.0,
        route_y_m=0.0,
    )
    state = _safe_cruise_state(vehicle, speed=5.0)
    state["x"][1] = 3.0
    state["x"][2] = 30.15

    controller.update(0.10, state)

    assert not controller.safety_override_active
    assert np.linalg.norm(controller._last_priority_accel_world) <= 0.500001


def test_back_transition_uses_most_nominal_budget_for_braking():
    vehicle = Tiltrotor()
    controller = ComfortEnvelopeController(vehicle)
    controller.max_accel = 0.50
    controller.max_command_jerk_mps3 = None
    controller.commander.transition_to_hover(
        0.0,
        altitude_m=30.0,
        x_m=260.0,
        y_m=0.0,
        duration_s=24.0,
    )
    state = _safe_cruise_state(vehicle, speed=15.0)

    controller.update(0.10, state)
    accel = controller._last_priority_accel_world

    assert not controller.safety_override_active
    assert np.linalg.norm(accel) <= 0.500001
    assert accel[0] <= -(BACK_TRANSITION_BRAKE_FRACTION * 0.50 - 1e-6)


def test_nominal_command_jerk_is_limited_as_3d_norm():
    vehicle = Tiltrotor()
    controller = ComfortEnvelopeController(vehicle)
    controller.max_accel = 0.50
    controller.max_command_jerk_mps3 = 1.50
    controller.command_jerk_headroom = 0.20
    state = _safe_cruise_state(vehicle, speed=15.0)

    controller.commander.cruise(
        0.0,
        altitude_m=30.0,
        airspeed_mps=15.0,
        route_x_m=0.0,
        route_y_m=0.0,
    )
    controller.update(0.00, state)

    controller.commander.transition_to_hover(
        0.05,
        altitude_m=30.0,
        x_m=260.0,
        y_m=0.0,
        duration_s=24.0,
    )
    controller.update(0.10, state)

    assert not controller.safety_override_active
    assert controller.last_command_jerk_norm_mps3 <= 0.300001
    assert np.linalg.norm(controller._last_priority_accel_world) <= 0.500001


def test_guard_aware_automatic_stop_plan_uses_available_deceleration():
    vehicle = Tiltrotor()
    controller = ComfortEnvelopeController(vehicle)
    controller.max_accel = 0.50
    controller.max_command_jerk_mps3 = 1.50
    controller.command_jerk_headroom = 0.20
    mission = GuardAwareAutomaticMission(
        controller.commander,
        controller=controller,
        altitude_m=30.0,
        cruise_speed_mps=15.0,
        cruise_distance_m=180.0,
    )
    mission.takeoff_x = 0.0
    mission.takeoff_y = 0.0
    controller.commander.cruise(
        0.0,
        altitude_m=30.0,
        airspeed_mps=15.0,
        route_x_m=0.0,
        route_y_m=0.0,
    )
    state = _safe_cruise_state(vehicle, speed=15.0)
    state["x"][0] = 180.0

    target_x, _ = mission._planned_back_transition_target(state)

    assert np.isclose(
        mission._effective_back_decel(),
        BACK_TRANSITION_BRAKE_FRACTION * 0.50,
    )
    assert mission.back_transition_duration_s > 33.0
    assert target_x - state["x"][0] > 245.0


def test_back_transition_tracking_reports_route_error_not_stop_distance_as_x_error():
    vehicle = Tiltrotor()
    state = deepcopy(vehicle.initial_state)
    state["x"] = np.array([100.0, 10.0, 30.0])
    state["v"] = np.array([12.0, 0.0, 0.0])
    control = ComfortEnvelopeController(vehicle).zero_control()
    control.update({
        "phase_code": np.array(float(FlightPhase.TRANSITION_TO_HOVER)),
        "target_altitude": np.array(30.0),
        "target_airspeed": np.array(0.0),
        "target_x": np.array(300.0),
        "target_y": np.array(0.0),
        "route_anchor_x": np.array(300.0),
        "route_anchor_y": np.array(0.0),
        "route_heading": np.array(0.0),
        "comfort_command_accel_mps2": np.array(0.45),
        "comfort_command_jerk_mps3": np.array(0.30),
        "altitude_safety_factor": np.array(0.0),
    })
    sample = SimulationSample(
        t=0.0,
        state=state,
        control=control,
        diagnostics=vehicle._empty_diagnostics(),
    )

    rec = sample_to_record_v045(sample, vehicle.params)

    assert abs(rec["x_error_m"]) < 1e-9
    assert np.isclose(rec["y_error_m"], -10.0)
    assert np.isclose(rec["distance_to_stop_m"], 200.0)
    assert rec["position_reference_type"] == 1.0


def test_v045_dashboard_builds_command_traces_and_wrap_safe_explanation():
    doc = Document()
    sim = _new_simulation()
    build_dashboard(doc, simulation=sim)

    divs = list(doc.select({"type": Div}))
    assert any(
        "v0.4.5 comfort envelope" in str(div.text)
        for div in divs
    )

    plot_titles = {
        getattr(getattr(model, "title", None), "text", "")
        for model in doc.models
    }
    assert "Position / route tracking and planned stopping distance" in plot_titles
