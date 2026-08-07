from copy import deepcopy

import numpy as np
from bokeh.document import Document

from rotorpy_tiltrotor.commands import FlightPhase
from rotorpy_tiltrotor.simulator import SimulationSample
from rotorpy_tiltrotor.v045_final_dashboard import (
    ForceFeasibleComfortController,
    _new_simulation,
    build_dashboard,
    sample_to_record_final,
)
from rotorpy_tiltrotor.vehicle import Tiltrotor


def _hover_rotor_speed(vehicle: Tiltrotor) -> float:
    p = vehicle.params
    total = vehicle.mass * p["gravity"]
    return float(np.sqrt(total / (vehicle.num_rotors * p["k_eta"])))


def _hover_state(vehicle: Tiltrotor) -> dict:
    state = deepcopy(vehicle.initial_state)
    state["x"] = np.array([0.0, 0.0, 30.0])
    state["v"] = np.zeros(3)
    state["wind"] = np.zeros(3)
    state["rotor_speeds"] = np.full(
        vehicle.num_rotors, _hover_rotor_speed(vehicle)
    )
    state["tilt_angle"] = np.array(0.0)
    return state


def test_low_speed_forward_transition_tilt_is_force_feasible_under_guard():
    vehicle = Tiltrotor()
    controller = ForceFeasibleComfortController(vehicle)
    controller.max_accel = 0.50
    controller.max_command_jerk_mps3 = 1.50
    controller.command_jerk_headroom = 0.20
    controller.commander.transition_to_cruise(
        0.0,
        altitude_m=30.0,
        airspeed_mps=15.0,
        route_x_m=0.0,
        route_y_m=0.0,
    )
    state = _hover_state(vehicle)
    state["v"] = np.array([2.0, 0.0, 0.0])

    controller._update_control_dt(0.10)
    tilt_cmd = controller._tilt_command(20.0, state)

    # With almost all weight still rotor-borne, 0.5 m/s^2 permits only a few
    # degrees of common-axis tilt.  This is the geometric coupling that caused
    # the earlier 5-6 m/s^2 forward-acceleration spike.
    assert tilt_cmd < np.deg2rad(4.0)


def test_ground_speed_is_separate_from_airspeed_in_windy_hover():
    vehicle = Tiltrotor()
    state = _hover_state(vehicle)
    state["wind"] = np.array([0.0, 5.0, 0.0])
    diagnostics = vehicle._empty_diagnostics()
    diagnostics["airspeed"] = 5.0
    control = ForceFeasibleComfortController(vehicle).zero_control()
    control["phase_code"] = np.array(float(FlightPhase.HOVER))
    sample = SimulationSample(0.0, state, control, diagnostics)

    rec = sample_to_record_final(sample, vehicle.params)

    assert np.isclose(rec["ground_speed_mps"], 0.0)
    assert np.isclose(rec["airspeed_mps"], 5.0)


def test_guarded_calm_forward_transition_removes_large_controller_induced_accel_spike():
    sim = _new_simulation()
    sim.reset(_hover_state(sim.vehicle))
    sim.controller.max_accel = 0.50
    sim.controller.max_command_jerk_mps3 = 1.50
    sim.controller.command_jerk_headroom = 0.20
    sim.commander.hover(0.0, 30.0, 0.0, 0.0)

    # Let the rotor/motor state settle before introducing the mode change.
    while sim.t < 2.0:
        sim.step(np.zeros(3))

    sim.commander.transition_to_cruise(
        sim.t,
        altitude_m=30.0,
        airspeed_mps=15.0,
        route_x_m=0.0,
        route_y_m=0.0,
    )
    entry = sim.t
    measured_accel = []
    command_accel = []
    command_jerk = []

    while sim.t - entry < 38.0:
        sample = sim.step(np.zeros(3))
        safety = float(getattr(
            sim.controller, "safety_override_factor", 0.0
        ))
        if sim.t - entry > 0.75 and safety <= 0.05:
            measured_accel.append(float(np.linalg.norm(
                sample.diagnostics.get("accel_world", np.zeros(3))
            )))
            command_accel.append(float(np.linalg.norm(
                sim.controller._last_priority_accel_world
            )))
            command_jerk.append(float(
                sim.controller.last_command_jerk_norm_mps3
            ))
        if (
            np.linalg.norm(sim.state["v"][:2]) > 13.5
            and float(sim.state["tilt_angle"]) > np.deg2rad(70.0)
        ):
            break

    assert command_accel
    assert max(command_accel) <= 0.50001
    assert max(command_jerk) <= 0.30001

    # This is a plant-response regression rather than a claim that external
    # disturbances can be hard-bounded.  In calm air it specifically prevents
    # the earlier controller/tilt-geometry-generated 5-6 m/s^2 spike.
    p95 = float(np.quantile(measured_accel, 0.95))
    peak = float(max(measured_accel))
    assert p95 < 0.90, f"p95={p95:.3f}, peak={peak:.3f}"
    assert peak < 1.50, f"p95={p95:.3f}, peak={peak:.3f}"


def test_final_dashboard_adds_ground_speed_and_comfort_tilt_cap_traces():
    doc = Document()
    build_dashboard(doc)
    titles = {
        getattr(getattr(model, "title", None), "text", "")
        for model in doc.models
    }
    assert "Flight state" in titles
    assert "Tilt and control surfaces" in titles
