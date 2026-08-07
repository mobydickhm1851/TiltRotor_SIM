import numpy as np
from bokeh.document import Document
from bokeh.models import Div, NumericInput, Select

from rotorpy_tiltrotor.commands import AutomaticMission, FlightPhase
from rotorpy_tiltrotor.scenarios import UrbanWindModel, WindScenarioConfig
from rotorpy_tiltrotor.v044_dashboard import (
    SafetyPriorityController,
    _new_simulation,
    build_dashboard,
)
from rotorpy_tiltrotor.vehicle import Tiltrotor


def test_comfort_limited_forward_transition_keeps_tilt_airspeed_gated():
    vehicle = Tiltrotor()
    controller = SafetyPriorityController(vehicle)
    controller.max_accel = 0.50
    controller.max_command_jerk_mps3 = 1.50
    controller.commander.transition_to_cruise(
        0.0, altitude_m=30.0, airspeed_mps=15.0
    )
    state = vehicle.initial_state
    state["x"] = np.array([0.0, 0.0, 30.0])
    state["v"] = np.array([2.0, 0.0, 0.0])
    state["tilt_angle"] = np.array(0.0)

    tilt_cmd = controller._tilt_command(20.0, state)
    assert tilt_cmd < np.deg2rad(20.0)


def test_altitude_loss_overrides_comfort_and_reduces_cruise_tilt():
    vehicle = Tiltrotor()
    controller = SafetyPriorityController(vehicle)
    controller.max_accel = 0.50
    controller.max_command_jerk_mps3 = 1.50
    controller.commander.cruise(
        0.0, altitude_m=30.0, airspeed_mps=15.0
    )
    state = vehicle.initial_state
    state["x"] = np.array([0.0, 0.0, 27.0])
    state["v"] = np.array([15.0, 0.0, -1.0])
    state["tilt_angle"] = np.array(np.deg2rad(90.0))

    controller._update_control_dt(0.1)
    _, accel = controller._desired_velocity_and_acceleration(0.1, state)
    tilt_cmd = controller._tilt_command(0.1, state)

    assert controller.safety_override_active
    assert accel[2] > 0.50
    assert tilt_cmd < np.deg2rad(80.0)


def _run_until_prelanding(sim, mission, wind_model, max_time_s=150.0):
    min_operational_altitude = float("inf")
    minimum_snapshot = None
    phases_seen = set()
    while sim.t < max_time_s:
        mission.update(sim.t, sim.state)
        phase = sim.commander.setpoint.phase
        phases_seen.add(phase)
        if phase == FlightPhase.VERTICAL_LANDING:
            break
        wind_sample = wind_model.sample(sim.t, sim.state)
        sim.step(wind_sample.vector_mps)
        if phase in {
            FlightPhase.HOVER,
            FlightPhase.TRANSITION_TO_CRUISE,
            FlightPhase.CRUISE,
            FlightPhase.TRANSITION_TO_HOVER,
        }:
            altitude = float(sim.state["x"][2])
            if altitude < min_operational_altitude:
                min_operational_altitude = altitude
                minimum_snapshot = {
                    "t": float(sim.t),
                    "phase": phase.name,
                    "altitude": altitude,
                    "velocity": np.asarray(sim.state["v"]).round(3).tolist(),
                    "tilt_deg": float(np.rad2deg(sim.state["tilt_angle"])),
                    "body_rate_deg_s": float(
                        np.linalg.norm(sim.state["w"]) * 180.0 / np.pi
                    ),
                    "wind": np.asarray(wind_sample.vector_mps).round(3).tolist(),
                    "safety": float(getattr(
                        sim.controller, "safety_override_factor", 0.0
                    )),
                }
        assert np.all(np.isfinite(sim.state["x"]))
        assert np.all(np.isfinite(sim.state["v"]))
        assert np.all(np.isfinite(sim.state["q"]))
    return min_operational_altitude, phases_seen, minimum_snapshot


def test_automatic_mission_survives_reported_10_mps_repeated_gust_case():
    sim = _new_simulation()
    mission = AutomaticMission(
        sim.commander,
        altitude_m=30.0,
        cruise_speed_mps=15.0,
        cruise_distance_m=180.0,
    )
    wind_model = UrbanWindModel(WindScenarioConfig(
        enabled=True,
        mode="repeated_gusts",
        base_wind_mps=np.array([0.0, 0.0, -2.25]),
        disturbance_amplitude_mps=10.0,
        disturbance_direction_deg=90.0,
        start_time_s=2.0,
        duration_s=3.0,
        wake_frequency_hz=0.25,
        reference_altitude_m=30.0,
        random_seed=7,
    ))

    minimum, phases, snapshot = _run_until_prelanding(
        sim, mission, wind_model, max_time_s=150.0
    )

    assert FlightPhase.CRUISE in phases
    assert FlightPhase.TRANSITION_TO_HOVER in phases
    assert FlightPhase.HOVER in phases
    assert minimum > 24.0, f"minimum snapshot: {snapshot}"


def test_comfort_guard_hover_to_cruise_does_not_trade_away_altitude():
    sim = _new_simulation()
    sim.controller.max_accel = 0.50
    sim.controller.max_command_jerk_mps3 = 1.50
    sim.controller.command_jerk_headroom = 0.20
    mission = AutomaticMission(
        sim.commander,
        altitude_m=30.0,
        cruise_speed_mps=15.0,
        cruise_distance_m=10000.0,
    )
    calm = UrbanWindModel(WindScenarioConfig(enabled=False))

    minimum, phases, snapshot = _run_until_prelanding(
        sim, mission, calm, max_time_s=100.0
    )

    assert FlightPhase.TRANSITION_TO_CRUISE in phases
    assert FlightPhase.CRUISE in phases
    assert minimum > 27.0, f"minimum snapshot: {snapshot}"


def test_v044_dashboard_uses_wrap_safe_explanations_and_short_titles():
    doc = Document()
    build_dashboard(doc)

    inputs = list(doc.select({"type": NumericInput}))
    titles = {str(model.title) for model in inputs}
    assert "Delay [s]" in titles
    assert "Duration [s]" in titles
    assert "Frequency [Hz]" in titles

    selector = next(
        model for model in doc.select({"type": Select})
        if model.title == "Wind scenario"
    )
    selector.value = "Repeated gusts"

    wrap_blocks = [
        model for model in doc.select({"type": Div})
        if "overflow-wrap" in dict(getattr(model, "styles", {}) or {})
    ]
    assert wrap_blocks
    assert any(
        "Altitude-priority factor" in str(model.text)
        for model in wrap_blocks
    )
