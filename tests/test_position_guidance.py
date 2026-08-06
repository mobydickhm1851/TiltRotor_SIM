import numpy as np

from rotorpy_tiltrotor import Tiltrotor, TiltrotorSimulation
from rotorpy_tiltrotor.aerodynamics import aerodynamic_wrench
from rotorpy_tiltrotor.commands import AutomaticMission, FlightPhase


def test_vertical_climb_does_not_activate_fixed_wing_model():
    vehicle = Tiltrotor()
    aero = aerodynamic_wrench(
        body_airspeed=np.array([0.0, 0.0, 5.0]),
        body_rates=np.zeros(3),
        surfaces=np.zeros(3),
        params=vehicle.params,
        tilt_angle=0.0,
    )

    assert aero.wing_blend == 0.0
    assert aero.qbar == 0.0
    assert aero.alpha == 0.0
    assert np.allclose(aero.moment_body, 0.0)
    assert np.allclose(aero.force_body[:2], 0.0)
    assert all(value == 0.0 for value in aero.coefficients.values())


def test_automatic_mission_preserves_original_takeoff_hold_point():
    sim = TiltrotorSimulation(dt=0.02)
    mission = AutomaticMission(sim.commander, altitude_m=30.0)

    initial = sim.state.copy()
    initial["x"] = np.array([12.0, -3.0, 0.0])
    mission.update(0.0, initial)
    assert sim.commander.setpoint.hold_x_m == 12.0
    assert sim.commander.setpoint.hold_y_m == -3.0

    drifted = {
        key: np.copy(value) if isinstance(value, np.ndarray) else value
        for key, value in initial.items()
    }
    drifted["x"] = np.array([12.7, -2.4, 29.5])
    drifted["v"] = np.array([0.0, 0.0, 0.1])
    mission.update(20.0, drifted)

    assert sim.commander.setpoint.phase == FlightPhase.HOVER
    assert sim.commander.setpoint.hold_x_m == 12.0
    assert sim.commander.setpoint.hold_y_m == -3.0


def test_cruise_guidance_commands_return_to_route_centreline():
    sim = TiltrotorSimulation(dt=0.01)
    sim.state["x"] = np.array([0.0, 5.0, 30.0])
    sim.state["v"] = np.array([15.0, 0.0, 0.0])
    sim.state["tilt_angle"] = np.array(np.deg2rad(90.0))
    sim.commander.cruise(
        0.0,
        altitude_m=30.0,
        airspeed_mps=15.0,
        heading_rad=0.0,
        route_x_m=0.0,
        route_y_m=0.0,
    )

    sim.controller._update_control_dt(0.0)
    sim.controller._update_phase_memory(sim.state)
    velocity_ref, acceleration_cmd = sim.controller._desired_velocity_and_acceleration(
        0.0, sim.state
    )

    assert sim.controller._last_cross_track_error == -5.0
    assert velocity_ref[1] < 0.0
    assert acceleration_cmd[1] < 0.0
