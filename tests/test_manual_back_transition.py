import numpy as np

from rotorpy_tiltrotor import TiltrotorSimulation
from rotorpy_tiltrotor.commands import FlightPhase
from rotorpy_tiltrotor.dashboard import planned_hover_target, supervise_manual_phase


def test_planned_hover_target_uses_route_centreline_and_stopping_distance():
    sim = TiltrotorSimulation(dt=0.01)
    sim.state["x"] = np.array([100.0, 5.0, 30.0])
    sim.state["v"] = np.array([15.0, 0.0, 0.0])
    sim.commander.cruise(
        0.0,
        altitude_m=30.0,
        airspeed_mps=15.0,
        heading_rad=0.0,
        route_x_m=0.0,
        route_y_m=0.0,
    )

    target, forward_distance = planned_hover_target(
        sim.state,
        sim.commander.setpoint,
        duration_s=14.0,
    )

    assert np.isclose(target[1], 0.0)
    assert forward_distance >= 100.0
    assert np.isclose(target[0], 100.0 + forward_distance)


def test_planned_hover_target_supports_nonzero_heading():
    sim = TiltrotorSimulation(dt=0.01)
    sim.state["x"] = np.array([4.0, 50.0, 30.0])
    sim.state["v"] = np.array([0.0, 12.0, 0.0])
    sim.commander.cruise(
        0.0,
        altitude_m=30.0,
        airspeed_mps=12.0,
        heading_rad=np.pi / 2.0,
        route_x_m=0.0,
        route_y_m=0.0,
    )

    target, _ = planned_hover_target(sim.state, sim.commander.setpoint)

    assert abs(target[0]) < 1e-9
    assert target[1] > 50.0


def test_manual_back_transition_completes_to_hover():
    sim = TiltrotorSimulation(dt=0.01)
    sim.commander.transition_to_hover(
        0.0,
        altitude_m=30.0,
        x_m=120.0,
        y_m=0.0,
        duration_s=14.0,
    )
    sim.state["x"] = np.array([120.0, 0.0, 30.0])
    sim.state["v"] = np.zeros(3)
    sim.state["tilt_angle"] = np.array(0.0)
    sim.t = 14.0

    settle_time = 0.0
    for _ in range(100):
        settle_time = supervise_manual_phase(sim, settle_time, 0.01)

    assert sim.commander.setpoint.phase == FlightPhase.HOVER
    assert sim.commander.setpoint.hold_x_m == 120.0
    assert sim.commander.setpoint.hold_y_m == 0.0


def test_manual_back_transition_does_not_complete_at_high_speed():
    sim = TiltrotorSimulation(dt=0.01)
    sim.commander.transition_to_hover(
        0.0,
        altitude_m=30.0,
        x_m=120.0,
        y_m=0.0,
        duration_s=14.0,
    )
    sim.state["x"] = np.array([120.0, 0.0, 30.0])
    sim.state["v"] = np.array([4.0, 0.0, 0.0])
    sim.state["tilt_angle"] = np.array(0.0)
    sim.t = 14.0

    settle_time = 0.0
    for _ in range(100):
        settle_time = supervise_manual_phase(sim, settle_time, 0.01)

    assert sim.commander.setpoint.phase == FlightPhase.TRANSITION_TO_HOVER
    assert settle_time == 0.0
