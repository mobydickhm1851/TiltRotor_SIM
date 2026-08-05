import numpy as np

from rotorpy_tiltrotor import TiltrotorSimulation
from rotorpy_tiltrotor.commands import FlightPhase
from rotorpy_tiltrotor.math_utils import euler_deg


def test_manual_takeoff_converges_without_pitch_divergence():
    sim = TiltrotorSimulation(dt=0.02)
    sim.commander.vertical_takeoff(0.0, 30.0, 0.0, 0.0)

    max_pitch = 0.0
    max_horizontal_error = 0.0
    for _ in range(1500):  # 30 simulated seconds
        sim.step(np.zeros(3))
        max_pitch = max(max_pitch, abs(float(euler_deg(sim.state["q"])[1])))
        max_horizontal_error = max(max_horizontal_error, abs(float(sim.state["x"][0])))

    assert max_pitch < 2.0
    assert max_horizontal_error < 0.5
    assert abs(float(sim.state["x"][2]) - 30.0) < 0.15
    assert abs(float(sim.state["v"][2])) < 0.10


def test_simulation_reset_clears_command_and_pid_memory():
    sim = TiltrotorSimulation(dt=0.02)
    sim.commander.vertical_takeoff(0.0, 10.0, 0.0, 0.0)
    for _ in range(50):
        sim.step(np.zeros(3))

    assert np.linalg.norm(sim.controller.position_integral) > 0.0
    sim.reset()

    assert sim.commander.setpoint.phase == FlightPhase.GROUND
    assert np.allclose(sim.controller.position_integral, 0.0)
    assert np.allclose(sim.controller.attitude_integral, 0.0)
    assert sim.t == 0.0
