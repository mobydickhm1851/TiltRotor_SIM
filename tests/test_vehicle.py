import numpy as np

from rotorpy_tiltrotor import Tiltrotor


def hover_control(vehicle):
    omega = np.sqrt(vehicle.mass * vehicle.g / (vehicle.num_rotors * vehicle.params["k_eta"]))
    return {
        "cmd_motor_speeds": np.full(vehicle.num_rotors, omega),
        "cmd_tilt_angle": np.array(0.0),
        "cmd_surfaces": np.zeros(3),
    }


def test_hover_thrust_matches_weight():
    vehicle = Tiltrotor()
    omega = np.sqrt(vehicle.mass * vehicle.g / (vehicle.num_rotors * vehicle.params["k_eta"]))
    force, _ = vehicle.rotor_wrench(np.full(vehicle.num_rotors, omega), 0.0)
    assert np.isclose(force[2], vehicle.mass * vehicle.g, rtol=1e-6)
    assert np.allclose(force[:2], 0.0, atol=1e-9)


def test_quaternion_remains_normalized():
    vehicle = Tiltrotor()
    state = vehicle.initial_state
    state["q"] = np.array([0.05, -0.02, 0.03, 0.998])
    state["q"] /= np.linalg.norm(state["q"])
    state["w"] = np.array([0.2, -0.1, 0.15])
    control = hover_control(vehicle)
    for _ in range(100):
        state = vehicle.step(state, control, 0.01)
    assert np.isclose(np.linalg.norm(state["q"]), 1.0, atol=1e-10)


def test_tilt_rate_and_bounds():
    vehicle = Tiltrotor()
    state = vehicle.initial_state
    control = hover_control(vehicle)
    control["cmd_tilt_angle"] = np.array(np.deg2rad(120.0))
    next_state = vehicle.step(state, control, 0.1)
    assert 0.0 <= float(next_state["tilt_angle"]) <= vehicle.params["tilt_max"]
    assert float(next_state["tilt_angle"]) <= vehicle.params["tilt_rate_max"] * 0.1 * 1.05
