"""Control-allocation helpers for common-axis quad tiltrotor propulsors."""
from __future__ import annotations

import numpy as np
from scipy.optimize import lsq_linear


def rotor_axis(tilt_angle: float) -> np.ndarray:
    return np.array([np.sin(tilt_angle), 0.0, np.cos(tilt_angle)])


def allocation_matrix(tilt_angle: float, params: dict) -> np.ndarray:
    axis = rotor_axis(tilt_angle)
    r = params["rotor_pos"]
    dirs = params["rotor_directions"]
    torque_per_thrust = params["k_m"] / params["k_eta"]
    moments = np.cross(r, axis[None, :]) + dirs[:, None] * torque_per_thrust * axis[None, :]
    return np.vstack([np.ones(params["num_rotors"]), moments.T])


def allocate_rotor_thrusts(collective_thrust: float, desired_moment: np.ndarray, tilt_angle: float, params: dict) -> np.ndarray:
    A = allocation_matrix(tilt_angle, params)
    collective_thrust = max(0.0, float(collective_thrust))
    target = np.concatenate([[collective_thrust], np.asarray(desired_moment, dtype=float)])
    weights = np.diag([2.5, 1.4, 1.4, 1.0])
    f_equal = np.full(params["num_rotors"], collective_thrust / params["num_rotors"])
    regularization = 0.35
    A_aug = np.vstack([weights @ A, regularization * np.eye(params["num_rotors"])])
    b_aug = np.concatenate([weights @ target, regularization * f_equal])
    f_max = params["k_eta"] * params["rotor_speed_max"] ** 2
    result = lsq_linear(A_aug, b_aug, bounds=(0.0, f_max), lsmr_tol="auto")
    return np.asarray(result.x)


def thrusts_to_speeds(thrusts: np.ndarray, params: dict) -> np.ndarray:
    speeds = np.sqrt(np.maximum(0.0, np.asarray(thrusts)) / params["k_eta"])
    return np.clip(speeds, params["rotor_speed_min"], params["rotor_speed_max"])
