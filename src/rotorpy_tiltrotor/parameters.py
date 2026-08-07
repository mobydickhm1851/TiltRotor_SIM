"""Default parameters for a four-propulsor winged tiltrotor.

The values are intentionally transparent, reduced-order starting points. They are
not tied to a certified aircraft and must be replaced with identified geometry,
wind-tunnel, CFD, or flight-test data before making performance claims.

Coordinate convention follows RotorPy:
- inertial/world: x forward/north, y left/west, z up
- body: x forward, y left, z up
- quaternion: scipy/RotorPy order [x, y, z, w], body-to-world
"""
from __future__ import annotations

from copy import deepcopy
import numpy as np


def default_tiltrotor_params() -> dict:
    params = {
        "mass": 25.0,
        "inertia": np.diag([18.0, 26.0, 38.0]),
        "gravity": 9.81,
        "rotor_pos": np.array([
            [0.90, 1.25, 0.05],
            [-0.90, 1.25, 0.05],
            [-0.90, -1.25, 0.05],
            [0.90, -1.25, 0.05],
        ], dtype=float),
        "rotor_directions": np.array([1.0, -1.0, 1.0, -1.0]),
        "num_rotors": 4,
        "k_eta": 4.2e-4,
        "k_m": 7.0e-6,
        "rotor_speed_min": 0.0,
        "rotor_speed_max": 650.0,
        "rotor_accel_max": 1800.0,
        "tau_m": 0.12,
        "tilt_min": np.deg2rad(0.0),
        "tilt_max": np.deg2rad(90.0),
        "tilt_rate_max": np.deg2rad(18.0),
        "tau_tilt": 0.30,
        "surface_min": np.deg2rad(np.array([-25.0, -25.0, -30.0])),
        "surface_max": np.deg2rad(np.array([25.0, 25.0, 30.0])),
        "surface_rate_max": np.deg2rad(np.array([80.0, 70.0, 90.0])),
        "tau_surface": np.array([0.08, 0.10, 0.08]),
        "rho": 1.225,
        "S": 2.80,
        "b": 4.20,
        "c": 0.72,
        "alpha_stall": np.deg2rad(16.0),
        # The lateral/directional derivatives below are a linear small-sideslip
        # model. Saturate their beta input outside this envelope instead of
        # extrapolating C_Y/C_l/C_n to +/-90 deg during low-speed crosswind.
        "beta_linear_limit": np.deg2rad(20.0),
        "wing_aero_start_mps": 2.0,
        "wing_aero_full_mps": 7.0,
        "wing_tilt_start": np.deg2rad(15.0),
        "wing_tilt_full": np.deg2rad(70.0),
        "wing_early_transition_fraction": 0.35,
        "CL0": 0.65,
        "CL_alpha": 4.70,
        "CL_de": -0.18,
        "CD0": 0.045,
        "CD_k": 0.070,
        "CY_beta": -0.90,
        "CY_p": 0.00,
        "CY_r": 0.18,
        "CY_dr": 0.22,
        "Cl_beta": -0.12,
        "Cl_p_damp": 0.48,
        "Cl_r": 0.10,
        "Cl_da": 0.14,
        "Cl_dr": 0.025,
        "Cm0": 0.0,
        "Cm_alpha": 0.82,
        "Cm_q_damp": 7.0,
        "Cm_de": 1.10,
        "Cn_beta": 0.20,
        "Cn_p": -0.055,
        "Cn_r_damp": 0.30,
        "Cn_da": -0.018,
        "Cn_dr": 0.18,
        "body_drag": np.diag([0.12, 0.20, 0.25]),
        "battery_capacity_j": 6.0 * 3.6e6,
        "battery_initial_soc": 1.0,
        "power_coeff": 1.5e-5,
        "avionics_power_w": 180.0,
        "ground_enabled": True,
        "ground_friction_rate": 3.0,
        "integrator": "rk4",
    }
    return deepcopy(params)


def initial_state(params: dict | None = None) -> dict:
    p = default_tiltrotor_params() if params is None else params
    return {
        "x": np.array([0.0, 0.0, 0.0]),
        "v": np.zeros(3),
        "q": np.array([0.0, 0.0, 0.0, 1.0]),
        "w": np.zeros(3),
        "wind": np.zeros(3),
        "rotor_speeds": np.zeros(p["num_rotors"]),
        "tilt_angle": np.array(0.0),
        "surfaces": np.zeros(3),
        "energy_j": np.array(
            p["battery_capacity_j"] * p["battery_initial_soc"]
        ),
    }