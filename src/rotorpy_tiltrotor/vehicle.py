"""RotorPy-compatible nonlinear 6-DOF quad-tiltrotor vehicle."""
from __future__ import annotations

from copy import deepcopy
import numpy as np

try:
    from rotorpy.vehicles.multirotor import Multirotor as RotorPyMultirotor
except Exception:
    class RotorPyMultirotor:
        pass

from .aerodynamics import aerodynamic_wrench, AeroResult
from .allocator import rotor_axis
from .math_utils import quat_dot, normalize_quat, rotation_matrix
from .parameters import default_tiltrotor_params, initial_state


class Tiltrotor(RotorPyMultirotor):
    def __init__(self, params: dict | None = None, initial: dict | None = None):
        self.params = default_tiltrotor_params() if params is None else deepcopy(params)
        self.mass = float(self.params["mass"])
        self.inertia = np.asarray(self.params["inertia"], dtype=float)
        self.inv_inertia = np.linalg.inv(self.inertia)
        self.g = float(self.params["gravity"])
        self.num_rotors = int(self.params["num_rotors"])
        self.initial_state = initial_state(self.params) if initial is None else deepcopy(initial)
        self.last_diagnostics = self._empty_diagnostics()

    @staticmethod
    def _empty_diagnostics() -> dict:
        return {
            "airspeed": 0.0,
            "alpha": 0.0,
            "beta": 0.0,
            "qbar": 0.0,
            "wing_blend": 0.0,
            "aero_force_body": np.zeros(3),
            "aero_moment_body": np.zeros(3),
            "rotor_force_body": np.zeros(3),
            "rotor_moment_body": np.zeros(3),
            "total_force_body": np.zeros(3),
            "total_moment_body": np.zeros(3),
            "power_w": 0.0,
            "coefficients": {
                "CL": 0.0,
                "CD": 0.0,
                "CY": 0.0,
                "Cl": 0.0,
                "Cm": 0.0,
                "Cn": 0.0,
            },
        }

    @staticmethod
    def _pack_state(state: dict) -> np.ndarray:
        return np.concatenate([
            np.asarray(state["x"], dtype=float),
            np.asarray(state["v"], dtype=float),
            np.asarray(state["q"], dtype=float),
            np.asarray(state["w"], dtype=float),
            np.asarray(state["rotor_speeds"], dtype=float),
            np.atleast_1d(state["tilt_angle"]).astype(float),
            np.asarray(state["surfaces"], dtype=float),
            np.atleast_1d(state["energy_j"]).astype(float),
        ])

    def _unpack_state(self, s: np.ndarray, wind: np.ndarray) -> dict:
        n = self.num_rotors
        return {
            "x": s[0:3].copy(),
            "v": s[3:6].copy(),
            "q": s[6:10].copy(),
            "w": s[10:13].copy(),
            "wind": np.asarray(wind, dtype=float).copy(),
            "rotor_speeds": s[13:13+n].copy(),
            "tilt_angle": np.array(s[13+n]),
            "surfaces": s[14+n:17+n].copy(),
            "energy_j": np.array(s[17+n]),
        }

    def aero_wrench_from_state(self, state: dict) -> AeroResult:
        R = rotation_matrix(state["q"])
        body_airspeed = R.T @ (
            state["v"] - state.get("wind", np.zeros(3))
        )
        return aerodynamic_wrench(
            body_airspeed,
            state["w"],
            state["surfaces"],
            self.params,
            tilt_angle=float(state["tilt_angle"]),
        )

    def rotor_wrench(self, rotor_speeds: np.ndarray, tilt_angle: float):
        p = self.params
        axis = rotor_axis(tilt_angle)
        thrusts = p["k_eta"] * np.asarray(rotor_speeds, dtype=float) ** 2
        forces = thrusts[:, None] * axis[None, :]
        reaction = (
            p["rotor_directions"][:, None]
            * p["k_m"]
            * np.asarray(rotor_speeds)[:, None] ** 2
            * axis[None, :]
        )
        moments = np.cross(p["rotor_pos"], forces) + reaction
        return np.sum(forces, axis=0), np.sum(moments, axis=0)

    def compute_wrench(self, state: dict):
        aero = self.aero_wrench_from_state(state)
        rotor_force, rotor_moment = self.rotor_wrench(
            state["rotor_speeds"], float(state["tilt_angle"])
        )
        return (
            aero.force_body + rotor_force,
            aero.moment_body + rotor_moment,
            aero,
            rotor_force,
            rotor_moment,
        )

    def _control_arrays(self, control: dict):
        cmd_speed = np.asarray(
            control.get("cmd_motor_speeds", np.zeros(self.num_rotors)),
            dtype=float,
        )
        if cmd_speed.shape != (self.num_rotors,):
            raise ValueError(
                f"cmd_motor_speeds must have shape ({self.num_rotors},)"
            )
        cmd_speed = np.clip(
            cmd_speed,
            self.params["rotor_speed_min"],
            self.params["rotor_speed_max"],
        )
        cmd_tilt = float(np.clip(
            control.get("cmd_tilt_angle", 0.0),
            self.params["tilt_min"],
            self.params["tilt_max"],
        ))
        cmd_surfaces = np.asarray(
            control.get("cmd_surfaces", np.zeros(3)),
            dtype=float,
        )
        cmd_surfaces = np.clip(
            cmd_surfaces,
            self.params["surface_min"],
            self.params["surface_max"],
        )
        return cmd_speed, cmd_tilt, cmd_surfaces

    def _s_dot(self, s: np.ndarray, wind: np.ndarray, control: dict):
        p = self.params
        state = self._unpack_state(s, wind)
        cmd_speed, cmd_tilt, cmd_surfaces = self._control_arrays(control)
        R = rotation_matrix(state["q"])

        rotor_accel = np.clip(
            (cmd_speed - state["rotor_speeds"]) / p["tau_m"],
            -p["rotor_accel_max"],
            p["rotor_accel_max"],
        )
        tilt_rate = np.clip(
            (cmd_tilt - float(state["tilt_angle"])) / p["tau_tilt"],
            -p["tilt_rate_max"],
            p["tilt_rate_max"],
        )
        surface_rate = np.clip(
            (cmd_surfaces - state["surfaces"]) / p["tau_surface"],
            -p["surface_rate_max"],
            p["surface_rate_max"],
        )

        (
            total_force_body,
            total_moment_body,
            aero,
            rotor_force,
            rotor_moment,
        ) = self.compute_wrench(state)
        total_force_world = (
            R @ total_force_body
            + np.array([0.0, 0.0, -self.mass * self.g])
        )
        v_dot = total_force_world / self.mass
        w = state["w"]
        w_dot = self.inv_inertia @ (
            total_moment_body - np.cross(w, self.inertia @ w)
        )

        power_w = (
            p["power_coeff"]
            * float(np.sum(np.maximum(state["rotor_speeds"], 0.0) ** 3))
            + p["avionics_power_w"]
        )
        energy_dot = -power_w if float(state["energy_j"]) > 0.0 else 0.0

        sdot = np.zeros_like(s)
        n = self.num_rotors
        sdot[0:3] = state["v"]
        sdot[3:6] = v_dot
        sdot[6:10] = quat_dot(state["q"], state["w"])
        sdot[10:13] = w_dot
        sdot[13:13+n] = rotor_accel
        sdot[13+n] = tilt_rate
        sdot[14+n:17+n] = surface_rate
        sdot[17+n] = energy_dot

        diagnostics = {
            "airspeed": aero.airspeed,
            "alpha": aero.alpha,
            "beta": aero.beta,
            "qbar": aero.qbar,
            "wing_blend": aero.wing_blend,
            "aero_force_body": aero.force_body,
            "aero_moment_body": aero.moment_body,
            "rotor_force_body": rotor_force,
            "rotor_moment_body": rotor_moment,
            "total_force_body": total_force_body,
            "total_moment_body": total_moment_body,
            "power_w": power_w,
            "coefficients": aero.coefficients,
            "vdot": v_dot,
            "wdot": w_dot,
        }
        return sdot, diagnostics

    def step(self, state: dict, control: dict, t_step: float) -> dict:
        s0 = self._pack_state(state)
        wind = np.asarray(state.get("wind", np.zeros(3)), dtype=float)
        f = lambda s: self._s_dot(s, wind, control)[0]
        h = float(t_step)
        k1 = f(s0)
        k2 = f(s0 + 0.5 * h * k1)
        k3 = f(s0 + 0.5 * h * k2)
        k4 = f(s0 + h * k3)
        s = s0 + (h / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)
        new_state = self._unpack_state(s, wind)
        new_state["q"] = normalize_quat(new_state["q"])
        new_state["rotor_speeds"] = np.clip(
            new_state["rotor_speeds"],
            self.params["rotor_speed_min"],
            self.params["rotor_speed_max"],
        )
        new_state["tilt_angle"] = np.array(np.clip(
            new_state["tilt_angle"],
            self.params["tilt_min"],
            self.params["tilt_max"],
        ))
        new_state["surfaces"] = np.clip(
            new_state["surfaces"],
            self.params["surface_min"],
            self.params["surface_max"],
        )
        new_state["energy_j"] = np.array(
            max(0.0, float(new_state["energy_j"]))
        )

        if self.params.get("ground_enabled", True) and new_state["x"][2] < 0.0:
            new_state["x"][2] = 0.0
            if new_state["v"][2] < 0.0:
                new_state["v"][2] = 0.0
            new_state["v"][:2] *= np.exp(
                -float(self.params.get("ground_friction_rate", 3.0)) * h
            )

        _, self.last_diagnostics = self._s_dot(
            self._pack_state(new_state),
            wind,
            control,
        )
        return new_state

    def statedot(
        self,
        state: dict,
        control: dict,
        t_step: float | None = None,
    ) -> dict:
        sdot, diagnostics = self._s_dot(
            self._pack_state(state),
            state.get("wind", np.zeros(3)),
            control,
        )
        self.last_diagnostics = diagnostics
        return {
            "vdot": sdot[3:6].copy(),
            "wdot": sdot[10:13].copy(),
        }
