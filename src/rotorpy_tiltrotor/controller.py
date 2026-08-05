"""Gain-scheduled controller for vertical, transition, and cruise phases."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from .allocator import rotor_axis, allocate_rotor_thrusts, thrusts_to_speeds
from .commands import CommandManager, FlightPhase
from .math_utils import rotation_matrix, smoothstep01, clamp_norm


class TiltrotorController:
    """Cascaded PID/PD controller with blended rotor and surface moments."""

    def __init__(self, vehicle, commander: CommandManager | None = None):
        self.vehicle = vehicle
        self.params = vehicle.params
        self.commander = CommandManager() if commander is None else commander
        self.k_pos_to_vel_hover = np.array([0.15, 0.15, 0.32])
        self.kp_vel_hover = np.array([0.60, 0.60, 1.25])
        self.ki_pos_hover = np.array([0.030, 0.030, 0.080])
        self.position_integral_limit = np.array([3.0, 3.0, 4.0])
        self.k_speed = 0.85
        self.k_altitude = 0.70
        self.k_vz = 1.35
        self.ki_altitude_transition = 0.060
        self.max_accel = 4.0
        self.k_att = np.array([45.0, 55.0, 38.0])
        self.k_rate = np.array([45.0, 55.0, 38.0])
        self.ki_att = np.array([1.5, 1.8, 1.0])
        self.attitude_integral_limit = np.deg2rad(np.array([8.0, 8.0, 10.0]))
        self.max_moment = np.array([65.0, 75.0, 55.0])
        self.last_control = self.zero_control()
        self.reset()

    def reset(self) -> None:
        self.position_integral = np.zeros(3)
        self.attitude_integral = np.zeros(3)
        self._last_update_time = None
        self._control_dt = 0.0
        self._active_phase = FlightPhase.GROUND
        self._phase_entry_speed = 0.0
        self.last_control = self.zero_control()

    def _update_control_dt(self, t: float) -> None:
        self._control_dt = 0.0 if self._last_update_time is None else float(np.clip(t - self._last_update_time, 0.0, 0.05))
        self._last_update_time = float(t)

    def zero_control(self) -> dict:
        return {
            "cmd_motor_speeds": np.zeros(self.params["num_rotors"]),
            "cmd_tilt_angle": np.array(0.0),
            "cmd_surfaces": np.zeros(3),
            "phase_code": np.array(float(FlightPhase.GROUND)),
            "target_altitude": np.array(0.0),
            "target_airspeed": np.array(0.0),
            "cmd_force_body": np.zeros(3),
            "cmd_moment": np.zeros(3),
        }

    def _tilt_command(self, t: float, state: dict) -> float:
        sp = self.commander.setpoint
        phase = sp.phase
        if phase in (FlightPhase.GROUND, FlightPhase.VERTICAL_TAKEOFF, FlightPhase.HOVER, FlightPhase.VERTICAL_LANDING, FlightPhase.COMPLETE):
            return 0.0
        if phase == FlightPhase.CRUISE:
            return self.params["tilt_max"]
        elapsed_progress = self.commander.phase_progress(t)
        speed = float(np.linalg.norm(state["v"][:2]))
        if phase == FlightPhase.TRANSITION_TO_CRUISE:
            speed_progress = np.clip(speed / max(sp.airspeed_mps, 1.0), 0.0, 1.0)
            progress = max(elapsed_progress, 0.75 * speed_progress)
            return self.params["tilt_max"] * smoothstep01(progress)
        if phase == FlightPhase.TRANSITION_TO_HOVER:
            speed_progress = 1.0 - np.clip(speed / 12.0, 0.0, 1.0)
            progress = max(elapsed_progress, 0.65 * speed_progress)
            return self.params["tilt_max"] * (1.0 - smoothstep01(progress) ** 0.62)
        return 0.0

    def _update_phase_memory(self, state: dict) -> None:
        phase = self.commander.setpoint.phase
        if phase != self._active_phase:
            sp = self.commander.setpoint
            forward = np.array([np.cos(sp.heading_rad), np.sin(sp.heading_rad), 0.0])
            self._phase_entry_speed = max(0.0, float(np.dot(state["v"], forward)))
            self.position_integral[:] = 0.0
            self.attitude_integral[:] = 0.0
            self._active_phase = phase

    def _desired_velocity_and_acceleration(self, t: float, state: dict):
        sp = self.commander.setpoint
        phase = sp.phase
        x, v = state["x"], state["v"]
        if phase in (FlightPhase.GROUND, FlightPhase.COMPLETE):
            return np.zeros(3), np.zeros(3)
        if phase in (FlightPhase.VERTICAL_TAKEOFF, FlightPhase.HOVER, FlightPhase.VERTICAL_LANDING):
            pos_ref = np.array([sp.hold_x_m, sp.hold_y_m, sp.altitude_m])
            pos_error = pos_ref - x
            self.position_integral += pos_error * self._control_dt
            self.position_integral = np.clip(self.position_integral, -self.position_integral_limit, self.position_integral_limit)
            if phase == FlightPhase.VERTICAL_LANDING:
                vz_ref = np.clip(self.k_pos_to_vel_hover[2] * pos_error[2], -0.85, 0.0)
                az_limit = 1.25
            else:
                vz_ref = np.clip(self.k_pos_to_vel_hover[2] * pos_error[2], -1.1, 1.7)
                az_limit = 1.8
            v_ref = np.array([
                np.clip(self.k_pos_to_vel_hover[0] * pos_error[0], -1.2, 1.2),
                np.clip(self.k_pos_to_vel_hover[1] * pos_error[1], -1.2, 1.2),
                vz_ref,
            ])
            velocity_error = v_ref - v
            a_xy = self.kp_vel_hover[:2] * velocity_error[:2] + self.ki_pos_hover[:2] * self.position_integral[:2]
            a_xy = clamp_norm(a_xy, 0.9)
            a_z = float(np.clip(self.kp_vel_hover[2] * velocity_error[2] + self.ki_pos_hover[2] * self.position_integral[2], -az_limit, az_limit))
            return v_ref, np.array([a_xy[0], a_xy[1], a_z])
        forward = np.array([np.cos(sp.heading_rad), np.sin(sp.heading_rad), 0.0])
        lateral = np.array([-np.sin(sp.heading_rad), np.cos(sp.heading_rad), 0.0])
        forward_speed = float(np.dot(v, forward))
        lateral_speed = float(np.dot(v, lateral))
        altitude_error = sp.altitude_m - x[2]
        self.position_integral[2] = float(np.clip(self.position_integral[2] + altitude_error * self._control_dt, -self.position_integral_limit[2], self.position_integral_limit[2]))
        if phase == FlightPhase.TRANSITION_TO_HOVER:
            progress = self.commander.phase_progress(t)
            decel_progress = smoothstep01(max(0.0, (progress - 0.10) / 0.90))
            speed_ref = self._phase_entry_speed * (1.0 - decel_progress)
            vz_ref = np.clip(0.92 * altitude_error, -1.8, 1.8)
            speed_gain, vz_gain = 0.58, 1.70
        else:
            speed_ref = sp.airspeed_mps
            vz_ref = np.clip(self.k_altitude * altitude_error, -2.0, 2.0)
            speed_gain, vz_gain = self.k_speed, self.k_vz
        speed_error = speed_ref - forward_speed
        v_ref = speed_ref * forward + np.array([0.0, 0.0, vz_ref])
        a_cmd = speed_gain * speed_error * forward - 0.9 * lateral_speed * lateral + np.array([0.0, 0.0, vz_gain * (vz_ref - v[2]) + self.ki_altitude_transition * self.position_integral[2]])
        return v_ref, clamp_norm(a_cmd, self.max_accel)

    def _desired_attitude(self, state: dict, a_cmd_world: np.ndarray) -> Rotation:
        sp = self.commander.setpoint
        forward = np.array([np.cos(sp.heading_rad), np.sin(sp.heading_rad), 0.0])
        lateral = np.array([-np.sin(sp.heading_rad), np.cos(sp.heading_rad), 0.0])
        lateral_accel = float(np.dot(a_cmd_world, lateral))
        forward_accel = float(np.dot(a_cmd_world, forward))
        vertical_rotor_fraction = float(np.cos(float(state["tilt_angle"])) ** 2)
        wing_fraction = 1.0 - vertical_rotor_fraction
        roll_cmd = float(np.clip(-vertical_rotor_fraction * lateral_accel / self.params["gravity"], -0.16, 0.16))
        pitch_cmd = vertical_rotor_fraction * forward_accel / self.params["gravity"] - 0.75 * wing_fraction * a_cmd_world[2] / self.params["gravity"]
        pitch_cmd = float(np.clip(pitch_cmd, -0.18, 0.18))
        return Rotation.from_euler("xyz", [roll_cmd, pitch_cmd, sp.heading_rad])

    def _surface_commands(self, desired_moment: np.ndarray, qbar: float):
        p = self.params
        q_low = 0.5 * p["rho"] * 6.0 ** 2
        q_high = 0.5 * p["rho"] * 14.0 ** 2
        blend = float(smoothstep01((qbar - q_low) / max(q_high - q_low, 1e-6)))
        effectiveness = np.array([qbar*p["S"]*p["b"]*p["Cl_da"], qbar*p["S"]*p["c"]*p["Cm_de"], qbar*p["S"]*p["b"]*p["Cn_dr"]])
        requested_surface_moment = blend * desired_moment
        cmd = np.zeros(3)
        valid = np.abs(effectiveness) > 1e-6
        cmd[valid] = requested_surface_moment[valid] / effectiveness[valid]
        cmd = np.clip(cmd, p["surface_min"], p["surface_max"])
        return cmd, effectiveness * cmd, blend

    def update(self, t: float, state: dict, flat: dict | None = None) -> dict:
        sp = self.commander.setpoint
        if sp.phase in (FlightPhase.GROUND, FlightPhase.COMPLETE):
            control = self.zero_control(); control["phase_code"] = np.array(float(sp.phase)); self.last_control = control; return control
        p = self.params
        R = rotation_matrix(state["q"])
        self._update_control_dt(t)
        self._update_phase_memory(state)
        _, a_cmd_world = self._desired_velocity_and_acceleration(t, state)
        aero = self.vehicle.aero_wrench_from_state(state)
        required_prop_world = self.vehicle.mass * (a_cmd_world + np.array([0.0, 0.0, p["gravity"]])) - R @ aero.force_body
        required_prop_body = R.T @ required_prop_world
        tilt_cmd = self._tilt_command(t, state)
        axis = rotor_axis(float(state["tilt_angle"]))
        collective = max(0.0, float(np.dot(required_prop_body, axis)))
        if sp.phase in (FlightPhase.VERTICAL_TAKEOFF, FlightPhase.HOVER, FlightPhase.TRANSITION_TO_HOVER, FlightPhase.VERTICAL_LANDING) and axis[2] > 0.18:
            guard = float(smoothstep01((axis[2] - 0.18) / 0.55))
            vertical_collective = max(0.0, required_prop_body[2]) / max(axis[2], 0.18)
            collective = max(collective, guard * vertical_collective)
        desired_rotation = self._desired_attitude(state, a_cmd_world)
        current_rotation = Rotation.from_quat(state["q"])
        attitude_error = (desired_rotation.inv() * current_rotation).as_rotvec()
        self.attitude_integral += attitude_error * self._control_dt
        self.attitude_integral = np.clip(self.attitude_integral, -self.attitude_integral_limit, self.attitude_integral_limit)
        desired_moment = -self.k_att * attitude_error - self.k_rate * state["w"] - self.ki_att * self.attitude_integral
        desired_moment = np.clip(desired_moment, -self.max_moment, self.max_moment)
        surface_cmd, surface_moment, _ = self._surface_commands(desired_moment, aero.qbar)
        thrusts = allocate_rotor_thrusts(collective, desired_moment - surface_moment, float(state["tilt_angle"]), p)
        cmd_speeds = thrusts_to_speeds(thrusts, p)
        if sp.phase == FlightPhase.VERTICAL_LANDING and state["x"][2] < 0.08 and abs(state["v"][2]) < 0.15:
            cmd_speeds *= 0.0
        control = {
            "cmd_motor_speeds": cmd_speeds,
            "cmd_tilt_angle": np.array(tilt_cmd),
            "cmd_surfaces": surface_cmd,
            "phase_code": np.array(float(sp.phase)),
            "target_altitude": np.array(sp.altitude_m),
            "target_airspeed": np.array(sp.airspeed_mps),
            "cmd_force_body": required_prop_body,
            "cmd_moment": desired_moment,
        }
        self.last_control = control
        return control
