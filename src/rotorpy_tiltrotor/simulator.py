"""Standalone real-time-friendly simulation loop around the RotorPy vehicle API."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import numpy as np
import pandas as pd

from .commands import AutomaticMission, CommandManager, FlightPhase
from .controller import TiltrotorController
from .math_utils import euler_deg
from .vehicle import Tiltrotor


@dataclass
class SimulationSample:
    t: float
    state: dict
    control: dict
    diagnostics: dict


ROUTE_TRACKING_PHASES = {
    FlightPhase.TRANSITION_TO_CRUISE,
    FlightPhase.CRUISE,
}


class TiltrotorSimulation:
    def __init__(
        self,
        vehicle: Tiltrotor | None = None,
        commander: CommandManager | None = None,
        controller: TiltrotorController | None = None,
        dt: float = 0.01,
    ) -> None:
        self.vehicle = Tiltrotor() if vehicle is None else vehicle
        self.commander = CommandManager() if commander is None else commander
        self.controller = (
            TiltrotorController(self.vehicle, self.commander)
            if controller is None
            else controller
        )
        self.dt = float(dt)
        self.reset()

    def reset(self, state: dict | None = None) -> None:
        self.t = 0.0
        if hasattr(self.commander, "reset"):
            self.commander.reset()
        if hasattr(self.controller, "reset"):
            self.controller.reset()
        self.vehicle.last_diagnostics = self.vehicle._empty_diagnostics()
        self.state = deepcopy(
            self.vehicle.initial_state if state is None else state
        )
        self.control = self.controller.zero_control()
        self._annotate_control_context()
        self.previous_accel = np.zeros(3)
        self.last_sample = self._make_sample()

    def set_wind(self, wind_world: np.ndarray) -> None:
        self.state["wind"] = np.asarray(
            wind_world,
            dtype=float,
        ).copy()

    def _annotate_control_context(self) -> None:
        """Attach command geometry needed by logging and the dashboard.

        ``hold_x_m``/``hold_y_m`` are a fixed point in vertical and landing
        modes, but they are a route-centreline anchor in transition-to-cruise
        and cruise. Keeping this context separate prevents the fixed route
        anchor from being misreported as a non-moving instantaneous x/y target.
        """
        sp = self.commander.setpoint
        self.control["route_anchor_x"] = np.array(float(sp.hold_x_m))
        self.control["route_anchor_y"] = np.array(float(sp.hold_y_m))
        self.control["route_heading"] = np.array(float(sp.heading_rad))

    def _make_sample(self) -> SimulationSample:
        return SimulationSample(
            self.t,
            deepcopy(self.state),
            deepcopy(self.control),
            deepcopy(self.vehicle.last_diagnostics),
        )

    def step(
        self,
        wind_world: np.ndarray | None = None,
    ) -> SimulationSample:
        if wind_world is not None:
            self.set_wind(wind_world)
        self.control = self.controller.update(
            self.t,
            self.state,
            None,
        )
        self._annotate_control_context()
        self.state = self.vehicle.step(
            self.state,
            self.control,
            self.dt,
        )
        self.t += self.dt
        accel = np.asarray(
            self.vehicle.last_diagnostics.get(
                "vdot",
                np.zeros(3),
            )
        )
        jerk = (
            np.zeros(3)
            if self.t <= self.dt
            else (accel - self.previous_accel) / self.dt
        )
        self.previous_accel = accel
        self.vehicle.last_diagnostics["accel_world"] = accel
        self.vehicle.last_diagnostics["jerk_world"] = jerk
        self.last_sample = self._make_sample()
        return self.last_sample

    def run(
        self,
        duration_s: float,
        mission: AutomaticMission | None = None,
        wind_fn=None,
        record_hz: float = 20.0,
        stop_on_complete: bool = True,
    ) -> pd.DataFrame:
        records = []
        stride = max(
            1,
            int(round(1.0 / (self.dt * record_hz))),
        )
        steps = int(np.ceil(duration_s / self.dt))
        for k in range(steps):
            if mission is not None:
                mission.update(self.t, self.state)
                if (
                    stop_on_complete
                    and self.commander.setpoint.phase
                    == FlightPhase.COMPLETE
                ):
                    self.control = self.controller.zero_control()
                    self.control["phase_code"] = np.array(
                        float(FlightPhase.COMPLETE)
                    )
                    self._annotate_control_context()
                    records.append(sample_to_record(
                        self._make_sample(),
                        self.vehicle.params,
                    ))
                    break
            wind = (
                np.zeros(3)
                if wind_fn is None
                else np.asarray(
                    wind_fn(self.t, self.state),
                    dtype=float,
                )
            )
            sample = self.step(wind)
            if k % stride == 0:
                records.append(sample_to_record(
                    sample,
                    self.vehicle.params,
                ))
            if (
                stop_on_complete
                and self.commander.setpoint.phase
                == FlightPhase.COMPLETE
            ):
                records.append(sample_to_record(
                    sample,
                    self.vehicle.params,
                ))
                break
        return pd.DataFrame.from_records(records)


def _tracking_reference(sample: SimulationSample) -> dict:
    """Return the active point/route target used for position-error display.

    Vertical, hover, back-transition and landing phases use a fixed point target.
    Transition-to-cruise and cruise are *path-following* phases: longitudinal
    motion is controlled by airspeed, while lateral position is controlled with
    cross-track error. Their meaningful instantaneous x/y target is therefore
    the closest point on the commanded route centreline, not the fixed route
    anchor. This makes x/y errors represent the actual route-normal correction
    instead of accumulating travelled distance as a false position error.
    """
    state = sample.state
    control = sample.control
    phase_value = int(round(float(control.get("phase_code", 0.0))))
    try:
        phase = FlightPhase(phase_value)
    except ValueError:
        phase = FlightPhase.GROUND

    actual_xy = np.asarray(state["x"][:2], dtype=float)
    anchor_xy = np.array([
        float(control.get("route_anchor_x", control.get("target_x", 0.0))),
        float(control.get("route_anchor_y", control.get("target_y", 0.0))),
    ])
    heading = float(control.get("route_heading", 0.0))
    forward = np.array([np.cos(heading), np.sin(heading)])
    lateral = np.array([-np.sin(heading), np.cos(heading)])

    from_anchor = actual_xy - anchor_xy
    along_track_distance = float(np.dot(from_anchor, forward))

    if phase in ROUTE_TRACKING_PHASES:
        target_xy = anchor_xy + along_track_distance * forward
        reference_type = 1.0
        along_track_error = 0.0
    else:
        target_xy = anchor_xy
        reference_type = 0.0
        along_track_error = float(np.dot(target_xy - actual_xy, forward))

    position_error = target_xy - actual_xy
    cross_track_error = float(np.dot(position_error, lateral))

    return {
        "target_x_m": float(target_xy[0]),
        "target_y_m": float(target_xy[1]),
        "route_anchor_x_m": float(anchor_xy[0]),
        "route_anchor_y_m": float(anchor_xy[1]),
        "x_error_m": float(position_error[0]),
        "y_error_m": float(position_error[1]),
        "cross_track_error_m": cross_track_error,
        "along_track_error_m": along_track_error,
        "along_track_distance_m": along_track_distance,
        "position_reference_type": reference_type,
    }


def sample_to_record(sample: SimulationSample, params: dict) -> dict:
    s, c, d = sample.state, sample.control, sample.diagnostics
    roll, pitch, yaw = euler_deg(s["q"])
    rotor_rpm = (
        np.asarray(s["rotor_speeds"])
        * 60.0
        / (2.0 * np.pi)
    )
    surfaces_deg = np.rad2deg(s["surfaces"])
    accel = np.asarray(d.get("accel_world", np.zeros(3)))
    jerk = np.asarray(d.get("jerk_world", np.zeros(3)))
    coeff = d.get("coefficients", {})
    tracking = _tracking_reference(sample)
    rec = {
        "time_s": sample.t,
        "phase": int(round(float(c.get("phase_code", 0.0)))),
        "x_m": s["x"][0],
        "y_m": s["x"][1],
        "altitude_m": s["x"][2],
        "vx_mps": s["v"][0],
        "vy_mps": s["v"][1],
        "vz_mps": s["v"][2],
        "airspeed_mps": d.get("airspeed", 0.0),
        "roll_deg": roll,
        "pitch_deg": pitch,
        "yaw_deg": yaw,
        "p_deg_s": np.rad2deg(s["w"][0]),
        "q_deg_s": np.rad2deg(s["w"][1]),
        "r_deg_s": np.rad2deg(s["w"][2]),
        "tilt_deg": np.rad2deg(float(s["tilt_angle"])),
        "aileron_deg": surfaces_deg[0],
        "elevator_deg": surfaces_deg[1],
        "rudder_deg": surfaces_deg[2],
        "power_kw": d.get("power_w", 0.0) / 1000.0,
        "soc_pct": (
            100.0
            * float(s["energy_j"])
            / params["battery_capacity_j"]
        ),
        "ax_mps2": accel[0],
        "ay_mps2": accel[1],
        "az_mps2": accel[2],
        "jerk_mps3": float(np.linalg.norm(jerk)),
        "alpha_deg": np.rad2deg(d.get("alpha", 0.0)),
        "beta_deg": np.rad2deg(d.get("beta", 0.0)),
        "qbar_pa": d.get("qbar", 0.0),
        "wing_blend": d.get("wing_blend", 0.0),
        "CL": coeff.get("CL", 0.0),
        "CD": coeff.get("CD", 0.0),
        "Cl": coeff.get("Cl", 0.0),
        "Cm": coeff.get("Cm", 0.0),
        "Cn": coeff.get("Cn", 0.0),
        "target_altitude_m": float(
            c.get("target_altitude", 0.0)
        ),
        "target_airspeed_mps": float(
            c.get("target_airspeed", 0.0)
        ),
        **tracking,
    }
    for i, rpm in enumerate(rotor_rpm, start=1):
        rec[f"rotor_{i}_rpm"] = rpm
    for name, arr in (
        (
            "aero_force",
            d.get("aero_force_body", np.zeros(3)),
        ),
        (
            "rotor_force",
            d.get("rotor_force_body", np.zeros(3)),
        ),
        (
            "total_moment",
            d.get("total_moment_body", np.zeros(3)),
        ),
    ):
        arr = np.asarray(arr)
        rec[f"{name}_x"] = arr[0]
        rec[f"{name}_y"] = arr[1]
        rec[f"{name}_z"] = arr[2]
    return rec
