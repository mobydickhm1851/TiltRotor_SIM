"""RotorPy trajectory adapter and standard demonstration mission helpers."""
from __future__ import annotations

import numpy as np

from .commands import CommandManager


class MissionTrajectory:
    """Minimal RotorPy-compatible trajectory object."""

    def __init__(self, commander: CommandManager):
        self.commander = commander

    def update(self, t: float) -> dict:
        sp = self.commander.setpoint
        x = np.array(
            [sp.hold_x_m, sp.hold_y_m, sp.altitude_m],
            dtype=float,
        )
        v = sp.airspeed_mps * np.array([
            np.cos(sp.heading_rad),
            np.sin(sp.heading_rad),
            0.0,
        ])
        return {
            "x": x,
            "x_dot": v,
            "x_ddot": np.zeros(3),
            "x_dddot": np.zeros(3),
            "x_ddddot": np.zeros(3),
            "yaw": np.array(sp.heading_rad),
            "yaw_dot": np.array(0.0),
            "phase_code": np.array(float(sp.phase)),
            "target_altitude": np.array(sp.altitude_m),
            "target_airspeed": np.array(sp.airspeed_mps),
            "target_x": np.array(sp.hold_x_m),
            "target_y": np.array(sp.hold_y_m),
        }
