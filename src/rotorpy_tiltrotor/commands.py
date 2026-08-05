"""Flight phases and command manager for vertical, transition, and cruise flight."""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import numpy as np


class FlightPhase(IntEnum):
    GROUND = 0
    VERTICAL_TAKEOFF = 1
    HOVER = 2
    TRANSITION_TO_CRUISE = 3
    CRUISE = 4
    TRANSITION_TO_HOVER = 5
    VERTICAL_LANDING = 6
    COMPLETE = 7


@dataclass
class FlightSetpoint:
    phase: FlightPhase = FlightPhase.GROUND
    altitude_m: float = 0.0
    airspeed_mps: float = 0.0
    heading_rad: float = 0.0
    hold_x_m: float = 0.0
    hold_y_m: float = 0.0
    transition_duration_s: float = 12.0
    entered_at_s: float = 0.0


class CommandManager:
    def __init__(self) -> None:
        self.setpoint = FlightSetpoint()

    def reset(self) -> None:
        self.setpoint = FlightSetpoint()

    def _set(self, phase: FlightPhase, t: float, **kwargs) -> FlightSetpoint:
        current = self.setpoint
        data = {
            "phase": phase,
            "altitude_m": kwargs.get("altitude_m", current.altitude_m),
            "airspeed_mps": kwargs.get("airspeed_mps", current.airspeed_mps),
            "heading_rad": kwargs.get("heading_rad", current.heading_rad),
            "hold_x_m": kwargs.get("hold_x_m", current.hold_x_m),
            "hold_y_m": kwargs.get("hold_y_m", current.hold_y_m),
            "transition_duration_s": kwargs.get("transition_duration_s", current.transition_duration_s),
            "entered_at_s": float(t),
        }
        self.setpoint = FlightSetpoint(**data)
        return self.setpoint

    def vertical_takeoff(self, t: float, altitude_m: float = 30.0, x_m: float = 0.0, y_m: float = 0.0):
        return self._set(FlightPhase.VERTICAL_TAKEOFF, t, altitude_m=max(0.5,float(altitude_m)), airspeed_mps=0.0, hold_x_m=float(x_m), hold_y_m=float(y_m))

    def hover(self, t: float, altitude_m: float, x_m: float, y_m: float):
        return self._set(FlightPhase.HOVER, t, altitude_m=float(altitude_m), airspeed_mps=0.0, hold_x_m=float(x_m), hold_y_m=float(y_m))

    def transition_to_cruise(self, t: float, altitude_m: float, airspeed_mps: float = 15.0, duration_s: float = 12.0):
        return self._set(FlightPhase.TRANSITION_TO_CRUISE, t, altitude_m=float(altitude_m), airspeed_mps=max(5.0,float(airspeed_mps)), transition_duration_s=max(3.0,float(duration_s)))

    def cruise(self, t: float, altitude_m: float, airspeed_mps: float = 15.0, heading_rad: float = 0.0):
        return self._set(FlightPhase.CRUISE, t, altitude_m=float(altitude_m), airspeed_mps=max(7.0,float(airspeed_mps)), heading_rad=float(heading_rad))

    def transition_to_hover(self, t: float, altitude_m: float, x_m: float, y_m: float, duration_s: float = 14.0):
        return self._set(FlightPhase.TRANSITION_TO_HOVER, t, altitude_m=float(altitude_m), airspeed_mps=0.0, hold_x_m=float(x_m), hold_y_m=float(y_m), transition_duration_s=max(4.0,float(duration_s)))

    def vertical_land(self, t: float, x_m: float, y_m: float):
        return self._set(FlightPhase.VERTICAL_LANDING, t, altitude_m=0.0, airspeed_mps=0.0, hold_x_m=float(x_m), hold_y_m=float(y_m))

    def complete(self, t: float):
        return self._set(FlightPhase.COMPLETE, t, altitude_m=0.0, airspeed_mps=0.0)

    def phase_progress(self, t: float) -> float:
        sp = self.setpoint
        return float(np.clip((t-sp.entered_at_s)/max(sp.transition_duration_s,1e-3),0.0,1.0))


class AutomaticMission:
    def __init__(self, commander: CommandManager, altitude_m: float = 30.0, cruise_speed_mps: float = 15.0, cruise_distance_m: float = 180.0) -> None:
        self.commander = commander
        self.altitude_m = altitude_m
        self.cruise_speed_mps = cruise_speed_mps
        self.cruise_distance_m = cruise_distance_m
        self.started = False
        self.cruise_start_x = None
        self.hover_since = None

    def reset(self) -> None:
        self.started = False
        self.cruise_start_x = None
        self.hover_since = None

    def update(self, t: float, state: dict) -> FlightSetpoint:
        x, v = state["x"], state["v"]
        altitude = float(x[2])
        speed = float(np.linalg.norm(v))
        tilt = float(state["tilt_angle"])
        phase = self.commander.setpoint.phase
        if not self.started:
            self.commander.vertical_takeoff(t, self.altitude_m, x[0], x[1]); self.started = True
            return self.commander.setpoint
        if phase == FlightPhase.VERTICAL_TAKEOFF:
            if altitude > self.altitude_m-0.8 and abs(v[2]) < 0.8:
                self.commander.hover(t, self.altitude_m, x[0], x[1]); self.hover_since = t
        elif phase == FlightPhase.HOVER:
            if self.hover_since is not None and t-self.hover_since > 2.0:
                self.commander.transition_to_cruise(t, self.altitude_m, self.cruise_speed_mps)
        elif phase == FlightPhase.TRANSITION_TO_CRUISE:
            if speed > 0.88*self.cruise_speed_mps and tilt > np.deg2rad(78.0):
                self.commander.cruise(t, self.altitude_m, self.cruise_speed_mps); self.cruise_start_x = float(x[0])
        elif phase == FlightPhase.CRUISE:
            if self.cruise_start_x is not None and x[0]-self.cruise_start_x > self.cruise_distance_m:
                self.commander.transition_to_hover(t, self.altitude_m, x[0]+35.0, x[1])
        elif phase == FlightPhase.TRANSITION_TO_HOVER:
            if speed < 0.75 and tilt < np.deg2rad(6.0):
                self.commander.vertical_land(t, x[0], x[1])
        elif phase == FlightPhase.VERTICAL_LANDING:
            horizontal_speed = float(np.linalg.norm(v[:2]))
            if altitude < 0.15 and abs(v[2]) < 0.15 and horizontal_speed < 0.20:
                self.commander.complete(t)
        return self.commander.setpoint
