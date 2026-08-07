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
    """Stateful command API used by both scripts and the live dashboard."""

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
            "transition_duration_s": kwargs.get(
                "transition_duration_s", current.transition_duration_s
            ),
            "entered_at_s": float(t),
        }
        self.setpoint = FlightSetpoint(**data)
        return self.setpoint

    def vertical_takeoff(
        self,
        t: float,
        altitude_m: float = 30.0,
        x_m: float = 0.0,
        y_m: float = 0.0,
    ):
        return self._set(
            FlightPhase.VERTICAL_TAKEOFF,
            t,
            altitude_m=max(0.5, float(altitude_m)),
            airspeed_mps=0.0,
            hold_x_m=float(x_m),
            hold_y_m=float(y_m),
        )

    def hover(self, t: float, altitude_m: float, x_m: float, y_m: float):
        return self._set(
            FlightPhase.HOVER,
            t,
            altitude_m=float(altitude_m),
            airspeed_mps=0.0,
            hold_x_m=float(x_m),
            hold_y_m=float(y_m),
        )

    def transition_to_cruise(
        self,
        t: float,
        altitude_m: float,
        airspeed_mps: float = 15.0,
        duration_s: float = 12.0,
        heading_rad: float | None = None,
        route_x_m: float | None = None,
        route_y_m: float | None = None,
    ):
        return self._set(
            FlightPhase.TRANSITION_TO_CRUISE,
            t,
            altitude_m=float(altitude_m),
            airspeed_mps=max(5.0, float(airspeed_mps)),
            heading_rad=(
                self.setpoint.heading_rad
                if heading_rad is None
                else float(heading_rad)
            ),
            hold_x_m=(
                self.setpoint.hold_x_m
                if route_x_m is None
                else float(route_x_m)
            ),
            hold_y_m=(
                self.setpoint.hold_y_m
                if route_y_m is None
                else float(route_y_m)
            ),
            transition_duration_s=max(3.0, float(duration_s)),
        )

    def cruise(
        self,
        t: float,
        altitude_m: float,
        airspeed_mps: float = 15.0,
        heading_rad: float = 0.0,
        route_x_m: float | None = None,
        route_y_m: float | None = None,
    ):
        return self._set(
            FlightPhase.CRUISE,
            t,
            altitude_m=float(altitude_m),
            airspeed_mps=max(7.0, float(airspeed_mps)),
            heading_rad=float(heading_rad),
            hold_x_m=(
                self.setpoint.hold_x_m
                if route_x_m is None
                else float(route_x_m)
            ),
            hold_y_m=(
                self.setpoint.hold_y_m
                if route_y_m is None
                else float(route_y_m)
            ),
        )

    def transition_to_hover(
        self,
        t: float,
        altitude_m: float,
        x_m: float,
        y_m: float,
        duration_s: float = 14.0,
    ):
        return self._set(
            FlightPhase.TRANSITION_TO_HOVER,
            t,
            altitude_m=float(altitude_m),
            airspeed_mps=0.0,
            hold_x_m=float(x_m),
            hold_y_m=float(y_m),
            transition_duration_s=max(4.0, float(duration_s)),
        )

    def vertical_land(self, t: float, x_m: float, y_m: float):
        return self._set(
            FlightPhase.VERTICAL_LANDING,
            t,
            altitude_m=0.0,
            airspeed_mps=0.0,
            hold_x_m=float(x_m),
            hold_y_m=float(y_m),
        )

    def complete(self, t: float):
        return self._set(
            FlightPhase.COMPLETE,
            t,
            altitude_m=0.0,
            airspeed_mps=0.0,
        )

    def phase_progress(self, t: float) -> float:
        sp = self.setpoint
        return float(np.clip(
            (t - sp.entered_at_s) / max(sp.transition_duration_s, 1e-3),
            0.0,
            1.0,
        ))


class AutomaticMission:
    """Trigger-based demonstration mission with safety-priority transitions.

    The original takeoff x/y coordinates are captured once and remain the
    hover and route-centreline reference.  v0.4.4 also uses a physically sized
    stopping point for automatic back transition, inserts a real post-transition
    hover before landing, and does not declare cruise while altitude is badly
    degraded.
    """

    def __init__(
        self,
        commander: CommandManager,
        altitude_m: float = 30.0,
        cruise_speed_mps: float = 15.0,
        cruise_distance_m: float = 180.0,
        back_transition_duration_s: float = 24.0,
        back_transition_decel_mps2: float = 0.75,
        post_transition_hover_s: float = 2.0,
    ) -> None:
        self.commander = commander
        self.altitude_m = altitude_m
        self.cruise_speed_mps = cruise_speed_mps
        self.cruise_distance_m = cruise_distance_m
        self.back_transition_duration_s = max(
            8.0, float(back_transition_duration_s)
        )
        self.back_transition_decel_mps2 = max(
            0.25, float(back_transition_decel_mps2)
        )
        self.post_transition_hover_s = max(
            0.5, float(post_transition_hover_s)
        )
        self.started = False
        self.cruise_start_x = None
        self.hover_since = None
        self.takeoff_x = None
        self.takeoff_y = None
        self.landing_x = None
        self.landing_y = None
        self.returning_to_land = False
        self.back_hover_since = None

    def reset(self) -> None:
        self.started = False
        self.cruise_start_x = None
        self.hover_since = None
        self.takeoff_x = None
        self.takeoff_y = None
        self.landing_x = None
        self.landing_y = None
        self.returning_to_land = False
        self.back_hover_since = None

    def _planned_back_transition_target(self, state: dict) -> tuple[float, float]:
        sp = self.commander.setpoint
        heading = float(sp.heading_rad)
        forward = np.array([np.cos(heading), np.sin(heading)], dtype=float)
        anchor = np.array([self.takeoff_x, self.takeoff_y], dtype=float)
        position = np.asarray(state["x"], dtype=float)[:2]
        velocity = np.asarray(state["v"], dtype=float)[:2]
        along = float(np.dot(position - anchor, forward))
        route_projection = anchor + along * forward
        forward_speed = max(0.0, float(np.dot(velocity, forward)))
        schedule_distance = (
            0.5 * forward_speed * self.back_transition_duration_s
        )
        braking_distance = (
            forward_speed * forward_speed
            / (2.0 * self.back_transition_decel_mps2)
        )
        forward_distance = max(8.0, schedule_distance, braking_distance) + 5.0
        target = route_projection + forward_distance * forward
        return float(target[0]), float(target[1])

    def update(self, t: float, state: dict) -> FlightSetpoint:
        x, v = state["x"], state["v"]
        altitude = float(x[2])
        speed = float(np.linalg.norm(v))
        horizontal_speed = float(np.linalg.norm(v[:2]))
        tilt = float(state["tilt_angle"])
        phase = self.commander.setpoint.phase

        if not self.started:
            self.takeoff_x = float(x[0])
            self.takeoff_y = float(x[1])
            self.commander.vertical_takeoff(
                t,
                self.altitude_m,
                self.takeoff_x,
                self.takeoff_y,
            )
            self.started = True
            return self.commander.setpoint

        if phase == FlightPhase.VERTICAL_TAKEOFF:
            if altitude > self.altitude_m - 0.8 and abs(v[2]) < 0.8:
                self.commander.hover(
                    t,
                    self.altitude_m,
                    self.takeoff_x,
                    self.takeoff_y,
                )
                self.hover_since = t

        elif phase == FlightPhase.HOVER:
            if self.returning_to_land:
                altitude_ready = (
                    abs(altitude - self.altitude_m) < 0.6
                    and abs(float(v[2])) < 0.45
                    and horizontal_speed < 0.85
                )
                if altitude_ready:
                    if self.back_hover_since is None:
                        self.back_hover_since = float(t)
                    elif t - self.back_hover_since >= self.post_transition_hover_s:
                        self.commander.vertical_land(
                            t,
                            self.landing_x,
                            self.landing_y,
                        )
                else:
                    self.back_hover_since = None
            elif self.hover_since is not None and t - self.hover_since > 2.0:
                self.commander.transition_to_cruise(
                    t,
                    self.altitude_m,
                    self.cruise_speed_mps,
                    route_x_m=self.takeoff_x,
                    route_y_m=self.takeoff_y,
                )

        elif phase == FlightPhase.TRANSITION_TO_CRUISE:
            altitude_ready = altitude > self.altitude_m - 1.5
            if (
                speed > 0.88 * self.cruise_speed_mps
                and tilt > np.deg2rad(78.0)
                and altitude_ready
            ):
                self.commander.cruise(
                    t,
                    self.altitude_m,
                    self.cruise_speed_mps,
                    heading_rad=self.commander.setpoint.heading_rad,
                    route_x_m=self.takeoff_x,
                    route_y_m=self.takeoff_y,
                )
                self.cruise_start_x = float(x[0])

        elif phase == FlightPhase.CRUISE:
            if (
                self.cruise_start_x is not None
                and x[0] - self.cruise_start_x > self.cruise_distance_m
            ):
                self.landing_x, self.landing_y = (
                    self._planned_back_transition_target(state)
                )
                self.commander.transition_to_hover(
                    t,
                    self.altitude_m,
                    self.landing_x,
                    self.landing_y,
                    duration_s=self.back_transition_duration_s,
                )

        elif phase == FlightPhase.TRANSITION_TO_HOVER:
            if speed < 0.75 and tilt < np.deg2rad(6.0):
                sp = self.commander.setpoint
                self.landing_x = float(sp.hold_x_m)
                self.landing_y = float(sp.hold_y_m)
                self.returning_to_land = True
                self.back_hover_since = None
                self.commander.hover(
                    t,
                    self.altitude_m,
                    self.landing_x,
                    self.landing_y,
                )

        elif phase == FlightPhase.VERTICAL_LANDING:
            if (
                altitude < 0.15
                and abs(v[2]) < 0.15
                and horizontal_speed < 0.20
            ):
                self.commander.complete(t)

        return self.commander.setpoint
