"""Final v0.4.5 dashboard: force-feasible comfort and transition semantics.

This layer keeps the v0.4.5 command-envelope/braking/tracking work and adds:

* a comfort-feasible forward-transition nacelle-tilt cap, because a common-axis
  tiltrotor that is still supporting weight can create a large forward thrust
  component even when the requested *net* acceleration is small;
* a back-transition nacelle schedule deliberately decoupled from the much longer
  comfort-limited stopping time;
* jerk-limited re-entry into the comfort acceleration envelope after an altitude
  safety override; and
* a separate ground-speed trace, because airspeed does not approach zero in a
  position-holding hover when wind/gust is present.
"""
from __future__ import annotations

import numpy as np

from . import dashboard as base_dashboard
from . import v045_dashboard as v045
from .allocator import rotor_axis
from .commands import CommandManager, FlightPhase
from .enhanced_dashboard import ComfortAwareSimulation
from .math_utils import clamp_norm, rotation_matrix, smoothstep01
from .vehicle import Tiltrotor


COMFORT_GEOMETRIC_ACCEL_FRACTION = 0.88
COMFORT_TILT_CAP_OPEN_RATE_RAD_S = np.deg2rad(7.0)
BACK_TRANSITION_NACELLE_RETURN_S = 10.0
FINAL_EXTRA_FIELDS = [
    "ground_speed_mps",
    "comfort_tilt_cap_deg",
    "comfort_reentry_active",
]


class ForceFeasibleComfortController(v045.ComfortEnvelopeController):
    """Make comfort commands consistent with the shared-axis thrust geometry.

    Forward transition
    ------------------
    Rotor force that is still required to support weight also produces forward
    thrust when the nacelles are tilted.  The forward-transition cap estimates
    the current aerodynamic force and only permits a nacelle angle whose
    quasi-steady forward acceleration remains inside most of the selected
    comfort envelope.  The remaining margin is left for attitude, actuator and
    aerodynamic transients.

    Back transition
    ---------------
    A negative longitudinal acceleration cannot be realised efficiently while
    the common rotor axis is still strongly tilted forward.  The body can create
    a small backward thrust-vector component with negative pitch only after the
    nacelles are close to vertical.  Therefore nacelle conversion and vehicle
    braking use separate time scales: the nacelles return smoothly from 90 to 0
    deg in about 10 s, while a 15 m/s aircraft with a 0.5 m/s^2 comfort target
    still needs about 30+ s to stop.

    Comfort re-entry
    ----------------
    An altitude-safety override may legitimately leave the acceleration command
    outside the passenger comfort sphere.  When safety clears, instantly
    projecting (for example) 2.0 m/s^2 back to 0.5 m/s^2 would itself create a
    very large command jerk.  Re-entry therefore follows the selected command
    jerk envelope.  During this short recovery the acceleration command can
    remain above the nominal target while monotonically returning toward it.

    Altitude safety remains higher priority than comfort in all cases.
    """

    def reset(self) -> None:
        super().reset()
        self._comfort_tilt_cap_rad = float(self.params["tilt_min"])
        self.comfort_reentry_active = False

    def _comfort_guard_active(self) -> bool:
        return (
            self.max_command_jerk_mps3 is not None
            or float(self.max_accel)
            < self.safety_max_accel_mps2 - 1e-9
        )

    def _desired_velocity_and_acceleration(self, t: float, state: dict):
        previous = self._last_priority_accel_world.copy()
        v_ref, target = super()._desired_velocity_and_acceleration(t, state)
        target = np.asarray(target, dtype=float)
        self.comfort_reentry_active = False

        if not self._comfort_guard_active():
            return v_ref, target

        safety = float(self.safety_override_factor)
        comfort_limit = max(0.05, min(
            float(self.max_accel), self.safety_max_accel_mps2
        ))
        previous_norm = float(np.linalg.norm(previous))
        jerk_target = self.max_command_jerk_mps3

        # Parent v0.4.5 correctly hard-bounds nominal commands, but if the
        # *previous* command came from a safety override it may be outside that
        # sphere.  Replacing it with a point on the 0.5 sphere in one 0.01-s
        # frame would violate the jerk target.  Undo only that radial jump and
        # move from the prior safe command toward the new nominal target at the
        # jerk-limited rate.
        if (
            safety <= v045.NORMAL_SAFETY_RELEASE_THRESHOLD
            and previous_norm > comfort_limit + 1e-6
            and jerk_target is not None
            and jerk_target > 0.0
            and self._control_dt > 0.0
        ):
            internal_jerk = max(
                0.02,
                float(jerk_target) * float(self.command_jerk_headroom),
            )
            delta = clamp_norm(
                target - previous,
                internal_jerk * self._control_dt,
            )
            target = previous + delta
            self._last_priority_accel_world = target.copy()
            self.last_command_jerk_norm_mps3 = float(
                np.linalg.norm(delta) / self._control_dt
            )
            self.comfort_reentry_active = (
                float(np.linalg.norm(target)) > comfort_limit + 1e-6
            )

        return v_ref, target

    def _raw_comfort_tilt_limit(self, state: dict) -> float:
        comfort_limit = max(0.05, min(
            float(self.max_accel), self.safety_max_accel_mps2
        ))
        allowed_forward_accel = (
            COMFORT_GEOMETRIC_ACCEL_FRACTION * comfort_limit
        )

        sp = self.commander.setpoint
        heading = float(sp.heading_rad)
        forward_world = np.array([
            np.cos(heading), np.sin(heading), 0.0
        ])
        R = rotation_matrix(state["q"])
        aero = self.vehicle.aero_wrench_from_state(state)
        aero_world = R @ np.asarray(aero.force_body, dtype=float)
        aero_forward = float(np.dot(aero_world, forward_world))

        # z is positive upward in this model.  For approximately level altitude,
        # the rotors only need to supply the part of weight not already carried
        # by the aerodynamic vertical force.
        vertical_prop_needed = max(
            0.0,
            self.vehicle.mass * self.params["gravity"] - float(aero_world[2]),
        )
        if vertical_prop_needed <= 1e-6:
            return float(self.params["tilt_max"])

        def predicted_forward_accel(tilt: float) -> float:
            axis_world = R @ rotor_axis(float(tilt))
            vertical_effectiveness = float(axis_world[2])
            if vertical_effectiveness <= 0.04:
                return float("inf")
            required_collective = (
                vertical_prop_needed / vertical_effectiveness
            )
            rotor_forward = required_collective * float(
                np.dot(axis_world, forward_world)
            )
            return float(
                (aero_forward + rotor_forward) / self.vehicle.mass
            )

        tilt_min = float(self.params["tilt_min"])
        tilt_max = float(self.params["tilt_max"])
        if predicted_forward_accel(tilt_min) >= allowed_forward_accel:
            return tilt_min
        if predicted_forward_accel(tilt_max) <= allowed_forward_accel:
            return tilt_max

        lo, hi = tilt_min, tilt_max
        for _ in range(28):
            mid = 0.5 * (lo + hi)
            if predicted_forward_accel(mid) <= allowed_forward_accel:
                lo = mid
            else:
                hi = mid
        return float(lo)

    def _back_transition_tilt_command(self, t: float, state: dict) -> float:
        """Return nacelles vertically on a dedicated ~10 s schedule.

        The actuator itself is limited to 18 deg/s.  A 10 s smoothstep has a
        peak requested schedule rate of about 13.5 deg/s, so this trajectory is
        dynamically feasible without simply slamming the tilt actuator into its
        hard rate limit.  Altitude safety can only make the return more vertical.
        """
        sp = self.commander.setpoint
        elapsed_s = max(0.0, float(t) - float(sp.entered_at_s))
        time_progress = float(np.clip(
            elapsed_s / BACK_TRANSITION_NACELLE_RETURN_S,
            0.0,
            1.0,
        ))
        safety = self._altitude_safety_factor(state)
        progress = max(time_progress, float(safety))
        return float(self.params["tilt_max"]) * (
            1.0 - float(smoothstep01(progress))
        )

    def _tilt_command(self, t: float, state: dict) -> float:
        phase = self.commander.setpoint.phase

        if phase == FlightPhase.TRANSITION_TO_HOVER:
            self._comfort_tilt_cap_rad = float(self.params["tilt_max"])
            return self._back_transition_tilt_command(t, state)

        base_cmd = float(super()._tilt_command(t, state))
        if (
            phase != FlightPhase.TRANSITION_TO_CRUISE
            or not self._comfort_guard_active()
            or self._altitude_safety_factor(state)
            > v045.NORMAL_SAFETY_RELEASE_THRESHOLD
        ):
            self._comfort_tilt_cap_rad = float(self.params["tilt_max"])
            return base_cmd

        raw_cap = self._raw_comfort_tilt_limit(state)
        current_tilt = float(state["tilt_angle"])
        previous_cap = max(current_tilt, self._comfort_tilt_cap_rad)
        dt = max(float(self._control_dt), 0.0)

        # Close the cap immediately when current force geometry becomes unsafe,
        # but open it only at a bounded rate as the wing progressively assumes
        # the vertical-load share.  This avoids gust-driven cap chatter.
        if raw_cap < previous_cap:
            filtered_cap = raw_cap
        else:
            filtered_cap = min(
                raw_cap,
                previous_cap + COMFORT_TILT_CAP_OPEN_RATE_RAD_S * dt,
            )
        self._comfort_tilt_cap_rad = float(np.clip(
            filtered_cap,
            self.params["tilt_min"],
            self.params["tilt_max"],
        ))
        return min(base_cmd, self._comfort_tilt_cap_rad)

    def update(self, t: float, state: dict, flat: dict | None = None) -> dict:
        control = super().update(t, state, flat)
        control["comfort_tilt_cap_deg"] = np.array(float(
            np.rad2deg(self._comfort_tilt_cap_rad)
        ))
        control["comfort_reentry_active"] = np.array(float(
            self.comfort_reentry_active
        ))
        return control


def sample_to_record_final(sample, params: dict) -> dict:
    rec = v045.sample_to_record_v045(sample, params)
    rec["ground_speed_mps"] = float(np.linalg.norm(
        np.asarray(sample.state["v"], dtype=float)[:2]
    ))
    rec["comfort_tilt_cap_deg"] = float(
        sample.control.get("comfort_tilt_cap_deg", np.nan)
    )
    rec["comfort_reentry_active"] = float(
        sample.control.get("comfort_reentry_active", 0.0)
    )
    return rec


def _new_simulation() -> ComfortAwareSimulation:
    vehicle = Tiltrotor()
    commander = CommandManager()
    controller = ForceFeasibleComfortController(vehicle, commander)
    return ComfortAwareSimulation(
        vehicle=vehicle,
        commander=commander,
        controller=controller,
        dt=0.01,
    )


def _find_plot(doc, title: str):
    for model in doc.models:
        model_title = getattr(model, "title", None)
        if getattr(model_title, "text", None) == title:
            return model
    raise LookupError(title)


def _plot_source(plot):
    for renderer in getattr(plot, "renderers", []):
        source = getattr(renderer, "data_source", None)
        if source is not None:
            return source
    raise LookupError("plot source")


def build_dashboard(doc, simulation=None):
    sim = _new_simulation() if simulation is None else simulation
    for field in FINAL_EXTRA_FIELDS:
        if field not in base_dashboard.SOURCE_FIELDS:
            base_dashboard.SOURCE_FIELDS.append(field)

    v045.build_dashboard(doc, simulation=sim)

    # Base-dashboard periodic callbacks look this global up when they stream a
    # new sample, so replace it after v045 installs its own tracking wrapper.
    base_dashboard.sample_to_record = sample_to_record_final

    flight_plot = _find_plot(doc, "Flight state")
    source = _plot_source(flight_plot)
    flight_plot.line(
        "time_s",
        "ground_speed_mps",
        source=source,
        legend_label="Ground speed",
        line_width=2.0,
        line_dash="dotdash",
        muted_alpha=0.12,
    )

    actuator_plot = _find_plot(doc, "Tilt and control surfaces")
    actuator_source = _plot_source(actuator_plot)
    actuator_plot.line(
        "time_s",
        "comfort_tilt_cap_deg",
        source=actuator_source,
        legend_label="Comfort tilt cap",
        line_width=1.8,
        line_dash="dotted",
        muted_alpha=0.12,
    )

    return sim
