"""v0.4.5 comfort-envelope, braking-priority and tracking-display update.

The passenger-comfort values entered in the dashboard are engineering targets,
not physically enforceable disturbance bounds.  v0.4.5 removes avoidable
controller-induced exceedances by enforcing the selected acceleration target as
a true 3-D command-norm envelope whenever altitude safety is not overriding it.
It also gives longitudinal braking priority during back transition and keeps the
position plot on a route-projection reference until the aircraft has actually
reached hover, so a distant planned stop point is not misreported as a sudden
x-tracking error.
"""
from __future__ import annotations

import numpy as np
from bokeh.models import Div

from . import dashboard as base_dashboard
from . import simulator as simulator_module
from . import v044_dashboard as v044
from .commands import AutomaticMission, CommandManager, FlightPhase
from .controller import TiltrotorController
from .enhanced_dashboard import ComfortAwareSimulation
from .math_utils import clamp_norm, smoothstep01
from .vehicle import Tiltrotor


BACK_TRANSITION_BRAKE_FRACTION = 0.90
BACK_TRANSITION_VERTICAL_NOMINAL_FRACTION = 0.35
MEASURED_ACCEL_FILTER_TAU_S = 0.18
MEASURED_ACCEL_FEEDBACK_GAIN = 0.30
NORMAL_SAFETY_RELEASE_THRESHOLD = 0.05
TILT_SCHEDULE_TIME_FRACTION = 0.55

_EXTRA_SOURCE_FIELDS = [
    "distance_to_stop_m",
    "comfort_command_accel_mps2",
    "comfort_command_jerk_mps3",
    "altitude_safety_factor",
]

_ORIGINAL_SAMPLE_TO_RECORD = simulator_module.sample_to_record
_ORIGINAL_PLANNED_HOVER_TARGET = base_dashboard.planned_hover_target


def _route_axes(heading_rad: float) -> tuple[np.ndarray, np.ndarray]:
    forward = np.array([np.cos(heading_rad), np.sin(heading_rad)], dtype=float)
    lateral = np.array([-np.sin(heading_rad), np.cos(heading_rad)], dtype=float)
    return forward, lateral


class ComfortEnvelopeController(v044.SafetyPriorityController):
    """Safety-priority controller with a real nominal 3-D comfort envelope.

    v0.4.4 limited horizontal acceleration as a 2-D norm while independently
    allowing the vertical command to reach the same comfort value.  Therefore
    the commanded 3-D norm could exceed the user target even without a safety
    override.  v0.4.5 first constructs one nominal 3-D comfort vector and only
    adds extra vertical authority when altitude safety genuinely requires it.

    During transition-to-hover the nominal comfort budget is allocated in this
    order: altitude keeping, longitudinal braking, lateral route correction.
    This uses the available comfort authority for stopping instead of letting
    cross-track control consume most of the deceleration budget.
    """

    def reset(self) -> None:
        super().reset()
        self._filtered_measured_accel_world = np.zeros(3)
        self.last_command_jerk_norm_mps3 = 0.0

    def _filtered_measured_accel(self) -> np.ndarray:
        measured = np.asarray(
            self.vehicle.last_diagnostics.get("accel_world", np.zeros(3)),
            dtype=float,
        )
        dt = max(float(self._control_dt), 0.0)
        if dt <= 0.0:
            return self._filtered_measured_accel_world.copy()
        alpha = float(np.clip(
            dt / (MEASURED_ACCEL_FILTER_TAU_S + dt), 0.0, 1.0
        ))
        self._filtered_measured_accel_world += alpha * (
            measured - self._filtered_measured_accel_world
        )
        return self._filtered_measured_accel_world.copy()

    def _back_transition_comfort_vector(
        self,
        raw_accel: np.ndarray,
        state: dict,
        comfort_limit: float,
    ) -> np.ndarray:
        """Allocate the nominal 3-D comfort budget with braking priority."""
        sp = self.commander.setpoint
        forward_xy, lateral_xy = _route_axes(float(sp.heading_rad))
        velocity_xy = np.asarray(state["v"], dtype=float)[:2]
        forward_speed = float(np.dot(velocity_xy, forward_xy))

        # Small altitude corrections remain available without consuming the
        # entire comfort budget.  If altitude actually becomes unsafe the
        # safety blend below releases additional vertical authority.
        z_cap = BACK_TRANSITION_VERTICAL_NOMINAL_FRACTION * comfort_limit
        vertical = float(np.clip(raw_accel[2], -z_cap, z_cap))
        xy_budget = float(np.sqrt(max(
            0.0, comfort_limit * comfort_limit - vertical * vertical
        )))

        if forward_speed > 0.35 and xy_budget > 1e-9:
            braking = min(
                BACK_TRANSITION_BRAKE_FRACTION * comfort_limit,
                xy_budget,
            )
            forward_accel = -braking
            lateral_budget = float(np.sqrt(max(
                0.0, xy_budget * xy_budget - forward_accel * forward_accel
            )))
            raw_lateral = float(np.dot(raw_accel[:2], lateral_xy))
            lateral_accel = float(np.clip(
                raw_lateral, -lateral_budget, lateral_budget
            ))
            horizontal = (
                forward_accel * forward_xy
                + lateral_accel * lateral_xy
            )
            return np.array([horizontal[0], horizontal[1], vertical])

        # Once nearly stopped, point capture is more important than forcing a
        # negative along-track command, so return to the generic norm envelope.
        return clamp_norm(np.asarray(raw_accel, dtype=float), comfort_limit)

    def _nominal_comfort_vector(
        self,
        raw_accel: np.ndarray,
        state: dict,
        comfort_limit: float,
    ) -> np.ndarray:
        if self.commander.setpoint.phase == FlightPhase.TRANSITION_TO_HOVER:
            return self._back_transition_comfort_vector(
                raw_accel, state, comfort_limit
            )
        return clamp_norm(np.asarray(raw_accel, dtype=float), comfort_limit)

    def _desired_velocity_and_acceleration(self, t: float, state: dict):
        configured_limit = float(self.max_accel)
        guard_active = (
            self.max_command_jerk_mps3 is not None
            or configured_limit < self.safety_max_accel_mps2 - 1e-9
        )

        # Compute the unrestricted flight-control request first.  The selected
        # comfort target is applied below as an envelope rather than replacing
        # the aircraft's safety-critical control authority.
        self.max_accel = self.safety_max_accel_mps2
        try:
            v_ref, raw_accel = (
                TiltrotorController._desired_velocity_and_acceleration(
                    self, t, state
                )
            )
        finally:
            self.max_accel = configured_limit

        raw_accel = np.asarray(raw_accel, dtype=float)
        safety = self._altitude_safety_factor(state)
        self.safety_override_factor = safety
        self.safety_override_active = safety > NORMAL_SAFETY_RELEASE_THRESHOLD

        tracking_scale = (
            1.0
            - safety * (1.0 - v044.SAFETY_HORIZONTAL_RETAIN_FRACTION)
        )

        previous = self._last_priority_accel_world.copy()

        if not guard_active:
            target = np.array([
                raw_accel[0] * tracking_scale,
                raw_accel[1] * tracking_scale,
                raw_accel[2],
            ])
            self._last_priority_accel_world = target.copy()
            if self._control_dt > 0.0:
                self.last_command_jerk_norm_mps3 = float(
                    np.linalg.norm(target - previous) / self._control_dt
                )
            else:
                self.last_command_jerk_norm_mps3 = 0.0
            return v_ref, target

        comfort_limit = max(0.05, min(
            configured_limit,
            self.safety_max_accel_mps2,
        ))

        nominal = self._nominal_comfort_vector(
            raw_accel, state, comfort_limit
        )
        nominal[:2] *= tracking_scale

        # Only altitude safety may add authority outside the nominal 3-D comfort
        # sphere.  At safety=0, ||target|| is guaranteed <= comfort_limit.
        vertical = (
            (1.0 - safety) * float(nominal[2])
            + safety * float(raw_accel[2])
        )
        target = np.array([
            float(nominal[0]),
            float(nominal[1]),
            vertical,
        ])

        # Use the previous measured aircraft acceleration as a slow plant-response
        # correction.  This does not pretend to cancel an instantaneous gust;
        # it only removes avoidable persistent overshoot caused by actuator and
        # closed-loop lag.  The correction is disabled during safety recovery.
        measured = self._filtered_measured_accel()
        measured_norm = float(np.linalg.norm(measured))
        if (
            safety <= NORMAL_SAFETY_RELEASE_THRESHOLD
            and measured_norm > 1.02 * comfort_limit
        ):
            allowed_measured = clamp_norm(measured, comfort_limit)
            excess = measured - allowed_measured
            target = clamp_norm(
                target - MEASURED_ACCEL_FEEDBACK_GAIN * excess,
                comfort_limit,
            )

        jerk_target = self.max_command_jerk_mps3
        if (
            jerk_target is not None
            and jerk_target > 0.0
            and self._control_dt > 0.0
        ):
            internal_jerk = max(
                0.02,
                float(jerk_target) * float(self.command_jerk_headroom),
            )
            delta = target - previous

            if safety <= NORMAL_SAFETY_RELEASE_THRESHOLD:
                # A true 3-D command-jerk norm limit in nominal comfort flight.
                delta = clamp_norm(
                    delta, internal_jerk * self._control_dt
                )
                target = previous + delta
                target = clamp_norm(target, comfort_limit)
            else:
                # Keep the nominal 3-D slew shaping, then allow only the
                # vertical channel to accelerate faster for altitude recovery.
                nominal_delta = clamp_norm(
                    delta, internal_jerk * self._control_dt
                )
                limited = previous + nominal_delta
                vertical_jerk_limit = (
                    (1.0 - safety) * internal_jerk
                    + safety * self.emergency_vertical_jerk_mps3
                )
                limited[2] = previous[2] + float(np.clip(
                    delta[2],
                    -vertical_jerk_limit * self._control_dt,
                    vertical_jerk_limit * self._control_dt,
                ))
                target = limited

        # Nominal flight is one 3-D sphere.  A safety override may exceed that
        # sphere only through the vertical channel, while horizontal tracking
        # has already been faded toward zero.
        if safety <= NORMAL_SAFETY_RELEASE_THRESHOLD:
            target = clamp_norm(target, comfort_limit)
        else:
            target[:2] = clamp_norm(target[:2], comfort_limit)
            target[2] = float(np.clip(
                target[2],
                -self.safety_max_accel_mps2,
                self.safety_max_accel_mps2,
            ))

        self._last_priority_accel_world = target.copy()
        if self._control_dt > 0.0:
            self.last_command_jerk_norm_mps3 = float(
                np.linalg.norm(target - previous) / self._control_dt
            )
        else:
            self.last_command_jerk_norm_mps3 = 0.0
        return v_ref, target

    def _tilt_command(self, t: float, state: dict) -> float:
        phase = self.commander.setpoint.phase
        if phase != FlightPhase.TRANSITION_TO_HOVER:
            return super()._tilt_command(t, state)

        tilt_max = float(self.params["tilt_max"])
        elapsed = self.commander.phase_progress(t)
        heading = float(self.commander.setpoint.heading_rad)
        forward_xy, _ = _route_axes(heading)
        forward_speed = max(
            0.0,
            float(np.dot(
                np.asarray(state["v"], dtype=float)[:2], forward_xy
            )),
        )
        reference_speed = max(self._phase_entry_speed, 12.0)
        speed_progress = 1.0 - float(np.clip(
            forward_speed / reference_speed, 0.0, 1.0
        ))
        safety = self._altitude_safety_factor(state)

        # Do not let elapsed time alone force the nacelles fully vertical while
        # the aircraft is still fast.  Speed reduction owns the final portion of
        # the schedule; altitude safety may still accelerate vertical conversion.
        progress = max(
            TILT_SCHEDULE_TIME_FRACTION * elapsed,
            speed_progress,
            0.90 * safety,
        )
        return tilt_max * (
            1.0 - float(smoothstep01(progress)) ** 0.62
        )

    def update(self, t: float, state: dict, flat: dict | None = None) -> dict:
        control = super().update(t, state, flat)
        control["comfort_command_accel_mps2"] = np.array(float(
            np.linalg.norm(self._last_priority_accel_world)
        ))
        control["comfort_command_jerk_mps3"] = np.array(float(
            self.last_command_jerk_norm_mps3
        ))
        control["altitude_safety_factor"] = np.array(float(
            self.safety_override_factor
        ))
        return control


class GuardAwareAutomaticMission(AutomaticMission):
    """Automatic mission whose stop planning follows the active comfort guard."""

    def __init__(self, *args, controller=None, **kwargs):
        self.controller = controller
        super().__init__(*args, **kwargs)

    def _effective_back_decel(self) -> float:
        nominal = float(self.back_transition_decel_mps2)
        if self.controller is None:
            return nominal
        configured = float(getattr(
            self.controller, "max_accel", v044.SAFETY_MAX_ACCEL_MPS2
        ))
        if configured >= v044.SAFETY_MAX_ACCEL_MPS2 - 1e-9:
            return nominal
        return max(
            0.10,
            min(nominal, BACK_TRANSITION_BRAKE_FRACTION * configured),
        )

    def _planned_back_transition_target(self, state: dict) -> tuple[float, float]:
        sp = self.commander.setpoint
        heading = float(sp.heading_rad)
        forward, _ = _route_axes(heading)
        anchor = np.array([self.takeoff_x, self.takeoff_y], dtype=float)
        position = np.asarray(state["x"], dtype=float)[:2]
        velocity = np.asarray(state["v"], dtype=float)[:2]
        along = float(np.dot(position - anchor, forward))
        route_projection = anchor + along * forward
        forward_speed = max(0.0, float(np.dot(velocity, forward)))
        decel = self._effective_back_decel()

        jerk_ramp_s = 0.0
        if self.controller is not None:
            jerk_target = getattr(
                self.controller, "max_command_jerk_mps3", None
            )
            if jerk_target is not None and jerk_target > 0.0:
                internal_jerk = max(
                    0.02,
                    float(jerk_target)
                    * float(getattr(
                        self.controller, "command_jerk_headroom", 1.0
                    )),
                )
                jerk_ramp_s = decel / internal_jerk

        physical_stop_time = (
            forward_speed / max(decel, 1e-6) + jerk_ramp_s
        )
        self.back_transition_duration_s = max(
            24.0, 1.03 * physical_stop_time
        )

        braking_distance = (
            forward_speed * forward_speed / (2.0 * max(decel, 1e-6))
            + 0.5 * forward_speed * jerk_ramp_s
        )
        schedule_distance = (
            0.5 * forward_speed * self.back_transition_duration_s
        )
        forward_distance = max(
            8.0, braking_distance, schedule_distance
        ) + 5.0
        target = route_projection + forward_distance * forward
        return float(target[0]), float(target[1])


def planned_hover_target_v045(
    state: dict,
    setpoint,
    duration_s: float = 24.0,
    comfortable_decel_mps2: float = 0.75,
) -> tuple[np.ndarray, float]:
    """Manual stop planning using the same usable braking fraction as control."""
    heading = float(setpoint.heading_rad)
    forward, _ = _route_axes(heading)
    anchor = np.array(
        [float(setpoint.hold_x_m), float(setpoint.hold_y_m)], dtype=float
    )
    position = np.asarray(state["x"], dtype=float)[:2]
    velocity = np.asarray(state["v"], dtype=float)[:2]
    along = float(np.dot(position - anchor, forward))
    route_projection = anchor + along * forward
    forward_speed = max(0.0, float(np.dot(velocity, forward)))
    effective_decel = max(
        0.10,
        BACK_TRANSITION_BRAKE_FRACTION
        * max(0.10, float(comfortable_decel_mps2)),
    )
    physical_duration = forward_speed / effective_decel
    planned_duration = max(float(duration_s), physical_duration)
    braking_distance = (
        forward_speed * forward_speed / (2.0 * effective_decel)
    )
    schedule_distance = 0.5 * forward_speed * planned_duration
    forward_distance = max(
        base_dashboard.BACK_TRANSITION_MIN_FORWARD_M,
        braking_distance,
        schedule_distance,
    ) + base_dashboard.BACK_TRANSITION_BUFFER_M
    return route_projection + forward_distance * forward, forward_distance


def sample_to_record_v045(sample, params: dict) -> dict:
    """Keep route tracking continuous during back transition.

    The fixed stop point remains available through ``route_anchor_*`` and a new
    signed ``distance_to_stop_m`` field.  x/y tracking errors remain route-normal
    errors, so changing from CRUISE to TRANSITION_TO_HOVER no longer creates a
    fictitious +200 m x error.
    """
    rec = _ORIGINAL_SAMPLE_TO_RECORD(sample, params)
    control = sample.control
    phase_value = int(round(float(control.get("phase_code", 0.0))))
    try:
        phase = FlightPhase(phase_value)
    except ValueError:
        phase = FlightPhase.GROUND

    rec["comfort_command_accel_mps2"] = float(
        control.get("comfort_command_accel_mps2", np.nan)
    )
    rec["comfort_command_jerk_mps3"] = float(
        control.get("comfort_command_jerk_mps3", np.nan)
    )
    rec["altitude_safety_factor"] = float(
        control.get("altitude_safety_factor", 0.0)
    )
    rec["distance_to_stop_m"] = np.nan

    if phase != FlightPhase.TRANSITION_TO_HOVER:
        return rec

    actual_xy = np.asarray(sample.state["x"], dtype=float)[:2]
    stop_xy = np.array([
        float(control.get("target_x", rec.get("route_anchor_x_m", 0.0))),
        float(control.get("target_y", rec.get("route_anchor_y_m", 0.0))),
    ])
    heading = float(control.get("route_heading", 0.0))
    forward, lateral = _route_axes(heading)

    from_stop = actual_xy - stop_xy
    projection = stop_xy + float(np.dot(from_stop, forward)) * forward
    error = projection - actual_xy

    rec["target_x_m"] = float(projection[0])
    rec["target_y_m"] = float(projection[1])
    rec["route_anchor_x_m"] = float(stop_xy[0])
    rec["route_anchor_y_m"] = float(stop_xy[1])
    rec["x_error_m"] = float(error[0])
    rec["y_error_m"] = float(error[1])
    rec["cross_track_error_m"] = float(np.dot(error, lateral))
    rec["along_track_error_m"] = 0.0
    rec["position_reference_type"] = 1.0
    rec["distance_to_stop_m"] = float(np.dot(stop_xy - actual_xy, forward))
    return rec


def _new_simulation() -> ComfortAwareSimulation:
    vehicle = Tiltrotor()
    commander = CommandManager()
    controller = ComfortEnvelopeController(vehicle, commander)
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
    """Build the v0.4.5 dashboard on top of the validated v0.4.4 UI."""
    sim = _new_simulation() if simulation is None else simulation

    for field in _EXTRA_SOURCE_FIELDS:
        if field not in base_dashboard.SOURCE_FIELDS:
            base_dashboard.SOURCE_FIELDS.append(field)

    # The base dashboard callbacks resolve these names at run time.  Replacing
    # them before constructing the document upgrades automatic mission planning,
    # manual stopping-point planning and record semantics without duplicating the
    # established Bokeh UI implementation.
    base_dashboard.planned_hover_target = planned_hover_target_v045
    base_dashboard.sample_to_record = sample_to_record_v045

    def mission_factory(commander, **kwargs):
        return GuardAwareAutomaticMission(
            commander, controller=sim.controller, **kwargs
        )

    base_dashboard.AutomaticMission = mission_factory

    v044.build_dashboard(doc, simulation=sim)

    position_plot = _find_plot(
        doc, "Position tracking (point hold / route projection)"
    )
    position_plot.title.text = (
        "Position / route tracking and planned stopping distance"
    )
    source = _plot_source(position_plot)
    position_plot.line(
        "time_s",
        "distance_to_stop_m",
        source=source,
        legend_label="Distance to stop",
        line_width=2.0,
        line_dash="dotted",
        muted_alpha=0.12,
    )

    accel_plot = _find_plot(doc, "Passenger comfort — acceleration")
    accel_source = _plot_source(accel_plot)
    accel_plot.line(
        "time_s",
        "comfort_command_accel_mps2",
        source=accel_source,
        legend_label="Command |a|",
        line_width=2.0,
        line_dash="dotdash",
        muted_alpha=0.12,
    )

    jerk_plot = _find_plot(doc, "Passenger comfort — jerk")
    jerk_source = _plot_source(jerk_plot)
    jerk_plot.line(
        "time_s",
        "comfort_command_jerk_mps3",
        source=jerk_source,
        legend_label="Command jerk",
        line_width=2.0,
        line_dash="dotdash",
        muted_alpha=0.12,
    )

    interpretation = Div(
        text=(
            "<b>v0.4.5 comfort envelope:</b> when altitude safety override is "
            "inactive, the selected acceleration target is applied to the full "
            "3-D command norm and the jerk target applies to the full command "
            "slew norm. The plots now show both command and measured motion. "
            "Measured acceleration/jerk can still exceed the target when an "
            "external gust changes faster than the plant can reject it, or when "
            "ALTITUDE SAFETY OVERRIDE intentionally releases vertical authority. "
            "During transition to hover, altitude is first priority, then "
            "longitudinal braking, then lateral tracking. The large planned stop "
            "distance is shown separately and is no longer called x error."
        ),
        sizing_mode="stretch_width",
        styles={
            "white-space": "normal",
            "overflow-wrap": "anywhere",
            "word-break": "break-word",
            "line-height": "1.35",
            "padding": "4px 6px",
        },
    )
    root = doc.roots[0]
    if hasattr(root, "children"):
        root.children.insert(min(10, len(root.children)), interpretation)

    return sim
