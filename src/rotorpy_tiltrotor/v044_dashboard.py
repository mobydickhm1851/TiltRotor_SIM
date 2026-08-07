"""v0.4.4 safety-priority dashboard and transition controller.

Priority order used by this module:

1. altitude / flight safety;
2. attitude stability;
3. route / speed tracking;
4. passenger-comfort shaping.

Comfort targets therefore remain soft engineering constraints.  When altitude
loss or downward velocity becomes safety-relevant, vertical acceleration and
jerk authority are progressively released and nacelle tilt is biased back
toward a configuration that can support weight. Horizontal tracking is also
progressively suppressed so the aircraft does not trade vertical lift for
position/speed recovery while altitude is unsafe.
"""
from __future__ import annotations

import numpy as np
from bokeh.models import Div, NumericInput, Select, Toggle
from scipy.spatial.transform import Rotation

from . import dashboard as base_dashboard
from . import v042_dashboard as v042
from . import v043_dashboard as v043
from .commands import CommandManager, FlightPhase
from .controller import TiltrotorController
from .enhanced_dashboard import ComfortAwareSimulation
from .math_utils import clamp_norm, smoothstep01
from .vehicle import Tiltrotor


SAFETY_MAX_ACCEL_MPS2 = 4.0
EMERGENCY_VERTICAL_JERK_MPS3 = 6.0
ALTITUDE_DEFICIT_START_M = 0.20
ALTITUDE_DEFICIT_FULL_M = 1.50
DESCENT_RATE_START_MPS = 0.25
DESCENT_RATE_FULL_MPS = 1.20
CRUISE_SAFETY_TILT_REDUCTION = np.deg2rad(28.0)
SAFETY_HORIZONTAL_RETAIN_FRACTION = 0.03
WIND_FEEDFORWARD_TAU_S = 0.85
MAX_CRAB_FEEDFORWARD_RAD = np.deg2rad(30.0)
MAX_FLOW_PITCH_FEEDFORWARD_RAD = np.deg2rad(14.0)


def _wrap_pi(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


class SafetyPriorityController(v043.BackTransitionSafeController):
    """Hierarchical controller: safety can override comfort and tracking.

    The base dashboard stores the user comfort-acceleration target in
    ``controller.max_accel`` when the guard is enabled. v0.4.4 interprets that
    value as a *comfort* limit, not as the aircraft's absolute control authority.
    The original 4 m/s^2 authority remains available to recover altitude.

    A second arbitration layer applies even when Comfort Guard is OFF: once an
    altitude safety factor grows, horizontal path/speed acceleration is faded
    almost to zero. This prevents a fixed-point or speed controller from tilting
    the aircraft aggressively while rotor thrust is needed primarily to arrest
    a descent.
    """

    vertical_floor_phases = set(
        v043.BackTransitionSafeController.vertical_floor_phases
    ) | {
        FlightPhase.TRANSITION_TO_CRUISE,
        FlightPhase.CRUISE,
    }

    def __init__(self, *args, **kwargs):
        self.safety_max_accel_mps2 = SAFETY_MAX_ACCEL_MPS2
        self.emergency_vertical_jerk_mps3 = EMERGENCY_VERTICAL_JERK_MPS3
        super().__init__(*args, **kwargs)

    def reset(self) -> None:
        super().reset()
        self._last_priority_accel_world = np.zeros(3)
        self.safety_override_factor = 0.0
        self.safety_override_active = False
        self._filtered_crab_rad = 0.0
        self._filtered_flow_pitch_rad = 0.0

    def _altitude_safety_factor(self, state: dict) -> float:
        sp = self.commander.setpoint
        phase = sp.phase
        if phase in (FlightPhase.GROUND, FlightPhase.COMPLETE):
            return 0.0

        # Being below the final target is expected during a vertical takeoff;
        # do not call the entire climb an emergency. A genuine downward motion
        # during takeoff can still trigger the descent-rate part of the guard.
        if phase in (FlightPhase.VERTICAL_TAKEOFF, FlightPhase.VERTICAL_LANDING):
            deficit_factor = 0.0
        else:
            deficit = max(0.0, float(sp.altitude_m - state["x"][2]))
            deficit_factor = float(smoothstep01(
                (deficit - ALTITUDE_DEFICIT_START_M)
                / max(
                    ALTITUDE_DEFICIT_FULL_M - ALTITUDE_DEFICIT_START_M,
                    1e-6,
                )
            ))

        downward_speed = max(0.0, -float(state["v"][2]))
        descent_factor = float(smoothstep01(
            (downward_speed - DESCENT_RATE_START_MPS)
            / max(DESCENT_RATE_FULL_MPS - DESCENT_RATE_START_MPS, 1e-6)
        ))
        return max(deficit_factor, descent_factor)

    def _desired_velocity_and_acceleration(self, t: float, state: dict):
        # ``max_accel`` may have been reduced by the dashboard to the comfort
        # target. Temporarily restore full flight-control authority while the
        # baseline command is computed, then arbitrate priorities below.
        configured_limit = float(self.max_accel)
        guard_active = (
            self.max_command_jerk_mps3 is not None
            or configured_limit < self.safety_max_accel_mps2 - 1e-9
        )
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
        self.safety_override_active = safety > 0.05

        # This tracking fade applies regardless of Comfort Guard. The first
        # v0.4.4 regression showed safety=1 while the aircraft was still being
        # driven to very high horizontal speed during back transition. Once
        # altitude is unsafe, path/speed recovery must yield to lift recovery.
        tracking_scale = (
            1.0
            - safety * (1.0 - SAFETY_HORIZONTAL_RETAIN_FRACTION)
        )
        safety_horizontal = raw_accel[:2] * tracking_scale

        if not guard_active:
            target = np.array([
                safety_horizontal[0],
                safety_horizontal[1],
                float(raw_accel[2]),
            ])
            self._last_priority_accel_world = target.copy()
            return v_ref, target

        comfort_limit = max(0.05, min(
            configured_limit,
            self.safety_max_accel_mps2,
        ))

        # Comfort owns horizontal manoeuvring authority while safety can reduce
        # it further. Vertical authority is comfort-limited only while altitude
        # is healthy; safety progressively restores the unrestricted command.
        horizontal = clamp_norm(safety_horizontal, comfort_limit)
        nominal_vertical = float(np.clip(
            raw_accel[2], -comfort_limit, comfort_limit
        ))
        vertical = (
            (1.0 - safety) * nominal_vertical
            + safety * float(raw_accel[2])
        )
        target = np.array([horizontal[0], horizontal[1], vertical])

        jerk_target = self.max_command_jerk_mps3
        if (
            jerk_target is not None
            and jerk_target > 0.0
            and self._control_dt > 0.0
        ):
            previous = self._last_priority_accel_world
            internal_jerk = max(
                0.02,
                float(jerk_target) * float(self.command_jerk_headroom),
            )

            delta_xy = target[:2] - previous[:2]
            delta_xy = clamp_norm(
                delta_xy,
                internal_jerk * self._control_dt,
            )

            # Passenger jerk remains the nominal target, but vertical emergency
            # recovery may exceed it. The measured exceedance remains visible.
            vertical_jerk_limit = (
                (1.0 - safety) * internal_jerk
                + safety * self.emergency_vertical_jerk_mps3
            )
            delta_z = float(np.clip(
                target[2] - previous[2],
                -vertical_jerk_limit * self._control_dt,
                vertical_jerk_limit * self._control_dt,
            ))
            target = np.array([
                previous[0] + delta_xy[0],
                previous[1] + delta_xy[1],
                previous[2] + delta_z,
            ])

        # Do not reapply a 3-D comfort norm here: that would steal vertical
        # authority again during a safety override.
        target[:2] = clamp_norm(target[:2], comfort_limit)
        target[2] = float(np.clip(
            target[2],
            -self.safety_max_accel_mps2,
            self.safety_max_accel_mps2,
        ))
        self._last_priority_accel_world = target.copy()
        return v_ref, target

    def _desired_attitude(
        self,
        state: dict,
        a_cmd_world: np.ndarray,
    ) -> Rotation:
        """Add low-pass wind-relative pitch/crab feed-forward in wing flight.

        Holding the ground-route yaw at exactly zero under a 10 m/s crosswind
        forces the wing to carry a very large sideslip. Likewise, a -2.25 m/s
        vertical wind changes the relative-flow angle enough to remove much of
        the nominal CL0 lift if the body remains level. A wing-borne aircraft
        normally crabs into crosswind and pitches relative to the air mass.

        The feed-forward is deliberately bounded and low-pass filtered because
        repeated gusts should not be followed sample-for-sample by yaw/pitch.
        Rotor-borne flight receives almost none of this correction.
        """
        sp = self.commander.setpoint
        route_heading = float(sp.heading_rad)
        forward = np.array([
            np.cos(route_heading), np.sin(route_heading), 0.0
        ])
        lateral = np.array([
            -np.sin(route_heading), np.cos(route_heading), 0.0
        ])
        lateral_accel = float(np.dot(a_cmd_world, lateral))
        forward_accel = float(np.dot(a_cmd_world, forward))
        vertical_rotor_fraction = float(
            np.cos(float(state["tilt_angle"])) ** 2
        )
        wing_fraction = 1.0 - vertical_rotor_fraction

        roll_cmd = float(np.clip(
            -lateral_accel / self.params["gravity"],
            -0.20,
            0.20,
        ))
        pitch_cmd = (
            vertical_rotor_fraction
            * forward_accel
            / self.params["gravity"]
            - 0.75
            * wing_fraction
            * a_cmd_world[2]
            / self.params["gravity"]
        )

        wind = np.asarray(state.get("wind", np.zeros(3)), dtype=float)
        air_relative_world = np.asarray(state["v"], dtype=float) - wind
        horizontal_air = float(np.linalg.norm(air_relative_world[:2]))
        if horizontal_air > 2.0:
            airflow_heading = float(np.arctan2(
                air_relative_world[1], air_relative_world[0]
            ))
            raw_crab = np.clip(
                _wrap_pi(airflow_heading - route_heading),
                -MAX_CRAB_FEEDFORWARD_RAD,
                MAX_CRAB_FEEDFORWARD_RAD,
            )
            raw_flow_pitch = float(np.clip(
                -np.arctan2(air_relative_world[2], horizontal_air),
                -MAX_FLOW_PITCH_FEEDFORWARD_RAD,
                MAX_FLOW_PITCH_FEEDFORWARD_RAD,
            ))
        else:
            raw_crab = 0.0
            raw_flow_pitch = 0.0

        dt = max(float(self._control_dt), 1e-3)
        alpha = float(np.clip(
            dt / (WIND_FEEDFORWARD_TAU_S + dt), 0.0, 1.0
        ))
        self._filtered_crab_rad += alpha * (
            float(raw_crab) - self._filtered_crab_rad
        )
        self._filtered_flow_pitch_rad += alpha * (
            raw_flow_pitch - self._filtered_flow_pitch_rad
        )

        # If altitude is already unsafe, level-flight safety is more important
        # than aggressively following a changing crosswind crab demand.
        safety = self._altitude_safety_factor(state)
        wind_ff_blend = wing_fraction * (1.0 - 0.55 * safety)
        yaw_cmd = route_heading + wind_ff_blend * self._filtered_crab_rad
        pitch_cmd += wind_ff_blend * self._filtered_flow_pitch_rad
        pitch_cmd = float(np.clip(pitch_cmd, -0.26, 0.26))

        return Rotation.from_euler(
            "xyz", [roll_cmd, pitch_cmd, yaw_cmd]
        )

    def _tilt_command(self, t: float, state: dict) -> float:
        sp = self.commander.setpoint
        phase = sp.phase
        tilt_max = float(self.params["tilt_max"])
        safety = self._altitude_safety_factor(state)

        if phase == FlightPhase.TRANSITION_TO_CRUISE:
            heading = float(sp.heading_rad)
            forward = np.array([
                np.cos(heading), np.sin(heading), 0.0
            ])
            wind = np.asarray(state.get("wind", np.zeros(3)), dtype=float)
            air_relative_world = np.asarray(state["v"], dtype=float) - wind
            forward_airspeed = max(
                0.0, float(np.dot(air_relative_world, forward))
            )
            speed_progress = float(np.clip(
                forward_airspeed / max(float(sp.airspeed_mps), 1.0),
                0.0,
                1.0,
            ))
            elapsed = self.commander.phase_progress(t)

            # Never hand lift to the wing faster than forward airspeed is being
            # established. This is crucial when a 0.5 m/s^2 comfort target makes
            # acceleration much slower than the nominal 12-s schedule.
            airspeed_cap = 0.05 + 0.95 * speed_progress
            progress = min(elapsed, airspeed_cap)

            # If altitude starts to decay, pause/partly reverse the conversion
            # so the rotor axis regains vertical effectiveness.
            progress *= (1.0 - 0.72 * safety)
            return tilt_max * float(smoothstep01(progress))

        if phase == FlightPhase.CRUISE and safety > 0.0:
            # Temporary lift-assist conversion: sacrifice propulsive efficiency
            # and speed before sacrificing altitude.
            return float(np.clip(
                tilt_max - safety * CRUISE_SAFETY_TILT_REDUCTION,
                self.params["tilt_min"],
                tilt_max,
            ))

        if phase == FlightPhase.TRANSITION_TO_HOVER:
            elapsed_progress = self.commander.phase_progress(t)
            speed = float(np.linalg.norm(state["v"][:2]))
            speed_progress = 1.0 - np.clip(speed / 12.0, 0.0, 1.0)
            progress = max(elapsed_progress, 0.65 * speed_progress)
            # Altitude loss should accelerate the return to vertical rotor lift.
            progress = max(progress, 0.88 * safety)
            return tilt_max * (
                1.0 - float(smoothstep01(progress)) ** 0.62
            )

        return super()._tilt_command(t, state)


def _new_simulation() -> ComfortAwareSimulation:
    vehicle = Tiltrotor()
    commander = CommandManager()
    controller = SafetyPriorityController(vehicle, commander)
    return ComfortAwareSimulation(
        vehicle=vehicle,
        commander=commander,
        controller=controller,
        dt=0.01,
    )


def _find_one(doc, model_type, *, title_prefix=None, label_prefix=None):
    for model in doc.select({"type": model_type}):
        text = str(getattr(model, "title", ""))
        label = str(getattr(model, "label", ""))
        if title_prefix is not None and text.startswith(title_prefix):
            return model
        if label_prefix is not None and label.startswith(label_prefix):
            return model
    raise LookupError(title_prefix or label_prefix or str(model_type))


def build_dashboard(doc, simulation=None):
    """Build v0.4.4 with safety-priority control and wrap-safe UI text."""
    sim = _new_simulation() if simulation is None else simulation
    v043.build_dashboard(doc, simulation=sim)

    wind_mode = _find_one(doc, Select, title_prefix="Wind scenario")
    amplitude = next(
        model for model in doc.select({"type": NumericInput})
        if str(model.title).startswith((
            "Disturbance",
            "Mean wind",
            "Reference mean wind",
        ))
    )
    delay = next(
        model for model in doc.select({"type": NumericInput})
        if str(model.title).startswith("Delay")
    )
    duration = next(
        model for model in doc.select({"type": NumericInput})
        if "duration" in str(model.title).lower()
    )
    frequency = next(
        model for model in doc.select({"type": NumericInput})
        if "frequency" in str(model.title).lower()
    )
    guard = _find_one(doc, Toggle, label_prefix="Comfort accel")

    # Keep widget captions short enough to fit their cells. Detailed wording is
    # placed in wrapping Divs below, where CSS can reliably wrap at any browser
    # width. Bokeh input titles themselves do not provide robust auto-wrap.
    delay.title = "Delay [s]"
    duration.title = "Duration [s]"
    frequency.title = "Frequency [Hz]"
    amplitude.title = "Disturbance [m/s]"
    wind_mode.width = max(int(wind_mode.width or 155), 210)

    def compact_wind_titles(attr, old, new) -> None:
        del attr, old, new
        amplitude.title = (
            "Mean wind U [m/s]"
            if wind_mode.value == v042.CS_AWO_LABEL
            else "Disturbance [m/s]"
        )
        delay.title = "Delay [s]"
        duration.title = "Duration [s]"
        frequency.title = "Frequency [Hz]"

    wind_mode.on_change("value", compact_wind_titles)
    compact_wind_titles(None, None, None)

    wrap_styles = {
        "white-space": "normal",
        "overflow-wrap": "anywhere",
        "word-break": "break-word",
        "line-height": "1.4",
        "max-width": "100%",
        "box-sizing": "border-box",
    }
    for div in doc.select({"type": Div}):
        text = str(div.text)
        if (
            "disturbance controls" in text
            or text.startswith("<b>Mode:")
            or text.startswith("<b>Urban wind:")
        ):
            merged = dict(getattr(div, "styles", {}) or {})
            merged.update(wrap_styles)
            merged["padding"] = "4px 6px"
            div.styles = merged
            div.width = None
            div.sizing_mode = "stretch_width"
            if "disturbance controls" in text:
                div.height = 72

    priority_status = Div(
        text="",
        height=58,
        sizing_mode="stretch_width",
        styles={
            **wrap_styles,
            "padding": "5px 7px",
            "border": "1px solid #d9d9d9",
            "border-radius": "4px",
        },
    )

    def update_priority_status() -> None:
        factor = float(getattr(sim.controller, "safety_override_factor", 0.0))
        active = bool(getattr(sim.controller, "safety_override_active", False))
        state = (
            "<span style='color:#d62728'><b>ALTITUDE SAFETY OVERRIDE</b></span>"
            if active
            else "<span style='color:#2ca02c'><b>Normal priority</b></span>"
        )
        priority_status.text = (
            f"{state} &nbsp; <b>Altitude-priority factor:</b> {factor:.2f}. "
            "Policy: altitude/attitude safety → path tracking → passenger comfort. "
            "When the safety override is active, horizontal tracking yields first; "
            "vertical acceleration or jerk may exceed the comfort target and the "
            "measured exceedance remains visible."
        )

    root = doc.roots[0]
    if hasattr(root, "children"):
        # Insert immediately before the status/plot section without putting the
        # explanation inside a fixed-width input cell.
        insert_at = max(0, len(root.children) - 3)
        root.children.insert(insert_at, priority_status)

    doc.add_periodic_callback(update_priority_status, 100)
    update_priority_status()
    guard.label = (
        "Comfort guard: ON" if guard.active else "Comfort guard: OFF"
    )
    return sim
