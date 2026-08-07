"""v0.4.3 dashboard: back-transition altitude protection and stable UI labels."""
from __future__ import annotations

import numpy as np
from bokeh.models import Div, NumericInput, Select, Toggle

from . import dashboard as base_dashboard
from . import v042_dashboard as v042
from .allocator import rotor_axis, thrusts_to_speeds
from .commands import CommandManager, FlightPhase
from .enhanced_dashboard import ComfortAwareController, ComfortAwareSimulation
from .math_utils import rotation_matrix, smoothstep01
from .vehicle import Tiltrotor


# The old 14-s back-transition schedule was shorter than the observed stopping
# time at the comfort-oriented deceleration levels used in the dashboard.  The
# result was a large overshoot followed by a position-recovery manoeuvre after
# the nacelles were already vertical.  Use a slower baseline so the target stop
# point and tilt/deceleration schedule are physically consistent.
BACK_TRANSITION_DURATION_S = 24.0
BACK_TRANSITION_DECEL_MPS2 = 0.75

# World-vertical collective floor becomes active only once the rotor axis has a
# useful upward component.  This avoids demanding impossible vertical thrust
# while the rotors are still nearly horizontal at the beginning of transition.
VERTICAL_FLOOR_START = 0.25
VERTICAL_FLOOR_FULL = 0.60
BACK_TRANSITION_VERTICAL_MARGIN = 1.02


class BackTransitionSafeController(ComfortAwareController):
    """Comfort-aware controller with a world-z rotor-thrust safety floor.

    The base controller already computes the full required propulsive force,
    but its back-transition vertical guard historically compared body-z force
    with body-z rotor-axis effectiveness.  During a bank/pitch excursion those
    are not the same as inertial vertical support.  This post-allocation floor
    adds equal collective thrust (preserving differential moments as far as
    actuator headroom allows) until the commanded rotor force can meet the
    required *world-z* component.
    """

    vertical_floor_phases = {
        FlightPhase.VERTICAL_TAKEOFF,
        FlightPhase.HOVER,
        FlightPhase.TRANSITION_TO_HOVER,
        FlightPhase.VERTICAL_LANDING,
    }

    @staticmethod
    def _add_uniform_collective(
        thrusts: np.ndarray,
        extra_thrust: float,
        maximum_thrust: float,
    ) -> np.ndarray:
        result = np.asarray(thrusts, dtype=float).copy()
        remaining = max(0.0, float(extra_thrust))
        for _ in range(len(result) + 1):
            if remaining <= 1e-9:
                break
            room = np.maximum(0.0, maximum_thrust - result)
            active = room > 1e-9
            count = int(np.count_nonzero(active))
            if count == 0:
                break
            share = remaining / count
            delta = np.zeros_like(result)
            delta[active] = np.minimum(share, room[active])
            added = float(np.sum(delta))
            result += delta
            remaining -= added
            if added <= 1e-12:
                break
        return result

    def update(self, t: float, state: dict, flat: dict | None = None) -> dict:
        control = super().update(t, state, flat)
        phase = self.commander.setpoint.phase
        if phase not in self.vertical_floor_phases:
            return control

        R = rotation_matrix(state["q"])
        axis_body = rotor_axis(float(state["tilt_angle"]))
        axis_world = R @ axis_body
        vertical_effectiveness = float(axis_world[2])
        if vertical_effectiveness <= VERTICAL_FLOOR_START:
            return control

        required_prop_body = np.asarray(
            control.get("cmd_force_body", np.zeros(3)), dtype=float
        )
        required_prop_world = R @ required_prop_body
        required_world_z = max(0.0, float(required_prop_world[2]))
        if required_world_z <= 1e-9:
            return control

        floor_blend = float(smoothstep01(
            (vertical_effectiveness - VERTICAL_FLOOR_START)
            / max(VERTICAL_FLOOR_FULL - VERTICAL_FLOOR_START, 1e-6)
        ))
        margin = (
            BACK_TRANSITION_VERTICAL_MARGIN
            if phase == FlightPhase.TRANSITION_TO_HOVER
            else 1.0
        )
        minimum_collective = (
            margin * required_world_z / max(vertical_effectiveness, 1e-6)
        )

        p = self.params
        cmd_speeds = np.asarray(control["cmd_motor_speeds"], dtype=float)
        thrusts = p["k_eta"] * cmd_speeds * cmd_speeds
        current_collective = float(np.sum(thrusts))
        target_collective = max(
            current_collective,
            floor_blend * minimum_collective,
        )
        shortfall = target_collective - current_collective
        if shortfall <= 1e-9:
            return control

        f_max = float(p["k_eta"] * p["rotor_speed_max"] ** 2)
        thrusts = self._add_uniform_collective(
            thrusts,
            shortfall,
            f_max,
        )
        control["cmd_motor_speeds"] = thrusts_to_speeds(thrusts, p)
        control["world_vertical_floor_n"] = np.array(target_collective)
        return control


def _new_simulation() -> ComfortAwareSimulation:
    vehicle = Tiltrotor()
    commander = CommandManager()
    controller = BackTransitionSafeController(vehicle, commander)
    return ComfortAwareSimulation(
        vehicle=vehicle,
        commander=commander,
        controller=controller,
        dt=0.01,
    )


def _find_one(doc, model_type, *, title_prefix=None, label_prefix=None):
    for model in doc.select({"type": model_type}):
        if title_prefix is not None and str(
            getattr(model, "title", "")
        ).startswith(title_prefix):
            return model
        if label_prefix is not None and str(
            getattr(model, "label", "")
        ).startswith(label_prefix):
            return model
    raise LookupError(title_prefix or label_prefix or str(model_type))


def _remove_guard_label_flicker_callback(doc) -> None:
    """Remove v0.4.2's second 100-ms guard-label writer.

    The base dashboard already refreshes the guard label from the 50-ms main
    callback.  v0.4.2 added another periodic callback using a different string
    ("accel" vs "accel + jerk"), so the same button alternated text fast enough
    to appear blurred/overprinted in the browser.
    """
    for callback in list(getattr(doc, "session_callbacks", [])):
        if getattr(callback, "period", None) == 100:
            doc.remove_periodic_callback(callback)


def build_dashboard(doc, simulation=None):
    """Build the v0.4.3 dashboard."""
    # These globals are read by the existing manual back-transition callback.
    # Keeping the calculation in one place preserves the established UI while
    # making the stop target and nacelle schedule more conservative.
    base_dashboard.BACK_TRANSITION_DURATION_S = BACK_TRANSITION_DURATION_S
    base_dashboard.BACK_TRANSITION_COMFORT_DECEL_MPS2 = (
        BACK_TRANSITION_DECEL_MPS2
    )

    sim = _new_simulation() if simulation is None else simulation
    v042.build_dashboard(doc, simulation=sim)
    _remove_guard_label_flicker_callback(doc)

    wind_mode = _find_one(doc, Select, title_prefix="Wind scenario")
    amplitude = next(
        model for model in doc.select({"type": NumericInput})
        if str(model.title).startswith((
            "Disturbance amplitude",
            "Turbulence RMS sigma",
            "Reference mean wind",
        ))
    )
    duration = next(
        model for model in doc.select({"type": NumericInput})
        if "duration" in str(model.title).lower()
    )
    frequency = next(
        model for model in doc.select({"type": NumericInput})
        if "frequency" in str(model.title).lower()
    )
    accel = next(
        model for model in doc.select({"type": NumericInput})
        if "accel" in str(model.title).lower()
        and "target" in str(model.title).lower()
    )
    jerk = next(
        model for model in doc.select({"type": NumericInput})
        if "jerk" in str(model.title).lower()
        and "target" in str(model.title).lower()
    )
    rate = next(
        model for model in doc.select({"type": NumericInput})
        if "angular-rate" in str(model.title).lower()
    )
    guard = _find_one(doc, Toggle, label_prefix="Comfort accel")

    # Use the same label as the base dashboard so there is only one writer and
    # no visual flicker.  A static help block below explains that v0.4.3 also
    # shapes jerk commands.
    guard.label = (
        "Comfort accel guard: ON"
        if guard.active
        else "Comfort accel guard: OFF"
    )

    # Long dynamic titles wrapped into multiple lines inside fixed-height Bokeh
    # rows.  Keep control titles short and move mode-specific semantics into a
    # dedicated help area with reserved height.
    accel.title = "Accel target [m/s²]"
    jerk.title = "Jerk target [m/s³]"
    rate.title = "Rate threshold [deg/s]"
    amplitude.width = max(int(amplitude.width or 185), 185)
    duration.width = max(int(duration.width or 130), 145)
    frequency.width = max(int(frequency.width or 160), 145)

    scenario_help = Div(
        text="",
        height=54,
        sizing_mode="stretch_width",
        styles={
            "line-height": "1.35",
            "padding": "4px 6px",
            "white-space": "normal",
        },
    )

    def apply_compact_semantics(attr, old, new) -> None:
        del attr, old, new
        is_continuous = wind_mode.value == v042.CS_AWO_LABEL
        if is_continuous:
            amplitude.title = "Mean wind U [m/s]"
            duration.title = "Duration [s]"
            frequency.title = "Frequency [Hz]"
            scenario_help.text = (
                "<b>Continuous turbulence:</b> CS-AWO vertical low-altitude "
                "benchmark; σ<sub>w</sub>=0.09U and L<sub>w</sub>=0.5z "
                "between 9.2 and 305 m. Duration/frequency controls are "
                "inactive in this mode."
            )
        else:
            amplitude.title = "Disturbance [m/s]"
            duration.title = "Duration [s]"
            frequency.title = "Frequency [Hz]"
            scenario_help.text = (
                f"<b>{wind_mode.value}:</b> disturbance controls use the "
                "selected amplitude, direction and timing. "
                "<b>Comfort guard:</b> v0.4.3 shapes both acceleration and "
                "jerk commands, while measured aircraft motion remains the KPI."
            )

    # v0.4.2 has its own scenario callbacks; register this one last so the final
    # visible titles always use the compact, non-overlapping form.
    wind_mode.on_change("value", apply_compact_semantics)
    apply_compact_semantics(None, None, None)

    root = doc.roots[0]
    if hasattr(root, "children"):
        # Insert after the two wind-control rows and before the comfort section.
        root.children.insert(min(6, len(root.children)), scenario_help)

    # The fixed 1500-px status Divs caused horizontal overflow on narrower
    # browser windows. Let them follow the document width instead.
    for div in doc.select({"type": Div}):
        if str(div.text).startswith("<b>Paused") or str(div.text).startswith(
            "<b>Urban scenario:"
        ):
            div.width = None
            div.sizing_mode = "stretch_width"

    return sim
