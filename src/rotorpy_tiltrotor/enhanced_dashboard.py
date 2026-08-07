"""Enhanced v0.4.1 dashboard for urban disturbance and ride-comfort studies.

This module deliberately wraps the established dashboard instead of duplicating
its flight-command and plotting implementation.  It adds:

* repeated gusts and continuous turbulence to the wind selector;
* scenario-relative timing (delay after enable);
* kinematic acceleration/jerk measurement suitable for ground/takeoff plots;
* a command-level acceleration + jerk guard;
* ISO-2631-style Wd/Wk weighted-vibration plots and source-backed defaults.
"""
from __future__ import annotations

import numpy as np
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, Div, NumericInput, Select, Toggle
from bokeh.plotting import figure

from . import dashboard as base_dashboard
from .commands import CommandManager
from .controller import TiltrotorController
from .math_utils import clamp_norm
from .scenarios import UrbanWindModel
from .simulator import TiltrotorSimulation
from .vehicle import Tiltrotor


EXTRA_SOURCE_FIELDS = [
    "scenario_elapsed_s",
    "gust_event_index",
    "comfort_iso_awx_mps2",
    "comfort_iso_awy_mps2",
    "comfort_iso_awz_mps2",
    "comfort_iso_weighted_rms_mps2",
    "comfort_iso_limit_mps2",
    "comfort_iso_index",
]


class FixedUrbanWindModel(UrbanWindModel):
    """Keep continuous-turbulence states in the local wind-axis frame."""

    def _continuous_turbulence(self, t, elapsed, cfg):
        if elapsed < cfg.start_time_s:
            self._last_turbulence_time = float(t)
            self._turbulence_state[:] = 0.0
            return np.zeros(3), False

        dt = (
            0.0
            if self._last_turbulence_time is None
            else max(0.0, float(t) - self._last_turbulence_time)
        )
        self._last_turbulence_time = float(t)
        along, cross = self._direction_vectors(cfg.disturbance_direction_deg)

        if dt > 0.0:
            tau = max(0.18, 1.0 / (2.0 * np.pi * cfg.wake_frequency_hz))
            axis_tau = np.array([1.0, 0.75, 0.55]) * tau
            alpha = np.exp(-dt / axis_tau)
            sigma = cfg.disturbance_amplitude_mps * np.array([0.65, 0.85, 0.35])
            innovation = self._rng.normal(size=3)
            self._turbulence_state = (
                alpha * self._turbulence_state
                + sigma
                * np.sqrt(np.maximum(0.0, 1.0 - alpha * alpha))
                * innovation
            )

        local = self._turbulence_state
        disturbance = (
            local[0] * along
            + local[1] * cross
            + local[2] * np.array([0.0, 0.0, 1.0])
        )
        return disturbance, True


class ComfortAwareController(TiltrotorController):
    """Base controller with global acceleration and jerk command limiting."""

    def __init__(self, *args, **kwargs):
        self.max_command_jerk_mps3: float | None = None
        super().__init__(*args, **kwargs)

    def reset(self) -> None:
        super().reset()
        self._last_limited_accel_world = np.zeros(3)

    def _desired_velocity_and_acceleration(self, t, state):
        v_ref, a_cmd = super()._desired_velocity_and_acceleration(t, state)
        # v0.4.0 only applied max_accel inside forward-flight code.  Apply it
        # here as a global guard so vertical takeoff/landing are covered too.
        a_cmd = clamp_norm(np.asarray(a_cmd, dtype=float), float(self.max_accel))

        jerk_limit = self.max_command_jerk_mps3
        if jerk_limit is not None and jerk_limit > 0.0:
            if self._control_dt <= 0.0:
                a_cmd = self._last_limited_accel_world.copy()
            else:
                delta = a_cmd - self._last_limited_accel_world
                delta = clamp_norm(
                    delta,
                    float(jerk_limit) * self._control_dt,
                )
                a_cmd = self._last_limited_accel_world + delta
                a_cmd = clamp_norm(a_cmd, float(self.max_accel))

        self._last_limited_accel_world = np.asarray(a_cmd, dtype=float).copy()
        return v_ref, a_cmd


class ComfortAwareSimulation(TiltrotorSimulation):
    """Use kinematic acceleration/jerk after constraints are applied.

    The original diagnostics differentiated the unconstrained force-model
    acceleration.  On the ground that value can contain -g even though the
    ground constraint keeps velocity at zero.  Differentiating it at lift-off
    creates an artificial jerk spike.  Here acceleration is calculated from
    the actual integrated velocity change, so ground contact is respected.
    """

    def reset(self, state=None) -> None:
        super().reset(state)
        self._kinematic_previous_accel: np.ndarray | None = None

    def step(self, wind_world=None):
        velocity_before = np.asarray(self.state["v"], dtype=float).copy()
        sample = super().step(wind_world)
        velocity_after = np.asarray(self.state["v"], dtype=float)
        accel = (velocity_after - velocity_before) / self.dt
        if self._kinematic_previous_accel is None:
            jerk = np.zeros(3)
        else:
            jerk = (
                accel - self._kinematic_previous_accel
            ) / self.dt
        self._kinematic_previous_accel = accel.copy()

        self.vehicle.last_diagnostics["accel_world"] = accel
        self.vehicle.last_diagnostics["jerk_world"] = jerk
        self.previous_accel = accel.copy()
        self.last_sample = self._make_sample()
        return self.last_sample


def _find_one(doc, model_type, title=None, label_prefix=None):
    for model in doc.select({"type": model_type}):
        if title is not None and getattr(model, "title", None) == title:
            return model
        if label_prefix is not None and str(getattr(model, "label", "")).startswith(label_prefix):
            return model
    raise LookupError(f"Dashboard control not found: {title or label_prefix}")


def _new_simulation() -> ComfortAwareSimulation:
    vehicle = Tiltrotor()
    commander = CommandManager()
    controller = ComfortAwareController(vehicle, commander)
    return ComfortAwareSimulation(
        vehicle=vehicle,
        commander=commander,
        controller=controller,
        dt=0.01,
    )


def build_dashboard(doc, simulation=None):
    """Build v0.4.1 dashboard while preserving the established flight UI."""
    # The old builder hard-codes the legacy value "Discrete gust" during
    # construction, so keep that key temporarily and then replace the options.
    base_dashboard.WIND_MODE_LABELS.clear()
    base_dashboard.WIND_MODE_LABELS.update({
        "Steady wind": "steady",
        "Discrete gust": "discrete_gust",
        "Single discrete gust": "discrete_gust",
        "Repeated gusts": "repeated_gusts",
        "Continuous turbulence": "continuous_turbulence",
        "Wind shear": "wind_shear",
        "Urban wake proxy": "urban_wake",
    })
    base_dashboard.UrbanWindModel = FixedUrbanWindModel
    for field in EXTRA_SOURCE_FIELDS:
        if field not in base_dashboard.SOURCE_FIELDS:
            base_dashboard.SOURCE_FIELDS.append(field)

    sim = _new_simulation() if simulation is None else simulation
    base_dashboard.build_dashboard(doc, simulation=sim)

    wind_mode = _find_one(doc, Select, title="Wind scenario")
    wind_mode.options = [
        "Steady wind",
        "Single discrete gust",
        "Repeated gusts",
        "Continuous turbulence",
        "Wind shear",
        "Urban wake proxy",
    ]
    wind_mode.value = "Repeated gusts"

    delay_input = _find_one(doc, NumericInput, title="Start time [s]")
    delay_input.title = "Delay after scenario enable [s]"
    delay_input.value = 2.0
    duration_input = _find_one(doc, NumericInput, title="Duration [s]")
    duration_input.title = "Single-gust / wake duration [s]"
    duration_input.value = 3.0
    frequency_input = _find_one(doc, NumericInput, title="Wake frequency [Hz]")
    frequency_input.title = "Repeat / turbulence frequency [Hz]"
    frequency_input.value = 0.25

    accel_input = _find_one(doc, NumericInput, title="Acceleration limit [m/s²]")
    accel_input.title = "Transient accel target [m/s²] (NASA-informed)"
    accel_input.value = 0.50
    jerk_input = _find_one(doc, NumericInput, title="Jerk limit [m/s³]")
    jerk_input.title = "Transient jerk target [m/s³] (NASA-informed)"
    jerk_input.value = 1.50
    rate_input = _find_one(doc, NumericInput, title="Angular-rate limit [deg/s]")
    rate_input.title = "Angular-rate study threshold [deg/s]"

    guard = _find_one(doc, Toggle, label_prefix="Comfort accel guard")

    def apply_jerk_guard(*_):
        if hasattr(sim.controller, "max_command_jerk_mps3"):
            sim.controller.max_command_jerk_mps3 = (
                float(jerk_input.value or 1.50)
                if guard.active
                else None
            )

    guard.on_change("active", apply_jerk_guard)
    jerk_input.on_change("value", apply_jerk_guard)
    apply_jerk_guard()

    # Locate the primary stream after the base dashboard added all fields.
    stream_source = None
    for candidate in doc.select({"type": ColumnDataSource}):
        if "comfort_iso_weighted_rms_mps2" in candidate.data:
            stream_source = candidate
            break
    if stream_source is None:
        raise RuntimeError("Unable to locate enhanced dashboard data source")

    iso_plot = figure(
        title="ISO 2631-style weighted whole-body vibration",
        x_axis_label="Time [s]",
        y_axis_label="weighted acceleration [m/s²]",
        height=260,
        sizing_mode="stretch_width",
    )
    iso_plot.line(
        "time_s", "comfort_iso_weighted_rms_mps2",
        source=stream_source, legend_label="Weighted RMS (Wd/Wk)",
        line_width=2.4,
    )
    iso_plot.line(
        "time_s", "comfort_iso_limit_mps2",
        source=stream_source, legend_label="0.315 m/s² target",
        line_width=2.0, line_dash="dashed",
    )
    iso_plot.legend.click_policy = "mute"

    explanation = Div(
        text=(
            "<h3>Comfort interpretation (v0.4.1)</h3>"
            "<b>Monitor ON</b> records motion only; it does not change control. "
            "<b>Comfort accel + jerk guard ON</b> clamps the controller's "
            "acceleration command and its rate of change. The default transient "
            "targets (0.50 m/s² acceleration, 1.50 m/s³ jerk) are conservative "
            "engineering targets informed by NASA UAM sudden-heave studies, not "
            "FAA/EASA certification limits. The separate 0.315 m/s² line is an "
            "ISO 2631-1 whole-body-vibration comfort-band boundary after Wd/Wk "
            "frequency weighting. This 100-Hz simulator implementation is an "
            "ISO-style low-frequency approximation, not ISO 8041 compliance."
        ),
        sizing_mode="stretch_width",
    )

    root = doc.roots[0]
    if isinstance(root, column().__class__):
        root.children.extend([explanation, iso_plot])
    else:
        # Bokeh roots are normally the Column built by the base dashboard.
        doc.add_root(column(explanation, iso_plot, sizing_mode="stretch_width"))

    def keep_labels_current():
        guard.label = (
            "Comfort accel + jerk guard: ON"
            if guard.active
            else "Comfort accel + jerk guard: OFF"
        )

    doc.add_periodic_callback(keep_labels_current, 100)
    keep_labels_current()
    return sim
