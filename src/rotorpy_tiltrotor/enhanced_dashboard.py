"""Enhanced v0.4.2 dashboard for urban disturbance and ride-comfort studies.

This module wraps the established dashboard and adds:

* scenario-relative wind timing;
* repeated gusts plus a FAR/CS 25.341-style continuous-turbulence benchmark;
* kinematic acceleration/jerk measurement for ground/takeoff plots;
* acceleration + jerk command guarding with plant-response headroom;
* ISO-2631-style Wd/Wk weighted-vibration plots.

The continuous-turbulence benchmark is intentionally one-dimensional and uses
an approximate finite Fourier realization of the normalized Von Karman spectrum
with L=2500 ft, matching the core FAR/CS 25.341 assumption.  It is a controller-
test benchmark, not a certification analysis and not a CFD-resolved urban flow.
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

FAR_CS_TURBULENCE_SCALE_M = 762.0  # 2500 ft
FAR_CS_REFERENCE_AIRSPEED_MPS = 15.0
VON_KARMAN_COMPONENTS = 256
COMMAND_JERK_HEADROOM = 0.60
TURBULENCE_FADE_IN_S = 3.0


class FixedUrbanWindModel(UrbanWindModel):
    """Wind model with a FAR/CS 25.341-style continuous-turbulence option.

    Repeated gusts and urban wake retain the reduced-order models inherited from
    ``UrbanWindModel``.  Only ``continuous_turbulence`` is replaced here.

    The regulation describes a one-dimensional Gaussian random atmosphere with
    a normalized Von Karman PSD and turbulence scale length L=2500 ft.  The
    dashboard uses a deterministic finite Fourier realization at a 15 m/s
    reference speed and normalizes it to the user-selected RMS sigma.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._vk_key = None
        self._vk_omega = np.zeros(0)
        self._vk_amplitude = np.zeros(0)
        self._vk_phase = np.zeros(0)

    def reset(self) -> None:
        super().reset()
        self._vk_key = None
        self._vk_omega = np.zeros(0)
        self._vk_amplitude = np.zeros(0)
        self._vk_phase = np.zeros(0)

    @staticmethod
    def _smoothstep01(value: float) -> float:
        u = float(np.clip(value, 0.0, 1.0))
        return u * u * (3.0 - 2.0 * u)

    def _ensure_von_karman_basis(self, cfg) -> None:
        key = (int(cfg.random_seed), FAR_CS_REFERENCE_AIRSPEED_MPS)
        if key == self._vk_key:
            return

        # Temporal frequencies are mapped to the regulation's spatial reduced
        # frequency Omega = omega / V.  The band is deliberately concentrated
        # on rigid-body flight-dynamics frequencies; this simulator is not a
        # structural-vibration certification model.
        frequency_hz = np.logspace(
            np.log10(0.002), np.log10(5.0), VON_KARMAN_COMPONENTS
        )
        omega = 2.0 * np.pi * frequency_hz
        reduced = omega / FAR_CS_REFERENCE_AIRSPEED_MPS
        x = 1.339 * FAR_CS_TURBULENCE_SCALE_M * reduced
        phi_spatial = (
            FAR_CS_TURBULENCE_SCALE_M
            / np.pi
            * (1.0 + (8.0 / 3.0) * x * x)
            / np.power(1.0 + x * x, 11.0 / 6.0)
        )
        # Convert the spatial spectrum to a temporal spectrum by dOmega=domega/V.
        phi_temporal = phi_spatial / FAR_CS_REFERENCE_AIRSPEED_MPS
        domega = np.gradient(omega)
        amplitude = np.sqrt(np.maximum(0.0, 2.0 * phi_temporal * domega))

        # Finite frequency limits truncate the theoretical spectrum. Normalize
        # the synthesized process so the user input has the unambiguous meaning
        # of RMS turbulence velocity sigma.
        expected_rms = float(np.sqrt(0.5 * np.sum(amplitude * amplitude)))
        if expected_rms > 1e-12:
            amplitude /= expected_rms

        rng = np.random.default_rng(int(cfg.random_seed))
        self._vk_omega = omega
        self._vk_amplitude = amplitude
        self._vk_phase = rng.uniform(0.0, 2.0 * np.pi, len(omega))
        self._vk_key = key

    def _continuous_turbulence(self, t, elapsed, cfg):
        del t
        if elapsed < cfg.start_time_s:
            return np.zeros(3), False

        self._ensure_von_karman_basis(cfg)
        local_time = float(elapsed - cfg.start_time_s)
        unit_value = float(np.sum(
            self._vk_amplitude
            * np.cos(self._vk_omega * local_time + self._vk_phase)
        ))
        fade = self._smoothstep01(local_time / TURBULENCE_FADE_IN_S)
        vertical_velocity = (
            float(cfg.disturbance_amplitude_mps) * fade * unit_value
        )

        # FAR/CS 25.341 evaluates vertical and lateral continuous turbulence;
        # the default benchmark here uses the vertical case, which is the most
        # directly relevant to altitude/lift robustness during transition.
        return np.array([0.0, 0.0, vertical_velocity]), True


class ComfortAwareController(TiltrotorController):
    """Base controller with global acceleration and jerk command limiting.

    ``max_command_jerk_mps3`` remains the passenger-facing measured jerk target.
    Internally only 60% is used for command slew, leaving headroom for motor lag,
    tilt dynamics and closed-loop plant response.  The measured aircraft jerk is
    still monitored independently and can exceed the command rate under strong
    external disturbances.
    """

    def __init__(self, *args, **kwargs):
        self.max_command_jerk_mps3: float | None = None
        self.command_jerk_headroom = COMMAND_JERK_HEADROOM
        super().__init__(*args, **kwargs)

    def reset(self) -> None:
        super().reset()
        self._last_limited_accel_world = np.zeros(3)

    def _desired_velocity_and_acceleration(self, t, state):
        v_ref, a_cmd = super()._desired_velocity_and_acceleration(t, state)
        a_cmd = clamp_norm(np.asarray(a_cmd, dtype=float), float(self.max_accel))

        jerk_target = self.max_command_jerk_mps3
        if jerk_target is not None and jerk_target > 0.0:
            internal_limit = (
                float(jerk_target) * float(self.command_jerk_headroom)
            )
            if self._control_dt <= 0.0:
                a_cmd = self._last_limited_accel_world.copy()
            else:
                delta = a_cmd - self._last_limited_accel_world
                delta = clamp_norm(
                    delta,
                    internal_limit * self._control_dt,
                )
                a_cmd = self._last_limited_accel_world + delta
                a_cmd = clamp_norm(a_cmd, float(self.max_accel))

        self._last_limited_accel_world = np.asarray(a_cmd, dtype=float).copy()
        return v_ref, a_cmd


class ComfortAwareSimulation(TiltrotorSimulation):
    """Use kinematic acceleration/jerk after ground constraints are applied."""

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
            jerk = (accel - self._kinematic_previous_accel) / self.dt
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
        if label_prefix is not None and str(
            getattr(model, "label", "")
        ).startswith(label_prefix):
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
    """Build v0.4.2 dashboard while preserving the established flight UI."""
    base_dashboard.WIND_MODE_LABELS.clear()
    base_dashboard.WIND_MODE_LABELS.update({
        "Steady wind": "steady",
        "Discrete gust": "discrete_gust",
        "Single discrete gust": "discrete_gust",
        "Repeated gusts": "repeated_gusts",
        "Continuous turbulence (FAR/CS 25.341-style)": "continuous_turbulence",
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
        "Continuous turbulence (FAR/CS 25.341-style)",
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
    frequency_input.title = "Repeat / wake frequency [Hz]"
    frequency_input.value = 0.25
    amplitude_input = _find_one(
        doc, NumericInput, title="Disturbance amplitude [m/s]"
    )

    accel_input = _find_one(doc, NumericInput, title="Acceleration limit [m/s²]")
    accel_input.title = "Transient accel target [m/s²] (NASA-informed)"
    accel_input.value = 0.50
    jerk_input = _find_one(doc, NumericInput, title="Jerk limit [m/s³]")
    jerk_input.title = "Measured jerk target [m/s³] (NASA-informed)"
    jerk_input.value = 1.50
    rate_input = _find_one(doc, NumericInput, title="Angular-rate limit [deg/s]")
    rate_input.title = "Angular-rate study threshold [deg/s]"

    guard = _find_one(doc, Toggle, label_prefix="Comfort accel guard")

    def set_jerk_guard_value() -> None:
        if hasattr(sim.controller, "max_command_jerk_mps3"):
            sim.controller.max_command_jerk_mps3 = (
                float(jerk_input.value or 1.50)
                if guard.active
                else None
            )

    def apply_jerk_guard(attr, old, new) -> None:
        del attr, old, new
        set_jerk_guard_value()

    def update_wind_control_semantics(attr, old, new) -> None:
        del attr, old, new
        is_continuous = wind_mode.value.startswith("Continuous turbulence")
        if is_continuous:
            amplitude_input.title = "Turbulence RMS sigma [m/s]"
            # v0.4.1 used 5 m/s as a generic disturbance amplitude. For a
            # continuous Gaussian process that value behaved like a very severe
            # standard deviation. Use a transparent 1 m/s operational test
            # default instead; the user may still choose another sigma.
            if float(amplitude_input.value or 0.0) >= 4.5:
                amplitude_input.value = 1.0
            frequency_input.title = "Repeat / wake frequency [Hz] (ignored here)"
            frequency_input.disabled = True
            duration_input.title = "Continuous turbulence duration: until OFF"
            duration_input.disabled = True
        else:
            amplitude_input.title = "Disturbance amplitude [m/s]"
            frequency_input.title = "Repeat / wake frequency [Hz]"
            frequency_input.disabled = False
            duration_input.title = "Single-gust / wake duration [s]"
            duration_input.disabled = False

    guard.on_change("active", apply_jerk_guard)
    jerk_input.on_change("value", apply_jerk_guard)
    wind_mode.on_change("value", update_wind_control_semantics)
    set_jerk_guard_value()
    update_wind_control_semantics(None, None, None)

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
        source=stream_source,
        legend_label="Weighted RMS (Wd/Wk)",
        line_width=2.4,
    )
    iso_plot.line(
        "time_s", "comfort_iso_limit_mps2",
        source=stream_source,
        legend_label="0.315 m/s² target",
        line_width=2.0,
        line_dash="dashed",
    )
    iso_plot.legend.click_policy = "mute"

    explanation = Div(
        text=(
            "<h3>v0.4.2 interpretation</h3>"
            "<b>Comfort guard:</b> the 1.50 m/s³ value is the measured passenger-"
            "motion target. The internal command slew limit uses 60% of that "
            "value to leave headroom for motor/tilt/plant dynamics; measured jerk "
            "remains the KPI. <b>Continuous turbulence:</b> this option is now a "
            "one-dimensional FAR/CS 25.341-style Gaussian Von Karman benchmark "
            "with L=2500 ft (762 m). The amplitude field is RMS turbulence sigma, "
            "not peak gust speed. The default sigma=1 m/s is an engineering test "
            "level, not a certification intensity. FAR/CS limit turbulence "
            "intensities are structural-load design values and should not be "
            "misread as everyday urban operating wind."
        ),
        sizing_mode="stretch_width",
    )

    root = doc.roots[0]
    if isinstance(root, column().__class__):
        root.children.extend([explanation, iso_plot])
    else:
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
