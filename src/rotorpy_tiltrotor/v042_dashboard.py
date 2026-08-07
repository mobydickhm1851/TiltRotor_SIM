"""v0.4.2 operational dashboard corrections.

For the project's low-altitude urban transition demonstration, continuous
vertical turbulence follows the low-altitude alternative described in EASA
CS-AWO wind model No. 1:

* Gaussian / Von-Karman-form turbulence;
* vertical RMS intensity sigma_w = 0.09 U;
* vertical scale L_w = 4.6 m below 9.2 m altitude;
* L_w = 0.5 z for 9.2 < z < 305 m.

For reference, the same CS-AWO section gives sigma=0.15 U and L=183 m for the
horizontal components; those horizontal values must not be applied directly to
the vertical component.  The current benchmark deliberately tests the vertical
component because loss of altitude/lift is the critical transition failure seen
in the dashboard. It is an engineering benchmark, not certification evidence.
"""
from __future__ import annotations

import numpy as np
from bokeh.models import Div, NumericInput, Select

from . import dashboard as base_dashboard
from . import enhanced_dashboard as enhanced


CS_AWO_HORIZONTAL_SIGMA_RATIO = 0.15
CS_AWO_HORIZONTAL_SCALE_M = 183.0
CS_AWO_VERTICAL_SIGMA_RATIO = 0.09
CS_AWO_REFERENCE_AIRSPEED_MPS = 15.0
CS_AWO_COMPONENTS = 256
CS_AWO_MIN_FREQUENCY_HZ = 0.005
CS_AWO_MAX_FREQUENCY_HZ = 1.0
CS_AWO_FADE_IN_S = 3.0
CS_AWO_LABEL = "Continuous turbulence (CS-AWO vertical low-altitude style)"


def cs_awo_vertical_scale_m(reference_altitude_m: float) -> float:
    """Return the low-altitude CS-AWO vertical turbulence scale used here."""
    z = max(0.0, float(reference_altitude_m))
    if z < 9.2:
        return 4.6
    # This project uses the low-altitude model only. Cap at the 305-m boundary
    # rather than extrapolating the formula beyond its stated range.
    return 0.5 * min(z, 305.0)


class LowAltitudeCSAWOWindModel(enhanced.FixedUrbanWindModel):
    """One-dimensional vertical CS-AWO-style low-altitude turbulence.

    The dashboard disturbance input is interpreted as reference mean wind U.
    The vertical turbulence RMS is sigma_w=0.09 U. The spectrum scale is based
    on the configured reference altitude (the target flight altitude), avoiding
    phase discontinuities that would occur if the spectral basis were rebuilt
    continuously as instantaneous altitude changed.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._awo_key = None
        self._awo_omega = np.zeros(0)
        self._awo_amplitude = np.zeros(0)
        self._awo_phase = np.zeros(0)

    def reset(self) -> None:
        super().reset()
        self._awo_key = None
        self._awo_omega = np.zeros(0)
        self._awo_amplitude = np.zeros(0)
        self._awo_phase = np.zeros(0)

    def _ensure_awo_basis(self, cfg) -> None:
        scale_m = cs_awo_vertical_scale_m(cfg.reference_altitude_m)
        key = (
            int(cfg.random_seed),
            CS_AWO_REFERENCE_AIRSPEED_MPS,
            round(scale_m, 6),
            CS_AWO_MIN_FREQUENCY_HZ,
            CS_AWO_MAX_FREQUENCY_HZ,
        )
        if key == self._awo_key:
            return

        frequency_hz = np.logspace(
            np.log10(CS_AWO_MIN_FREQUENCY_HZ),
            np.log10(CS_AWO_MAX_FREQUENCY_HZ),
            CS_AWO_COMPONENTS,
        )
        omega = 2.0 * np.pi * frequency_hz
        reduced = omega / CS_AWO_REFERENCE_AIRSPEED_MPS
        x = 1.339 * scale_m * reduced
        # Von-Karman-form lateral/vertical normalized spectrum. The finite
        # synthesis is normalized below, so cfg controls the desired RMS.
        phi_spatial = (
            scale_m
            / np.pi
            * (1.0 + (8.0 / 3.0) * x * x)
            / np.power(1.0 + x * x, 11.0 / 6.0)
        )
        phi_temporal = phi_spatial / CS_AWO_REFERENCE_AIRSPEED_MPS
        domega = np.gradient(omega)
        amplitude = np.sqrt(np.maximum(0.0, 2.0 * phi_temporal * domega))
        expected_rms = float(np.sqrt(0.5 * np.sum(amplitude * amplitude)))
        if expected_rms > 1e-12:
            amplitude /= expected_rms

        rng = np.random.default_rng(int(cfg.random_seed))
        self._awo_omega = omega
        self._awo_amplitude = amplitude
        self._awo_phase = rng.uniform(
            0.0, 2.0 * np.pi, len(self._awo_omega)
        )
        self._awo_key = key

    def _continuous_turbulence(self, t, elapsed, cfg):
        del t
        if elapsed < cfg.start_time_s:
            return np.zeros(3), False

        self._ensure_awo_basis(cfg)
        local_time = float(elapsed - cfg.start_time_s)
        unit_value = float(np.sum(
            self._awo_amplitude
            * np.cos(self._awo_omega * local_time + self._awo_phase)
        ))
        fade = self._smoothstep01(local_time / CS_AWO_FADE_IN_S)
        horizontal_base = float(np.linalg.norm(
            np.asarray(cfg.base_wind_mps, dtype=float)[:2]
        ))
        reference_mean_wind = max(
            float(cfg.disturbance_amplitude_mps),
            horizontal_base,
        )
        sigma_w = CS_AWO_VERTICAL_SIGMA_RATIO * reference_mean_wind
        vertical_velocity = sigma_w * fade * unit_value
        return np.array([0.0, 0.0, vertical_velocity]), True


def _find(doc, model_type, title=None):
    for model in doc.select({"type": model_type}):
        if title is None or getattr(model, "title", None) == title:
            yield model


def build_dashboard(doc, simulation=None):
    """Build the v0.4.2 low-altitude operational dashboard."""
    enhanced.FixedUrbanWindModel = LowAltitudeCSAWOWindModel
    sim = enhanced.build_dashboard(doc, simulation=simulation)

    wind_mode = next(_find(doc, Select, title="Wind scenario"))
    old_label = "Continuous turbulence (FAR/CS 25.341-style)"
    wind_mode.options = [
        CS_AWO_LABEL if item == old_label else item
        for item in wind_mode.options
    ]
    base_dashboard.WIND_MODE_LABELS[CS_AWO_LABEL] = "continuous_turbulence"
    base_dashboard.WIND_MODE_LABELS.pop(old_label, None)

    amplitude = next(
        model for model in doc.select({"type": NumericInput})
        if model.title in {
            "Disturbance amplitude [m/s]",
            "Turbulence RMS sigma [m/s]",
        }
    )

    def set_low_alt_semantics(attr, old, new):
        del attr, old, new
        if wind_mode.value == CS_AWO_LABEL:
            amplitude.title = "Reference mean wind U [m/s] (vertical sigma_w = 0.09 U)"
            if float(amplitude.value or 0.0) <= 1.1:
                amplitude.value = 5.0
        elif amplitude.title.startswith("Reference mean wind"):
            amplitude.title = "Disturbance amplitude [m/s]"

    wind_mode.on_change("value", set_low_alt_semantics)

    for div in doc.select({"type": Div}):
        if "v0.4.2 interpretation" in str(div.text):
            div.text = (
                "<h3>v0.4.2 interpretation</h3>"
                "<b>Comfort guard:</b> 1.50 m/s³ remains the measured passenger-"
                "motion target; internal command slew uses 60% of the selected "
                "target to leave plant-response headroom. "
                "<b>Continuous turbulence:</b> the dashboard now tests the "
                "CS-AWO low-altitude <i>vertical</i> component: sigma_w=0.09U "
                "and L_w=0.5z between 9.2 and 305 m (therefore L_w=15 m at a "
                "30-m target altitude). Horizontal CS-AWO values are different "
                "(sigma=0.15U, L=183 m) and are not incorrectly reused for the "
                "vertical axis. FAR/CS 25.341 remains a separate transport-"
                "airplane structural-load benchmark."
            )
            break

    set_low_alt_semantics(None, None, None)
    return sim
